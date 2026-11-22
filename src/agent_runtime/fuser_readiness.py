"""One real supervised backward pass; no optimizer, checkpoint or online labels."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .project import ROOT


def differentiable_fusion(combined, source, native, indices):
    """Private dense training branch; keep the published runtime policy exactly."""
    import torch
    from transformers import DynamicCache
    from .kv import _validate_indices, _fusion_mode
    length=native.get_seq_length();_validate_indices(tuple(indices),length)
    if _fusion_mode(combined)!='sequential':raise ValueError('Readiness is pinned to sequential fusion')
    if source.get_seq_length()!=length:raise ValueError('Unaligned cache')
    if any(t.is_inference() or t.requires_grad for cache in (source,native) for pair in cache for t in pair):
        raise ValueError('Training requires ordinary constant source/native caches')
    keep=torch.zeros((1,1,length,1),device='cuda:0',dtype=torch.bool);keep[:,:,list(indices),:]=True
    result=DynamicCache()
    routes=combined.projector_dict[0][1]
    for layer,original in enumerate(native):
        if layer in routes:
            pairs=routes[layer]
            if len(pairs)!=1:raise ValueError('Expected one source per target layer')
            source_layer,projector=pairs[0]
            projected=combined.projector_list[projector](source[source_layer],original)
            pair=tuple(torch.where(keep,y.to(x.dtype),x) for x,y in zip(original,projected))
        else:pair=tuple(t.clone() for t in original)
        result.update(*pair,layer)
    return result


def run_fuser_readiness(output_dir, *, scalar_heads_only=False):
    import torch
    from .models import ModelHost
    from .cache_ops import clone_cache,cache_digest
    from .transcript import worker_ids
    from .diagnostic import commands

    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'data/handoff-candidates-v1'
    selection=json.loads((data/'selection.json').read_text())
    assert hashlib.sha256((data/'public.json').read_bytes()).hexdigest()==selection['public_sha256']
    paper=json.loads((data/'public.json').read_text())[0]
    assert selection['records'][0]['split']=='train' and selection['records'][0]['id']==paper['id']
    config=json.loads((ROOT/'configs/cloud.json').read_text());lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    report=dict(status='running',pid=os.getpid(),paper_id=paper['id'],question_id=paper['questions'][0]['id'],
        optimizer_steps=0,checkpoint_saved=False,scope='training-chain readiness only; fixed QASPER dispatch, not autonomous delegation or performance evaluation',
        models=lock,config=config,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py']})
    events=(out/'events.jsonl').open('x',encoding='utf-8');started=perf_counter();host=None
    def emit(kind,**fields):events.write(json.dumps(dict(type=kind,**fields))+'\n');events.flush()
    try:
        load_at=perf_counter();host=ModelHost(config,lock);report['loading_seconds']=perf_counter()-load_at
        plan=host.prepare(paper,emit,mechanism=True)
        command=commands(paper,plan)[paper['questions'][0]['id']]
        ids=worker_ids(host.tokenizers['receiver'],host.prefix,command,config['max_input_tokens'])
        report.update(command=command,input_ids=ids,source_cache_sha256=host.hashes['source'],native_cache_sha256=host.hashes['native'],
            semantic_indices=list(host.payloads['c2c_shared'].semantic_token_indices),preparation=host.preparation_metrics)
        # The planning and command construction above never read labels. Only
        # this explicit offline supervised phase can access the selected target.
        labels=json.loads((data/'supervision.offline.json').read_text())
        reviewed=json.loads((data/'semantic-review.json').read_text())
        target=next(r for r in labels if r['paper_id']==paper['id'] and r['question_id']==paper['questions'][0]['id'])
        review=next(r for r in reviewed['cases'] if r['paper_id']==paper['id'] and r['question_id']==target['question_id'])
        assert review['target_supported'] and target['target'] and review['target']==target['target']
        report.update(supervision_sha256=hashlib.sha256((data/'supervision.offline.json').read_bytes()).hexdigest(),
                      semantic_review_sha256=hashlib.sha256((data/'semantic-review.json').read_bytes()).hexdigest(),
                      target=target['target'],labels_only_used_after_planning=True)
        tok=host.tokenizers['receiver'];target_ids=tok.encode(target['target'],add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]
        suffix=ids[len(host.prefix):];teacher_ids=suffix+target_ids[:-1]
        if len(host.prefix)+len(teacher_ids)>config['max_input_tokens']:raise ValueError('No training truncation')
        # Inference tensors cannot be saved for backward. Clone OUTSIDE
        # inference_mode, without re-encoding or recomputing the source cache.
        at=perf_counter();source=clone_cache(host.source);native=clone_cache(host.native);torch.cuda.synchronize()
        report['constant_copy_seconds']=perf_counter()-at
        params=list(host.combined.projector_list.named_parameters())
        versions={n:p._version for n,p in params}
        base_versions=[[p._version for p in m.parameters()] for m in host.combined.model_list]
        with torch.no_grad():
            exact=differentiable_fusion(host.combined,source,native,report['semantic_indices'])
            report['eval_cache_exactly_runtime']=all(torch.equal(a,b) for pair,ref in zip(exact,host.payloads['c2c_shared'].receiver_cache) for a,b in zip(pair,ref))
        assert report['eval_cache_exactly_runtime'];del exact
        if scalar_heads_only:
            candidate=json.loads((ROOT/'configs/fuser-adaptation-candidate.json').read_text())
            names=set(candidate['trainable_tensor_names'])
            assert names=={n for n,_ in params if n.split('.')[1] in ('key_scalar_head','value_scalar_head')}
            for name,param in params:param.requires_grad_(name in names)
            report['fixed_gates']=[dict(layer=i,key=float(p.key_gate_logit.detach()),value=float(p.value_gate_logit.detach()))
                                   for i,p in enumerate(host.combined.projector_list)]
        else:
            for projector in host.combined.projector_list:projector.train().requires_grad_(True)
        assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in host.combined.model_list)
        report['training_mode']=('Only exact scalar-head allowlist requires gradients; all modules eval, fixed hard gates, no dropout'
            if scalar_heads_only else 'Only projectors.train(): published Gumbel gates/dropout; base models eval and frozen. Eval hard gates checked separately.')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();at=perf_counter()
        fused=differentiable_fusion(host.combined,source,native,report['semantic_indices'])
        forward=host.receiver(input_ids=torch.tensor([teacher_ids],device='cuda:0'),past_key_values=fused,
            attention_mask=torch.ones((1,len(host.prefix)+len(teacher_ids)),device='cuda:0',dtype=torch.long),
            use_cache=True,logits_to_keep=len(target_ids))
        loss=torch.nn.functional.cross_entropy(forward.logits[0].float(),torch.tensor(target_ids,device='cuda:0'))
        report.update(loss=float(loss.detach()),loss_requires_grad=loss.requires_grad,target_token_ids=target_ids,
            target_tokens=len(target_ids),teacher_forced_input_ids=teacher_ids,
            loss_scope='Only Worker target tokens including EOS; no loss on source planning, background or command')
        loss.backward();torch.cuda.synchronize();report['forward_backward_seconds']=perf_counter()-at
        grads=[dict(name=n,numel=p.numel(),present=p.grad is not None,
            finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,
            nonzero=int(torch.count_nonzero(p.grad)) if p.grad is not None else 0,
            norm=float(p.grad.float().norm()) if p.grad is not None else None) for n,p in params]
        report.update(gradients=grads,trainable_fuser_parameters=sum(p.numel() for _,p in params if p.requires_grad),
            scalar_heads_only=scalar_heads_only,
            base_gradients_absent=all(p.grad is None for m in host.combined.model_list for p in m.parameters()),
            fuser_versions_unchanged=all(p._version==versions[n] for n,p in params),
            base_versions_unchanged=all([p._version for p in m.parameters()]==v for m,v in zip(host.combined.model_list,base_versions)),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        audit_at=perf_counter()
        report['cache_audit']=host.audit()
        report['constant_copies_unchanged']=cache_digest(source)==host.hashes['source'] and cache_digest(native)==host.hashes['native']
        report['final_integrity_seconds']=perf_counter()-audit_at
        assert report['loss_requires_grad'] and report['base_gradients_absent'] and report['base_versions_unchanged'] and report['fuser_versions_unchanged'] and report['constant_copies_unchanged']
        assert all(g['finite'] for g in grads if g['present']) and any(g['nonzero'] for g in grads)
        if scalar_heads_only:
            assert all(not g['present'] for g in grads if g['name'] not in names)
            assert all(g['present'] and g['finite'] for g in grads if g['name'] in names)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if host:
            for projector in host.combined.projector_list:projector.zero_grad(set_to_none=True);projector.eval().requires_grad_(False)
        report['wall_seconds']=perf_counter()-started
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');events.close()
