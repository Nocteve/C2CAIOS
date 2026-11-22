"""One frozen-order scalar-head training pass and held-out four-arm handoff replay."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .project import ROOT


def run_fuser_pilot(output_dir):
    import torch
    from .models import ModelHost, PlanningContractError
    from .cache_ops import clone_cache, cache_digest
    from .kv import PlanningPayload, fuse_context
    from .fuser_readiness import differentiable_fusion
    from .transcript import worker_ids
    from .diagnostic import commands
    from .runtime import validate_output
    from .data_partitions import validate_partitions

    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ROOT/'configs/fuser-pilot.json').read_text())
    for name,sha in spec['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    papers={p['id']:p for name in spec['public_files'] for p in json.loads((ROOT/name).read_text())}
    assert validate_partitions(spec['records'],json.loads((ROOT/'configs/data-exclusions.json').read_text()))['partition_valid']
    targets=json.loads((ROOT/spec['training_targets_file']).read_text())
    assert all(t['paper_id'] in spec['train_ids'] for t in targets)
    assert [(t['paper_id'],t['question_id']) for t in targets]==[tuple(x) for x in spec['ordered_targets']]
    report=dict(status='running',pid=os.getpid(),protocol=spec,optimizer_steps=0,training=[],validation=[],
        scope='fixed handoff mechanism only; no autonomous routing, answer successors, OS workers, repairs or network transfer',
        validation_reference_answers_read=False,
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py']})
    for pid in spec['validation_ids']:
        for q in papers[pid]['questions']:
            for method in spec['methods']:
                report['validation'].append(dict(paper_id=pid,question_id=q['id'],method=method,status='not_run'))
    started=perf_counter();events=(out/'events.jsonl').open('x',encoding='utf-8');host=None
    overhead={'weight_and_final_cache_audit_seconds':0.0,'event_logging_seconds':0.0}
    def emit(kind,**fields):
        before=perf_counter()
        events.write(json.dumps(dict(type=kind,**fields))+'\n');events.flush()
        overhead['event_logging_seconds']+=perf_counter()-before
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def weights_digest(items):
        before=perf_counter()
        h=hashlib.sha256()
        for name,p in items:
            h.update(name.encode());h.update(p.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        overhead['weight_and_final_cache_audit_seconds']+=perf_counter()-before
        return h.hexdigest()
    def cache_audit(source=None,native=None):
        before=perf_counter()
        if source is not None:
            assert cache_digest(source)==host.hashes['source'] and cache_digest(native)==host.hashes['native']
        value=host.audit()
        overhead['weight_and_final_cache_audit_seconds']+=perf_counter()-before
        return value
    try:
        load_at=perf_counter();host=ModelHost(json.loads((ROOT/'configs/cloud.json').read_text()),json.loads((ROOT/'configs/cloud.models.lock.json').read_text()))
        report['loading_seconds']=perf_counter()-load_at
        all_params=dict(host.combined.projector_list.named_parameters())
        allow=spec['trainable_tensor_names'];assert set(allow)=={n for n in all_params if n.split('.')[1] in ('key_scalar_head','value_scalar_head')}
        selected=[(n,all_params[n]) for n in allow];frozen=[(n,p) for n,p in all_params.items() if n not in allow]
        assert sum(p.numel() for _,p in selected)==459200
        assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in host.combined.model_list)
        original={n:p.detach().cpu().clone() for n,p in selected}
        frozen_sha=weights_digest(frozen)
        base_versions=[[p._version for p in m.parameters()] for m in host.combined.model_list]
        report.update(original_heads_sha256=weights_digest(selected),frozen_fuser_sha256=frozen_sha,
            gate_values=[dict(layer=i,key=float(p.key_gate_logit),value=float(p.value_gate_logit)) for i,p in enumerate(host.combined.projector_list)])
        masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in selected]
        optimizer=torch.optim.AdamW(masters,lr=spec['learning_rate'],weight_decay=0.0,betas=(0.9,0.999),eps=1e-8)
        for _,p in selected:p.requires_grad_(True)
        assert not any(p.training for p in host.combined.projector_list)
        # One pass only. No validation labels or result-dependent LR/epoch changes.
        for pid in spec['train_ids']:
            selected_targets=[t for t in targets if t['paper_id']==pid]
            if not selected_targets:
                emit('training_parent_without_usable_supervision',paper_id=pid);continue
            parent_at=perf_counter()
            try:plan=host.prepare(papers[pid],lambda k,**v:emit(k,phase='train',paper_id=pid,**v),mechanism=True)
            except PlanningContractError as exc:
                for t in selected_targets:report['training'].append(dict(**t,status='planning_failed',error=str(exc)))
                save();continue
            requests=commands(papers[pid],plan)
            source=clone_cache(host.source);native=clone_cache(host.native)
            positions=host.payloads['c2c_shared'].semantic_token_indices
            if report['optimizer_steps']==0:
                with torch.no_grad():
                    parity=differentiable_fusion(host.combined,source,native,positions)
                    assert all(torch.equal(a,b) for x,y in zip(parity,host.payloads['c2c_shared'].receiver_cache) for a,b in zip(x,y))
                report['zero_step_cache_equals_public_fuser']=True;del parity
            for t in selected_targets:
                row=dict(paper_id=pid,question_id=t['question_id'],status='running',target=t['target'],started_batch_seconds=perf_counter()-started)
                report['training'].append(row);step_at=perf_counter()
                tok=host.tokenizers['receiver'];ids=worker_ids(tok,host.prefix,requests[t['question_id']],host.config['max_input_tokens'])
                answer=tok.encode(t['target'],add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]
                suffix=ids[len(host.prefix):];inputs=suffix+answer[:-1]
                assert len(host.prefix)+len(inputs)<=host.config['max_input_tokens']
                optimizer.zero_grad(set_to_none=True)
                for _,p in selected:p.grad=None
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();grad_at=perf_counter()
                fused=differentiable_fusion(host.combined,source,native,positions)
                prediction=host.receiver(input_ids=torch.tensor([inputs],device='cuda:0'),past_key_values=fused,
                    attention_mask=torch.ones((1,len(host.prefix)+len(inputs)),device='cuda:0',dtype=torch.long),
                    use_cache=True,logits_to_keep=len(answer))
                loss=torch.nn.functional.cross_entropy(prediction.logits[0].float(),torch.tensor(answer,device='cuda:0'))
                loss.backward();torch.cuda.synchronize();grad_seconds=perf_counter()-grad_at
                assert all(p.grad is None for m in host.combined.model_list for p in m.parameters())
                assert all(p.grad is None for _,p in frozen)
                for (_,p),master in zip(selected,masters):
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all())
                    master.grad=p.grad.detach().float().clone()
                norm=float(torch.nn.utils.clip_grad_norm_(masters,spec['gradient_clip']))
                if not torch.isfinite(loss) or not __import__('math').isfinite(norm):raise RuntimeError('Nonfinite training gradient')
                before=[p.detach().clone() for _,p in selected];optimizer.step()
                with torch.no_grad():
                    for (_,p),master in zip(selected,masters):p.copy_(master.to(p.dtype))
                changed=sum(int(torch.count_nonzero(p!=b)) for (_,p),b in zip(selected,before))
                report['optimizer_steps']+=1
                row.update(status='updated',step=report['optimizer_steps'],loss=float(loss.detach()),gradient_norm_before_clip=norm,
                    deployment_bf16_changed_elements=changed,source_sha256=host.hashes['source'],native_sha256=host.hashes['native'],
                    prefix_ids=host.prefix,command=requests[t['question_id']],input_ids=ids,target_token_ids=answer,
                    forward_backward_seconds=grad_seconds,wall_seconds=perf_counter()-step_at,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                emit('optimizer_step',**row);save();del fused,prediction,loss,before
            emit('training_parent_end',paper_id=pid,audit=cache_audit(source,native),wall_seconds=perf_counter()-parent_at)
            del source,native
            print('trained parent',pid,'steps',report['optimizer_steps'],flush=True)
        assert report['optimizer_steps']<=spec['max_updates']
        trained={n:p.detach().cpu().clone() for n,p in selected}
        report['trained_heads_sha256']=weights_digest(selected)
        report['total_changed_bf16_elements']=sum(int(torch.count_nonzero(trained[n]!=original[n])) for n in allow)
        assert weights_digest(frozen)==frozen_sha
        assert all([p._version for p in m.parameters()]==v for m,v in zip(host.combined.model_list,base_versions))
        report['training_frozen_parameters_unchanged']=True
        torch.save(dict(heads=trained,model_pins=json.loads((ROOT/'configs/cloud.models.lock.json').read_text()),protocol=spec),out/'last-heads.pt')
        report['checkpoint_sha256']=hashlib.sha256((out/'last-heads.pt').read_bytes()).hexdigest()
        for _,p in selected:p.grad=None;p.requires_grad_(False)
        del optimizer,masters

        def install(values):
            with torch.no_grad():
                for name,p in selected:p.copy_(values[name].to(device=p.device,dtype=p.dtype))
        for parent_index,pid in enumerate(spec['validation_ids']):
            paper=papers[pid];install(original);parent_at=perf_counter()
            try:plan=host.prepare(paper,lambda k,**v:emit(k,phase='validation',paper_id=pid,**v))
            except PlanningContractError as exc:
                for row in report['validation']:
                    if row['paper_id']==pid:row.update(status='planning_failed',error=str(exc))
                save();continue
            requests=commands(paper,plan)
            original_payload=host.payloads['c2c_shared'];original_hash=host.hashes['fused']
            install(trained)
            adapted_cache,adapt_seconds=fuse_context(host.combined,host.source,host.native,token_indices=original_payload.semantic_token_indices)
            adapted=PlanningPayload(tuple(host.prefix),adapted_cache,fused=True,semantic_token_indices=original_payload.semantic_token_indices)
            adapted_hash=cache_digest(adapted_cache)
            order=spec['methods'][parent_index%4:]+spec['methods'][:parent_index%4]
            for method in order:
                is_adapted=method=='c2c_adapted';install(trained if is_adapted else original)
                host.payloads['c2c_shared']=adapted if is_adapted else original_payload
                host.hashes['fused']=adapted_hash if is_adapted else original_hash
                mapped='c2c_shared' if is_adapted else method
                for q in paper['questions']:
                    row=next(r for r in report['validation'] if r['paper_id']==pid and r['question_id']==q['id'] and r['method']==method)
                    row.update(command=requests[q['id']],source_sha256=host.hashes['source'],native_sha256=host.hashes['native'],
                        prefix_ids=host.prefix,head_weights_sha256=report['trained_heads_sha256'] if is_adapted else report['original_heads_sha256'])
                    if mapped not in host.payloads:
                        row.update(status='handoff_failed',error='Actual text handoff incomplete');continue
                    request_at=perf_counter();row.update(status='running',started_batch_seconds=request_at-started)
                    try:
                        generation=host.infer(mapped,requests[q['id']],budget=96);row['generation']=generation
                        row['value']=validate_output(generation,dict(kind='locate',valid_ids=[p['id'] for p in paper['paragraphs']],output_contract='evidence_ids_v2'))
                        row['status']='delivered'
                    except (ValueError,KeyError,TypeError) as exc:row.update(status='failed',error=str(exc))
                    except Exception as exc:
                        row.update(status='execution_failed',error=dict(type=type(exc).__name__,message=str(exc)));raise
                    finally:
                        row['request_wall_seconds']=perf_counter()-request_at
                        emit('validation_output',**row)
                assert cache_audit()['unchanged']
            emit('validation_parent_end',paper_id=pid,planning_seconds=host.planning_seconds,preparation=host.preparation_metrics,
                preparation_costs=host.costs,adapted_fusion_seconds=adapt_seconds,order=order,wall_seconds=perf_counter()-parent_at)
            save();print('validated parent',pid,flush=True)
        assert weights_digest(frozen)==frozen_sha
        assert all([p._version for p in m.parameters()]==v for m,v in zip(host.combined.model_list,base_versions))
        install(original);report['original_heads_restored']=weights_digest(selected)==report['original_heads_sha256']
        report['status']='completed'
    except Exception as exc:
        for row in report['training']+report['validation']:
            if row.get('status')=='running':
                row.update(status='execution_failed',error=dict(type=type(exc).__name__,message=str(exc)),
                    elapsed_seconds=perf_counter()-started-row['started_batch_seconds'])
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        report['batch_wall_seconds']=perf_counter()-started;report['overheads']=overhead
        report['timing_scope']='Batch includes loading, real planning/handoff, receiver prefill, fusion, training, validation, hashes and output logging. Host preparation integrity costs are separately in events and are additional to these final/weight audits. No OS workflow or network measurement.'
        save();events.close()
