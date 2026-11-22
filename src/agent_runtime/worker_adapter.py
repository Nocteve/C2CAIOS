"""Small explicit q/v LoRA on the Worker only; pinned base tensors stay frozen."""
import hashlib,json,math
from pathlib import Path
from time import perf_counter
import torch


class LowRankWorkerLinear(torch.nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.scale=alpha/rank;self.enabled=False
        self.lora_A=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        value=self.base(x)
        if self.enabled:
            update=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)
            value=value+(update*self.scale).to(value.dtype)
        return value


def base_digest(model):
    tensors=[]
    for name,value in model.state_dict().items():
        if '.lora_' not in name:tensors.append((name.replace('.base.','.'),value))
    digest=hashlib.sha256()
    for name,value in sorted(tensors,key=lambda item:item[0]):
        digest.update(name.encode());digest.update(str((tuple(value.shape),str(value.dtype))).encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def adapter_enabled(host,enabled):
    for module in getattr(host,'worker_adapter_modules',{}).values():module.enabled=enabled
    host.component_adapter_identity=host.worker_adapter_sha256 if enabled else None


def train_worker_adapter(host,request):
    if getattr(host,'worker_adapter_modules',None):raise ValueError('One fixed training pass only')
    policy=request['policy'];directory=Path(request['directory'])
    directory.mkdir(parents=True,exist_ok=False)
    path=Path(request['training_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=request['training_sha256']:raise ValueError('Frozen training data changed')
    records=json.loads(path.read_text())
    expected_families=set(policy['train_families'])
    if len(records)!=policy['training_examples'] or any(r['family'] not in expected_families for r in records):
        raise ValueError('Training denominator or family isolation changed')
    if host.versions or host.component_caches:raise ValueError('No inference caches may survive training')
    torch.manual_seed(policy['seed']);model=host.receiver;model.requires_grad_(False)
    audit_at=perf_counter();before=base_digest(model);base_audit_seconds=perf_counter()-audit_at
    candidates=[(name,module) for name,module in model.named_modules() if name.endswith(('.self_attn.q_proj','.self_attn.v_proj'))]
    if not candidates:raise ValueError('Pinned Worker has no supported q/v modules')
    count=sum(policy['rank']*(module.in_features+module.out_features) for _,module in candidates)
    (directory/'parameter-preflight.json').write_text(json.dumps({'policy':policy,'trainable_parameters':count,
        'modules':{name:{'in_features':module.in_features,'out_features':module.out_features,
            'A_shape':[policy['rank'],module.in_features],'B_shape':[module.out_features,policy['rank']],
            'A_parameters':policy['rank']*module.in_features,'B_parameters':policy['rank']*module.out_features} for name,module in candidates},
        'base_sha256_before':before,'optimizer_steps':0},indent=2))
    if count>policy['max_trainable_parameters']:raise ValueError(f'Small training allowlist exceeded: {count} parameters')
    modules={}
    for name,module in candidates:
        parent_name,field=name.rsplit('.',1);parent=model.get_submodule(parent_name)
        adapter=LowRankWorkerLinear(module,policy['rank'],policy['alpha']);setattr(parent,field,adapter);modules[name]=adapter
    host.worker_adapter_modules=modules
    parameters=[p for module in modules.values() for p in (module.lora_A,module.lora_B)]
    if sum(p.numel() for p in parameters)!=count:raise RuntimeError('Adapter parameter identity changed after installation')
    encoded=[]
    tokenizer=host.tokenizers['receiver']
    for record in records:
        ids=record['input_ids'];target=record['target_ids']
        if len(ids)+len(target)>policy['max_sequence_tokens']:raise ValueError('Training exceeds frozen context; no truncation')
        if tokenizer.decode(target,skip_special_tokens=False)!=record['target_token_text']:
            raise ValueError('Frozen training target token identity mismatch')
        encoded.append((ids+target,[-100]*len(ids)+target))
    model.train()
    for module in modules.values():module.enabled=True
    optimizer=torch.optim.AdamW(parameters,lr=policy['learning_rate'],weight_decay=0.0)
    began=perf_counter();rows=[];step_logging_seconds=0.0;torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(policy['epochs']):
            # Deterministic order is fixed in the source manifest; no held-out selection.
            for index,(ids,labels) in enumerate(encoded):
                step_at=perf_counter()
                optimizer.zero_grad(set_to_none=True)
                result=model(input_ids=torch.tensor([ids],device='cuda:0'),
                    labels=torch.tensor([labels],device='cuda:0'),use_cache=False)
                loss=result.loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite actual training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,policy['clip_norm']);optimizer.step()
                torch.cuda.synchronize()
                row={'epoch':epoch,'example':index,'loss':float(loss.detach()),'gradient_norm':float(norm),'family':records[index]['family'],
                     'model_step_seconds':perf_counter()-step_at}
                rows.append(row)
                log_at=perf_counter()
                with (directory/'steps.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                step_logging_seconds+=perf_counter()-log_at
                del result,loss
        torch.cuda.synchronize()
        training_seconds=perf_counter()-began
        peak=torch.cuda.max_memory_allocated()/2**20
        model.eval()
        for parameter in parameters:parameter.requires_grad_(False)
        audit_at=perf_counter();after=base_digest(model);base_audit_seconds+=perf_counter()-audit_at
        if before!=after:raise RuntimeError('Pinned Worker base weights changed')
        save_at=perf_counter()
        state={name+'.'+field:getattr(module,field).detach().cpu() for name,module in modules.items() for field in ('lora_A','lora_B')}
        checkpoint=directory/'adapter.pt';torch.save(state,checkpoint)
        host.worker_adapter_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report={'status':'completed','training_examples':len(encoded),'optimizer_steps':len(rows),
            'trainable_parameters':count,'module_names':list(modules),'policy':policy,'adapter_sha256':host.worker_adapter_sha256,
            'base_sha256_before':before,'base_sha256_after':after,'base_unchanged':True,
            'training_seconds':training_seconds,'peak_allocated_mib':peak,'loss_rows':rows,
            'training_model_step_seconds':sum(r['model_step_seconds'] for r in rows),'step_logging_seconds':step_logging_seconds,
            'base_weight_integrity_audit_seconds':base_audit_seconds,'checkpoint_write_hash_seconds':perf_counter()-save_at,
            'teacher_origin':'Offline synthetic integer supervision checked by execution; not original Worker trajectories',
            'planner_updated':False,'fuser_updated':False,'c2c_compatibility_verified':False}
        (directory/'result.json').write_text(json.dumps(report,indent=2))
        adapter_enabled(host,False)
        return report
    finally:
        optimizer.zero_grad(set_to_none=True);model.eval()
        for parameter in parameters:parameter.requires_grad_(False)
