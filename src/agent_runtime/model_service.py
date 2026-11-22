"""Separate GPU owner for the autonomous runtime. Scheduler never owns CUDA."""
import json
import os
from time import perf_counter
from .records import fingerprint


class AutonomousModels:
    def __init__(self, config, lock, limits):
        from .loading import load_models
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        self.config, self.limits = config, limits
        self.combined, self.tokenizers = load_models(config,lock)
        self.combined.multi_source_fusion_mode = "sequential"
        self.receiver, self.sender = self.combined.model_list
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        prepared_at=perf_counter()
        self.planner_tokenizer_data=build_token_enforcer_tokenizer_data(self.tokenizers["sender"])
        from .grammar import boundary_token_trie
        self.planner_boundary_trie=boundary_token_trie(self.planner_tokenizer_data)
        self.grammar_vocabulary_setup_seconds=perf_counter()-prepared_at
        self.reset()

    def reset(self):
        self.current, self.versions, self.used_tokens = None, {}, 0
        self.planner_frames=0
        self.method="text_full"
        self.last_planner_stages=[]
        self.component_caches={}

    def budget(self, requested):
        remaining=self.limits["max_generated_tokens_per_parent"]-self.used_tokens
        if remaining < requested:
            raise ValueError("Parent token budget cannot reserve this model call")
        return requested

    def planner(self, messages, schema):
        self.last_planner_stages=[]
        from .transcript import capture_plan
        from .cache_ops import cache_bytes, cache_digest
        from lmformatenforcer import JsonSchemaParser
        from .grammar import build_boundary_aware_prefix_fn
        parser_at=perf_counter()
        native_tools=self.limits.get("planner_interface")=="native_tools"
        tools=None
        if native_tools:
            from .planner_tools import native_tool_input
            messages,tools=native_tool_input(messages,schema,
                feedback=self.limits.get("planner_feedback_mode")=="native_tool_response",
                thinking=self.limits.get('planner_native_thinking',False))
            allowed=None
        else:
            allowed=build_boundary_aware_prefix_fn(self.planner_tokenizer_data,JsonSchemaParser(schema),self.planner_boundary_trie)
        parser_seconds=perf_counter()-parser_at
        began=perf_counter()
        fresh_state=self.limits.get("planner_context_mode")=="fresh_current_state"
        previous=None if fresh_state else self.current
        if fresh_state:
            self.current=None  # Published versions retain their own immutable references.
        self.planner_frames+=1
        self.last_planner_stages=[]
        analysis=None
        analysis_record=None
        thinking=self.limits.get('planner_native_thinking',False)
        if self.limits.get("planner_analysis_max_new_tokens",0):
            from .planning_policy import analysis_messages, ACTION_MESSAGE
            if not fresh_state:
                raise ValueError("Analysis policy requires fresh current state; no implicit history rewrite")
            note_budget=self.limits["planner_analysis_max_new_tokens"]
            self.budget(note_budget+self.limits["planner_max_new_tokens"])
            messages=analysis_messages(messages)
            note_started=perf_counter()
            analysis=capture_plan(self.sender,self.tokenizers["sender"],self.tokenizers["receiver"],
                messages,note_budget,self.config["max_input_tokens"],producer_role="Planner analysis")
            self.used_tokens+=len(analysis.generated_ids)
            analysis_record={"phase":"analysis","output":analysis.output,"input_ids":analysis.prompt_ids,
                "generated_token_ids":analysis.generated_ids,"prefix_ids":analysis.prefix_ids,
                "prompt_text":analysis.prompt_text,"metrics":analysis.metrics,
                "model_wall_seconds":perf_counter()-note_started,"output_tokens":len(analysis.generated_ids),
                "output_constraints":False,"semantic_correctness":"unverified"}
            self.last_planner_stages.append(analysis_record)
            if not analysis.metrics["complete_assistant_turn"] or not analysis.output.strip():
                return {"generation":{**analysis_record,"output_tokens":len(analysis.generated_ids),
                    "parent_used_tokens":self.used_tokens,"stop_reason":"incomplete_analysis",
                    "model_wall_seconds":perf_counter()-began},
                    "error":"Planner analysis incomplete or empty; no action, fallback or fabricated closure"}
            previous=analysis
        t=capture_plan(self.sender,self.tokenizers["sender"],self.tokenizers["receiver"],
            ACTION_MESSAGE if analysis is not None else messages,
            self.budget(self.limits["planner_max_new_tokens"]),self.config["max_input_tokens"],
            prior_prefix_ids=previous.prefix_ids if previous else None,
            prior_cache=previous.source_cache if previous else None,allowed_tokens=allowed,tools=tools,
            thinking=thinking,sampling=self.limits.get('planner_sampling'))
        inference_seconds=perf_counter()-began
        self.used_tokens += len(t.generated_ids)
        if analysis is not None:
            if t.prompt_ids[:len(analysis.prefix_ids)] != analysis.prefix_ids:
                raise RuntimeError("Action did not continue the exact actual analysis token prefix")
            t.planning_input_ids=analysis.prompt_ids
            t.planning_input_text=analysis.prompt_text
            t.analysis_token_range=(len(analysis.prompt_ids),len(analysis.prompt_ids)+len(analysis.generated_ids)-1)
        record={"output":t.output,"input_ids":t.prompt_ids,"generated_token_ids":t.generated_ids,
            "prefix_ids":t.prefix_ids,"prompt_text":t.prompt_text,"metrics":t.metrics,
            "model_wall_seconds":inference_seconds,"stop_reason":"eos" if t.metrics["complete_assistant_turn"] else "token_limit",
            "output_tokens":len(t.generated_ids)+(len(analysis.generated_ids) if analysis else 0),
            "action_output_tokens":None if thinking else len(t.generated_ids),"analysis":analysis_record,
            "analysis_token_range":t.analysis_token_range,"parent_used_tokens":self.used_tokens,
            "producer":"Planner","source_history_reencoded":fresh_state,
            "context_mode":"fresh_current_state" if fresh_state else "autoregressive_continuation",
            "context_frame":self.planner_frames,"current_decision_kv_origin":"actual autoregressive generation on this recorded input",
            "history_text_scope":"runtime task records and last decision may contain old generated text; their old KV is not reused" if fresh_state else "exact original token/cache continuation",
            "grammar_parser_setup_seconds":parser_seconds,"grammar_vocabulary_reused":True,
            "grammar_token_policy":"no output masking" if native_tools else "native cache plus character-validated internal-quote token paths",
            "action_interface":"native_tools" if native_tools else "constrained_action_json",
            "tool_signatures":tools,"action_schema":schema if native_tools else None,
            "planner_feedback_mode":self.limits.get("planner_feedback_mode","state_packet"),
            "rendered_messages":messages if native_tools else None}
        self.last_planner_stages.append({"phase":"action","output":t.output,
            "input_ids":t.prompt_ids,"generated_token_ids":t.generated_ids,"metrics":t.metrics})
        if not t.metrics["complete_assistant_turn"]:
            return {"generation":record,"error":"Planner turn truncated; no action or fabricated closure"}
        try:
            if native_tools:
                from .planner_tools import parse_native_action
                action_text=t.output
                if thinking:
                    from .planner_tools import separate_native_thinking
                    action_text=separate_native_thinking(t,self.tokenizers['sender'])
                    record['analysis_token_range']=t.analysis_token_range
                    record['action_output']=action_text
                    record['action_output_tokens']=len(t.generated_ids)-(t.analysis_token_range[1]-len(t.prompt_ids))-1
                action=parse_native_action(action_text,schema)
            else:
                action=json.loads(t.output)
        except Exception as exc:
            return {"generation":record,"error":f"Planner action validation failed: {type(exc).__name__}: {exc}"}
        self.current=t
        version=None
        if action.get("action")=="delegate":
            name="ctx"+str(len(self.versions)+1)
            audit_at=perf_counter()
            sha=cache_digest(t.source_cache)
            seconds=perf_counter()-audit_at
            version={"id":name,"prefix_sha256":fingerprint(t.prefix_ids),"prefix_tokens":len(t.prefix_ids),
                "context_frame":self.planner_frames,"context_mode":record["context_mode"],
                "source_cache_sha256":sha,"source_cache_bytes":cache_bytes(t.source_cache),
                "integrity_audit_seconds":seconds,"origin":"actual autoregressive Planner dispatch turn",
                "receiver_prefill_seconds":0.0,"fusion_seconds":0.0,"fused":False,
                "permission":"all participants can read this parent task's public shared background"}
            version["actual_analysis_token_range"]=t.analysis_token_range
            self.versions[name]={"transcript":t,"messages":messages,"source_tools":tools,"metadata":version,"native":None,"native_sha256":None}
        return {"action":action,"generation":record,"published_version":version}

    def decode_worker(self, ids, payload, method, budget, command):
        from .kv import generate_worker
        from .planner_tools import separate_native_thinking
        from types import SimpleNamespace
        thinking=self.limits.get('worker_native_thinking',False)
        sampling=self.limits.get('worker_sampling')
        call=json.loads(command).get('call_budget',{}).get('call_index',0)
        seed=self.limits.get('worker_sample_seed',42)+call if sampling is not None else None
        result=generate_worker(self.combined,self.tokenizers,ids,payload,method,budget,sampling=sampling,sample_seed=seed)
        result['native_thinking']=thinking
        if thinking:
            t=SimpleNamespace(generated_ids=result['generated_token_ids'],prompt_ids=ids)
            try:
                if result['stop_reason']!='eos':raise ValueError('Worker native thinking/final response truncated')
                result['action_output']=separate_native_thinking(t,self.tokenizers['receiver'])
                result['analysis_token_range']=t.analysis_token_range
                result['action_output_tokens']=len(t.generated_ids)-(t.analysis_token_range[1]-len(ids))-1
            except ValueError as exc:
                result.update(error=str(exc),action_output=None,action_output_tokens=None)
        return result

    def worker(self, version_id, command):
        if self.limits.get('worker_context_mode') in {'independent_session', 'semantic_session'}:
            return self.independent_worker(version_id, command)
        if self.method != "text_full":
            return self.communication_worker(version_id,command)
        import torch
        from .cache_ops import prefill, cache_digest, cache_bytes
        from .kv import PlanningPayload, generate_worker
        from .transcript import worker_ids
        version=self.versions[version_id]
        t=version["transcript"]
        preparation=None
        budget=self.budget(self.limits["worker_max_new_tokens"])
        from .planner_tools import WORKER_ROLE_SYSTEM
        role_system=WORKER_ROLE_SYSTEM if self.limits.get("planner_interface")=="native_tools" else None
        native_worker=self.limits.get('worker_interface')=='native_tools'
        close_dispatch=bool(role_system and json.loads(command).get('dispatch_receipt'))
        ids=worker_ids(self.tokenizers["receiver"],t.prefix_ids,command,self.config["max_input_tokens"],role_system=role_system,native_tools=native_worker,close_dispatch=close_dispatch,thinking=self.limits.get('worker_native_thinking',False))
        if len(ids)+budget>self.config["max_input_tokens"]:
            raise ValueError("Worker command plus generation exceeds context budget")
        if version["native"] is None:
            preparation={"version":version_id,"status":"started","source_forward_repeated":False,"method":"text_full"}
            version['metadata']['preparation']=preparation
            cache,seconds=prefill(self.receiver,t.prefix_ids)
            version["native"]=cache
            audit_at=perf_counter()
            version["native_sha256"]=cache_digest(cache)
            audit_seconds=perf_counter()-audit_at
            preparation.update({"receiver_prefill_seconds":seconds,
                "receiver_cache_bytes":cache_bytes(cache),"receiver_cache_sha256":version["native_sha256"],
                "integrity_audit_seconds":audit_seconds,"source_forward_repeated":False,"method":"text_full",
                "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"status":"ready"})
            version["metadata"].update(receiver_prefill_seconds=seconds,receiver_cache_bytes=cache_bytes(cache))
        result=self.decode_worker(ids,PlanningPayload(tuple(t.prefix_ids),version["native"]),"text_full_cached",budget,command)
        self.used_tokens += result["output_tokens"]
        result.update(input_ids=ids,command=command,context_version=version_id,
            parent_used_tokens=self.used_tokens,producer="Worker",output_constraints=False,
            current_role_system_suffix=role_system,worker_interface='native_tools' if native_worker else 'json_action',
            dispatch_tool_response_appended=close_dispatch)
        return {"generation":result,"context_preparation":preparation}

    def independent_worker(self, version_id, command):
        from .worker_session import independent_worker_input
        from .cache_ops import prefill, cache_digest, cache_bytes
        from .kv import PlanningPayload
        from .transcript import capture_plan
        if self.method not in {'text_full', 'text_brief'}:
            raise ValueError('Independent Worker C2C has no validated source/receiver position mapping')
        if self.limits.get('worker_interface') != 'native_tools':
            raise ValueError('Independent Worker sessions require Worker-owned native tools')
        version = self.versions[version_id]
        t = version['transcript']
        context_mode = self.limits['worker_context_mode']
        preparation = None
        handoff = version.get('independent_handoff')
        if self.method == 'text_brief' and handoff is None:
            self.budget(self.limits['handoff_max_new_tokens'] + self.limits['worker_max_new_tokens'])
            began = perf_counter()
            preparation = {'version': version_id, 'method': self.method, 'context_mode': context_mode,
                           'status': 'started', 'source_forward_repeated': False}
            version['metadata']['preparation'] = preparation
            message = {'role': 'user', 'content': json.dumps({'invocation_role': 'Planner',
                'operation': 'prepare_worker_handoff',
                'instructions': 'Write a practical concise handoff for the actual delegated tasks above. Include needed background facts, exact file/field identities, rules, reasoning, useful conclusions or repair suggestions, and evidence locations where applicable. You may solve parts when useful. This is handoff text, not another dispatch action. Do not claim operations or tests were executed.'})}
            receipt = json.loads(command).get('dispatch_receipt')
            messages = ([{'role': 'tool', 'name': 'delegate', 'content': json.dumps(receipt, ensure_ascii=False)}]
                        if receipt and self.limits.get('planner_interface') == 'native_tools' else []) + [message]
            brief = capture_plan(self.sender, self.tokenizers['sender'], self.tokenizers['receiver'], messages,
                self.limits['handoff_max_new_tokens'], self.config['max_input_tokens'],
                prior_prefix_ids=t.prefix_ids, prior_cache=t.source_cache, producer_role='Planner handoff')
            self.used_tokens += len(brief.generated_ids)
            preparation['handoff'] = {'output': brief.output, 'input_ids': brief.prompt_ids,
                                     'generated_token_ids': brief.generated_ids, 'metrics': brief.metrics}
            preparation['preparation_wall_seconds_excluding_integrity_audit'] = perf_counter() - began
            if not brief.metrics['complete_assistant_turn']:
                preparation['status'] = 'handoff_truncated'
                raise ValueError('Actual handoff truncated; no fabricated completion')
            handoff = brief.output
            version['independent_handoff'] = handoff
        assembly_started = perf_counter()
        prefix, ids, session = independent_worker_input(self.tokenizers['receiver'], t, command, self.method,
            self.config['max_input_tokens'], handoff=handoff,
            thinking=self.limits.get('worker_native_thinking', False), source_messages=version['messages'],
            source_tools=version.get('source_tools'), source_thinking=self.limits.get('planner_native_thinking', False),
            semantic=self.limits.get('worker_context_mode') == 'semantic_session')
        assembly_seconds = perf_counter() - assembly_started
        budget = self.budget(self.limits['worker_max_new_tokens'])
        if len(ids) + budget > self.config['max_input_tokens']:
            raise ValueError('Independent Worker command plus generation exceeds context budget')
        caches = version.setdefault('independent_caches', {})
        identity = session['cache_identity']
        if identity not in caches:
            if preparation is None:
                preparation = {'version': version_id, 'method': self.method, 'context_mode': context_mode,
                               'status': 'started', 'source_forward_repeated': False}
                version['metadata']['preparation'] = preparation
            began = perf_counter()
            cache, seconds = prefill(self.receiver, prefix)
            audit_at = perf_counter()
            sha = cache_digest(cache)
            audit_seconds = perf_counter() - audit_at
            preparation.update(receiver_prefill_seconds=seconds, receiver_cache_bytes=cache_bytes(cache),
                receiver_cache_sha256=sha, receiver_prefix_sha256=session['receiver_prefix_sha256'],
                receiver_prefix_tokens=len(prefix), cache_identity=identity, integrity_audit_seconds=audit_seconds)
            preparation['preparation_wall_seconds_excluding_integrity_audit'] = (
                preparation.get('preparation_wall_seconds_excluding_integrity_audit', 0.0)
                + assembly_seconds + perf_counter() - began - audit_seconds)
            preparation['status'] = 'ready'
            caches[identity] = {'cache': cache, 'sha256': sha, 'prefix_ids': tuple(prefix)}
            version['metadata'].setdefault('receiver_session_preparations', []).append(preparation)
            version['metadata'].update(method=self.method, worker_context_mode=context_mode, fused=False,
                receiver_prefill_seconds=sum(p['receiver_prefill_seconds'] for p in version['metadata']['receiver_session_preparations']),
                receiver_cache_bytes=sum(cache_bytes(entry['cache']) for entry in caches.values()))
        entry = caches[identity]
        method = 'text_full_cached' if self.method == 'text_full' else 'text_brief_cached'
        result = self.decode_worker(ids, PlanningPayload(tuple(prefix), entry['cache']), method, budget, command)
        self.used_tokens += result['output_tokens']
        result.update(input_ids=ids, command=command, context_version=version_id, parent_used_tokens=self.used_tokens,
            producer='Worker', output_constraints=False, worker_interface='native_tools',
            dispatch_tool_response_appended=False, independent_session=session)
        result['receiver_session_assembly_seconds'] = assembly_seconds
        return {'generation': result, 'context_preparation': preparation}

    def communication_worker(self, version_id, command):
        from .cache_ops import prefill, cache_digest, cache_bytes
        from .kv import PlanningPayload, generate_worker, fuse_context
        from .transcript import capture_plan, worker_ids
        from .communication_probe import state_planning_scope
        from .delegation_contract import SHARED_SYSTEM
        version=self.versions[version_id];t=version["transcript"];preparation=None
        if self.method == "c2c_shared":
            # C2C is only meaningful when the receiver gets its own Worker
            # control boundary.  The legacy JSON path appended a user turn to
            # the Planner transcript and could make cache preparation look
            # like a valid semantic handoff while leaving the Worker in the
            # Planner's role/history.  Refuse that path explicitly.
            if self.limits.get('planner_interface') != 'native_tools':
                raise ValueError('c2c_shared requires native Planner tools so the Worker receives an explicit role boundary')
        if version["native"] is None:
            started=perf_counter()
            preparation={"version":version_id,"method":self.method,"source_forward_repeated":False}
            version["metadata"]["preparation"]=preparation
            preparation["status"]="started"
            if self.method=="c2c_shared":
                cache,seconds=prefill(self.receiver,t.prefix_ids)
                preparation["receiver_prefill_seconds"]=seconds
                scope_at=perf_counter()
                selected,scope=state_planning_scope(t,version["messages"],self.tokenizers["sender"])
                preparation.update(scope=scope,selection_seconds=perf_counter()-scope_at)
                fusion_timings={}
                cache,fusion=fuse_context(self.combined,t.source_cache,cache,token_indices=selected,timings=fusion_timings)
                preparation.update(fusion_seconds=fusion,fusion_timings=fusion_timings)
                prefix=t.prefix_ids
            elif self.method=="text_brief":
                self.budget(self.limits["handoff_max_new_tokens"]+self.limits["worker_max_new_tokens"])
                message={"role":"user","content":json.dumps({"invocation_role":"Planner",
                    "operation":"prepare_worker_handoff",
                    "instructions":"Write a practical concise handoff for the actual delegated tasks above. Include needed background facts, exact file/field identities, rules, reasoning, useful conclusions or repair suggestions, and evidence locations where applicable. You may solve parts when useful. This is handoff text, not another dispatch action. Do not claim operations or tests were executed."})}
                receipt=json.loads(command).get('dispatch_receipt')
                handoff_messages=([{'role':'tool','name':'delegate','content':json.dumps(receipt,ensure_ascii=False)}]
                    if receipt and self.limits.get('planner_interface')=='native_tools' else [])+[message]
                brief=capture_plan(self.sender,self.tokenizers["sender"],self.tokenizers["receiver"],handoff_messages,
                    self.limits["handoff_max_new_tokens"],self.config["max_input_tokens"],
                    prior_prefix_ids=t.prefix_ids,prior_cache=t.source_cache,producer_role="Planner handoff")
                self.used_tokens+=len(brief.generated_ids)
                preparation["handoff"]={"output":brief.output,"input_ids":brief.prompt_ids,
                    "generated_token_ids":brief.generated_ids,"metrics":brief.metrics}
                if not brief.metrics["complete_assistant_turn"]:
                    preparation.update(status="handoff_truncated",preparation_wall_seconds_excluding_integrity_audit=perf_counter()-started)
                    raise ValueError("Actual handoff truncated; no fabricated completion")
                messages=[{"role":"system","content":SHARED_SYSTEM},{"role":"user","content":brief.output}]
                prefix=self.tokenizers["receiver"].apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
                cache,seconds=prefill(self.receiver,prefix)
                preparation["receiver_prefill_seconds"]=seconds
            else:
                raise ValueError("Unknown communication method")
            preparation["preparation_wall_seconds_excluding_integrity_audit"]=perf_counter()-started
            audit_at=perf_counter();sha=cache_digest(cache)
            preparation.update(integrity_audit_seconds=perf_counter()-audit_at,
                               receiver_cache_bytes=cache_bytes(cache),receiver_cache_sha256=sha)
            # Publish only after preparation and identity checks succeeded.
            version.update(native=cache,native_sha256=sha,receiver_prefix=prefix)
            if self.method=="c2c_shared":version["selected"]=selected
            version["metadata"].update(method=self.method,fused=self.method=="c2c_shared",preparation=preparation)
            version["metadata"].update(receiver_prefill_seconds=preparation.get("receiver_prefill_seconds",0.0),
                                       fusion_seconds=preparation.get("fusion_seconds",0.0))
            preparation["status"]="ready"
        prefix=version["receiver_prefix"]
        from .planner_tools import WORKER_ROLE_SYSTEM
        role_system=WORKER_ROLE_SYSTEM if self.limits.get("planner_interface")=="native_tools" else None
        native_worker=self.limits.get('worker_interface')=='native_tools'
        close_dispatch=bool(self.method=='c2c_shared' and role_system and json.loads(command).get('dispatch_receipt'))
        ids=worker_ids(self.tokenizers["receiver"],prefix,command,self.config["max_input_tokens"],role_system=role_system,native_tools=native_worker,close_dispatch=close_dispatch,thinking=self.limits.get('worker_native_thinking',False))
        budget=self.budget(self.limits["worker_max_new_tokens"])
        if len(ids)+budget>self.config["max_input_tokens"]:
            raise ValueError("Worker command plus generation exceeds context budget")
        payload=PlanningPayload(tuple(prefix),version["native"],self.method=="c2c_shared",version.get("selected"))
        result=self.decode_worker(ids,payload,"c2c_shared" if self.method=="c2c_shared" else "text_brief_cached",budget,command)
        self.used_tokens+=result["output_tokens"]
        result.update(input_ids=ids,command=command,context_version=version_id,
                      parent_used_tokens=self.used_tokens,producer="Worker",output_constraints=False,
                      current_role_system_suffix=role_system,worker_interface='native_tools' if native_worker else 'json_action',
                      dispatch_tool_response_appended=close_dispatch)
        if self.method == "c2c_shared":
            result.update(worker_role_boundary='native_worker_system_after_published_prefix',
                          worker_consumed_fused_cache=True,
                          worker_consumed_cache_sha256=version.get('native_sha256'),
                          source_cache_was_reencoded=False)
        return {"generation":result,"context_preparation":preparation}

    def end(self):
        from .cache_ops import cache_digest
        began=perf_counter()
        audits=[]
        for name,version in self.versions.items():
            source=cache_digest(version["transcript"].source_cache)==version["metadata"]["source_cache_sha256"]
            native=version["native"] is None or cache_digest(version["native"])==version["native_sha256"]
            sessions = {key: cache_digest(entry['cache']) == entry['sha256']
                        for key, entry in version.get('independent_caches', {}).items()}
            audits.append({"version":name,"source_unchanged":source,"receiver_unchanged":native and all(sessions.values()),
                           "independent_receiver_caches_unchanged":sessions})
        report={"used_tokens":self.used_tokens,"versions":[v["metadata"] for v in self.versions.values()],
            "cache_audits":audits,"integrity_audit_seconds":perf_counter()-began}
        component_audits={key:cache_digest(entry['cache'])==entry['sha256'] for key,entry in self.component_caches.items()}
        report['component_receiver_cache_audits']=component_audits
        report['integrity_audit_seconds']=perf_counter()-began
        if not all(component_audits.values()):raise RuntimeError('Component receiver cache changed')
        if not all(a["source_unchanged"] and a["receiver_unchanged"] for a in audits):
            raise RuntimeError("Published Planner or receiver cache changed")
        self.reset()
        return report


def gpu_service(connection, config, lock, limits, canceled):
    started=perf_counter()
    try:
        models=AutonomousModels(config,lock,limits)
        import torch,transformers
        connection.send({"type":"ready","pid":os.getpid(),"loading_seconds":perf_counter()-started,
            "grammar_vocabulary_setup_seconds":models.grammar_vocabulary_setup_seconds,
            "torch":str(torch.__version__),"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(),
            "model_service_separate_process":True,"physical_gpu_parallelism":False})
        while True:
            request=connection.recv()
            if request["kind"]=="stop":
                break
            began=perf_counter()
            used_before=models.used_tokens
            torch.cuda.reset_peak_memory_stats()
            allocated_before=torch.cuda.memory_allocated()/2**20
            response={"type":"model_response","request_id":request["id"],"kind":request["kind"],
                "pid":os.getpid(),"queue_seconds":began-request["submitted_at"]}
            preparation_before=models.versions.get(request.get("version"),{}).get("metadata",{}).get("preparation")
            try:
                slot=request.get("worker_slot")
                if slot is not None and canceled[slot]:
                    response.update(canceled=True,result=None)
                elif request["kind"]=="begin":
                    models.reset()
                    models.method=request.get("method","text_full")
                    response["result"]={"ready":True}
                elif request["kind"]=="planner":
                    response["result"]=models.planner(request["messages"],request["schema"])
                elif request["kind"]=="worker":
                    response["result"]=models.worker(request["version"],request["command"])
                elif request['kind']=='component_worker':
                    from .component_model import component_worker
                    response['result']=component_worker(models,request['case'],request['command'],request['adapted'])
                elif request['kind']=='component_train':
                    from .worker_adapter import train_worker_adapter
                    response['result']=train_worker_adapter(models,request)
                elif request["kind"]=="end":
                    response["result"]=models.end()
                else:
                    raise ValueError("Unknown GPU service request")
            except Exception as exc:
                response.update(error={"type":type(exc).__name__,"message":str(exc)},
                                fatal=isinstance(exc,RuntimeError))
                if request["kind"]=="planner":
                    response["actual_planner_stages_before_error"]=models.last_planner_stages
                if request.get("version") in models.versions:
                    preparation_after=models.versions[request["version"]]["metadata"].get("preparation")
                    # A later failed call must not charge an earlier handoff again.
                    if preparation_after is not preparation_before:
                        response["failed_context_preparation"]=preparation_after
                if request['kind']=='component_worker' and getattr(models,'last_component_preparation',None):
                    response['failed_context_preparation']=models.last_component_preparation
                response["parent_used_tokens"]=models.used_tokens
            response["observed_service_wall_seconds"]=perf_counter()-began
            response["generated_tokens_this_request"]=max(0,models.used_tokens-used_before)
            preparation=(response.get("result") or {}).get("context_preparation") or {}
            peak=max(torch.cuda.max_memory_allocated()/2**20,preparation.get("peak_allocated_mib",0))
            response.update(allocated_before_mib=allocated_before,peak_allocated_mib=peak,
                            peak_incremental_mib=peak-allocated_before)
            connection.send(response)
            if response.get("fatal"):
                break
    except (EOFError,BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"type":"service_failed","pid":os.getpid(),"error":str(exc)})
        except (EOFError,BrokenPipeError):
            pass
    finally:
        connection.close()
