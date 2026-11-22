"""Ledger capability tasks on the existing real delegated Worker lifecycle."""
import json,os
from pathlib import Path
from time import perf_counter
from .records import fingerprint
from .delegation_contract import worker_command
from .worker_tools import worker_schema
from .planner_tools import parse_native_action
from .ledger_contract import ledger_changes,ledger_check


def ledger_worker(connection,task,case,limits,directory):
    observations=[]
    stage='worker_setup'
    try:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        (directory/'input.json').write_text(json.dumps(task,indent=2))
        connection.send({'type':'started','pid':os.getpid()})
        for call in range(limits['worker_max_model_calls']):
            remaining=limits['worker_max_model_calls']-call
            connection.send({'type':'model_request','call':call,
                'command':worker_command(task,observations,call_index=call,remaining_calls=remaining),
                'requested_at':perf_counter()})
            stage='generation_transport';generation=connection.recv()
            if generation.get('error'):raise ValueError(generation['error'])
            stage='model_output'
            if generation.get('stop_reason')!='eos':raise ValueError('Worker turn incomplete')
            value=parse_native_action(generation['output'],worker_schema(task['files'],terminal_only=remaining==1,fields=case['mutable_fields']))
            if 'changes' in value: ledger_changes(case,task['snapshot']['files'],value['changes'],task['files'])
            if value['action'] in {'report','need_info'}:
                artifact={'producer':task['id'],'base_version':task['snapshot']['version'],'base_sha256':task['snapshot']['sha256'],
                    'input_artifacts':task['dependency_artifacts'],'context_version':task['context_version'],
                    'value':value,'semantic_verified':False,'output_validation':{'native_contract_passed':True,'output_masking':False}}
                artifact['sha256']=fingerprint(artifact)
                (directory/'artifact.json').write_text(json.dumps(artifact,indent=2))
                connection.send({'type':'result','artifact':artifact,'pid':os.getpid(),'model_calls':call+1});return
            stage='public_tool';began=perf_counter()
            if value['action']=='inspect': result={'version':task['snapshot']['version'],'files':task['snapshot']['files']}
            else:
                candidate=ledger_changes(case,task['snapshot']['files'],value['changes'],task['files'])
                (directory/f'candidate-{call}.json').write_text(json.dumps(candidate,indent=2))
                result=ledger_check(case,candidate)
            observation={'worker_action':value,'tool_result':result};observations.append(observation)
            connection.send({'type':'tool_result',**observation,'seconds':perf_counter()-began})
        stage='model_output';raise ValueError('Actual Worker call budget exhausted')
    except Exception as exc:
        try:connection.send({'type':'failed','error':f'{type(exc).__name__}: {exc}','pid':os.getpid(),'failure_stage':stage})
        except (BrokenPipeError,EOFError,OSError):pass
    finally:connection.close()


def component_input(tokenizer,case,command):
    from .planner_tools import WORKER_ROLE_SYSTEM
    from .worker_tools import native_worker_input
    if set(case)!={'id','family','goal','facts','files','mutable_fields','rules'}:
        raise ValueError('Only the complete known public business fields may enter Worker inputs')
    messages,tools,_=native_worker_input(command,fields=case['mutable_fields'])
    background={'source':'Complete public receipt task; no Planner generation or KV exists for this capability input','case':case,
                'expression_language':'ref follows public facts paths; sum adds items; multiply/subtract/floor_div/min/max use left and right; if_ge chooses then when left>=right otherwise else; sum_where sums field of records whose where.field equals where.equals. All amounts are integer toy units.'}
    base=[{'role':'system','content':WORKER_ROLE_SYSTEM},{'role':'user','content':json.dumps(background,ensure_ascii=False)}]
    options={'tools':tools,'enable_thinking':False}
    prefix=tokenizer.apply_chat_template(base,tokenize=True,add_generation_prompt=False,**options)
    complete=[*base,*messages]
    rendered=tokenizer.apply_chat_template(complete,tokenize=False,add_generation_prompt=True,**options)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    if ids[:len(prefix)]!=prefix:raise ValueError('Native component prefix mismatch')
    return prefix,ids,{'public_case_sha256':fingerprint(case),'source_kv_transferred':False,'planner_generated':False,
        'input_origin':'independent public component task, not autonomous delegation','rendered_messages':complete,
        'rendered_input':rendered,'worker_tools':tools,'prefix_sha256':fingerprint(prefix)}
