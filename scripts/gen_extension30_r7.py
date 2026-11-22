"""Offline same-question parallel-number copies; never execute models or read outcomes for scaling."""
import copy,csv,hashlib,json
from pathlib import Path
from extension30_replica_math import scaled_copy,solve
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/c2c-core-os/extension30-20261024-r7'
ART=ROOT/'artifacts/extension30-20261024-r7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def generate():
 prior=ROOT/'data/c2c-core-os/extension30-20261023-r6';m=json.loads((prior/'manifest.json').read_text(encoding='utf8'));prior_refs=json.loads((ROOT/'artifacts/extension30-20261023-r6/references.offline.json').read_text(encoding='utf8'))
 DATA.mkdir(exist_ok=False);ART.mkdir(exist_ok=False);records=[];refs={};ledger=[]
 for rec in m['cases']:
  p=ROOT/'data/c2c-core-os'/rec['public_filename'];src=ROOT/rec['source'];original=json.loads(src.read_text(encoding='utf8'));slug=src.stem.removeprefix('public-probe3c-');family=slug.rsplit('-',1)[0];k=rec['intended_tasks'];dest=DATA/('public-extension30r7-'+slug+'.json');groups=[]
  assert sha(src)==rec['source_sha256']
  if not rec['revised']:
   dest.write_bytes(p.read_bytes());case=json.loads(dest.read_text(encoding='utf8'));refs[case['id']]=prior_refs[case['id']]
  else:
   case=copy.deepcopy(original);bg=case['public_background'];primary=bg['pending_decision']['field'];packet_a={'public_background':copy.deepcopy(bg),'current_state':copy.deepcopy(case['current_state'])};base=solve(packet_a,family);assert base==original['probe']['expected_answer'];fields={primary:base}
   requests=[{'group':'A','target_field':primary,'case_ref':'Original root public_background/current_state','question':bg['pending_decision']['question'],'report_key':primary}]
   copies=[]
   groups.append({'group':'A','factor':1,'target':primary,'offline_value':base})
   for group,factor in [('B',2),('C',3)][:k-1]:
    packet=scaled_copy(original,factor);value=solve(packet,family);target=f'replica_{group.lower()}_value';pd=packet['public_background']['pending_decision'];oldfield=pd['field'];pd['field']=target;pd['controller_note']=pd['controller_note'].replace(oldfield,target)
    copies.append({'group':group,'target_field':target,**packet})
    requests.append({'group':group,'target_field':target,'case_ref':f'parallel_question_copies[group={group}]','question':pd['question'],'report_key':target});fields[target]=value
    groups.append({'group':group,'factor':factor,'target':target,'offline_value':value})
    assert value==(base if family=='gpu-waves' else base*factor)
   bg['parallel_question_copies']=copies
   bg['verification_requests']=requests
   bg['communication_test_harness']={
    'requested_complete_question_groups':k,
    'scope':'Communication-test load: solve the same complete question for independent number groups A/B/C. No intermediate-value checks and no cross-group arithmetic.',
    'group_policy':'A is the original question and original numbers. Each additional group has its own complete public state. Same task IDs in different groups do not share state.',
    'assignment_contract':'A full load has one distinct Worker report per group. For each actual assignment name its exact target_field in command and deliverable, quote the operands from THAT group, and ask for the complete answer under report_key as a JSON numeric literal. Never mix groups or substitute an intermediate value.',
    'measurement':'Record requested and actual coverage separately; original group A is scored independently. Other groups cannot supply a missing A report even if their answers coincide.',
    'system_boundary':'This is a test condition, not the system design. Planner actions remain real generated decisions; self-execution, zero delegation and takeover are legal. Do not host-fill tasks.'}
   case['id']=original['id'].replace('probe3c-','extension30r7-');case['probe']={'family':'parallel_question_development','primary_field':primary};dest.write_text(json.dumps(case,ensure_ascii=False,indent=2)+'\n',encoding='utf8');refs[case['id']]={'primary_field':primary,'expected_fields':fields}
   assert all(bg[key]==val for key,val in original['public_background'].items());assert case['current_state']==original['current_state'] and case['goal']==original['goal'];assert len(requests)==k
  records.append({**rec,'case_id':case['id'],'public_filename':str(dest.relative_to(ROOT/'data/c2c-core-os')).replace('\\','/'),'sha256':sha(dest),'previous_r6_sha256':sha(p)})
  ledger.append({'case':slug,'revised':rec['revised'],'requested_groups':k,'mode':'complete_parallel_question_copies' if rec['revised'] else 'unchanged_preserved_r4_input','groups':groups,'input_sha256':sha(dest)})
 protocol={**m,'protocol':'extension30-r7-complete-parallel-questions','cases':records,'revision_scope':'18 candidate cases use original complete question plus parameter-scaled copies, replacing intermediate checks;12 preserved inputs remain byte-identical. Original A facts/question unchanged. No extra single-Worker condition.','transformation':'Fixed factors2/3, never chosen using outputs. Scale resource quantities and durations; keep counts/percentages; preserve throughput and scale dataset+handshake. Update textual clock/GPU capacity questions. GPU packing geometry scales with capacity so wave answers may coincide.','scoring':'Original-group score must use its assigned output only; also record complete-group reports, actual coverage and total cost. Whole-output union integer matches remain supplementary and cannot recover A credit from B/C.','execution':'Offline prepared only; server remains shut down, no model/remote operation.'}
 (DATA/'manifest.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8');(ART/'references.offline.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf8');(ART/'revision-ledger.offline.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf8')
 with (ART/'question-groups.offline.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['case','requested_groups','group','factor','target','offline_value'])
  for row in ledger:
   for g in row['groups']:w.writerow([row['case'],row['requested_groups'],g['group'],g['factor'],g['target'],g['offline_value']])
 print('Prepared parallel complete questions on18;12 preserved; references local only; no model operation')
if __name__=='__main__':generate()
