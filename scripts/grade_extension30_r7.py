"""Local offline scoring of a complete or partial frozen extension. Never upload refs."""
import argparse,collections,csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=['c2c_hidden_analysis','c2c_state_hidden','c2c_shared','text_full','text_brief','state_native','c2c_hidden_full','c2c_command']
def norm(s):return re.sub('[^a-z0-9]','',str(s).lower())
def num(x):
 if isinstance(x,bool):return None
 if isinstance(x,(int,float)) and int(x)==x:return int(x)
 if isinstance(x,str) and re.fullmatch(r'\s*-?\d+(?:\.0+)?\s*',x):return int(float(x))
 return None
def report(text):
 text=text.split('</think>')[-1].strip()
 text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
 try:
  obj=json.loads(text)
  return obj.get('report') if isinstance(obj,dict) and obj.get('action')=='report' else None
 except ValueError:return None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('batch',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 manifest=json.loads((ROOT/'data/c2c-core-os/extension30-20261024-r7/manifest.json').read_text(encoding='utf8'))
 refs=json.loads((ROOT/'artifacts/extension30-20261024-r7/references.offline.json').read_text(encoding='utf8'))
 rows=[]
 for c in manifest['cases']:
  path=a.batch/c['case_id']/'result.json';r=json.loads(path.read_text(encoding='utf8')) if path.exists() else {};ref=refs[c['case_id']];fields=ref['expected_fields'];primary=ref['primary_field'];action=(r.get('planner') or {}).get('action') or {};tasks=action.get('tasks') or []
  row={'case_id':c['case_id'],'intended_tasks':c['intended_tasks'],'actual_tasks':len(tasks),'status':r.get('status','missing'),'arms':{}}
  for arm in ARMS:
   data=(r.get('arms') or {}).get(arm) or {};calls=data.get('calls',[data] if data else []);observed={f:[] for f in fields};strict={f:[] for f in fields};details=[]
   original_group_hit=False
   legacy=any(fields[primary] in {int(v) for v in re.findall(r'(?<![\d.])(\d+)(?![\d.])',call.get('output',''))} for call in calls)
   for task,call in zip(tasks,calls):
    command=norm(str(task.get('command',''))+' '+str(task.get('deliverable','')));named=[f for f in fields if norm(f) in command]
    if not named and len(fields)==1 and len(tasks)==1:named=[primary]
    if named==[primary]:
     original_group_hit |= fields[primary] in {int(v) for v in re.findall(r'(?<![\d.])(\d+)(?![\d.])',call.get('output',''))}
    rep=report(call.get('output',''));values={};exact={}
    for f in named:
     value=None;ok=False
     if isinstance(rep,dict):
      matches=[v for k,v in rep.items() if norm(k)==norm(f)]
      if len(matches)==1:value=num(matches[0])
      ok=f in rep and num(rep[f]) is not None
     elif len(named)==1:
      value=num(rep)
      if value is None and isinstance(rep,str):
       ns=re.findall(r'(?<![\w.])-?\d+(?![\w.])',rep)
       if len(ns)==1:value=int(ns[0])
     observed[f].append(value);strict[f].append(ok and value==fields[f]);values[f]=value;exact[f]=ok
    details.append({'named_fields':named,'values':values,'exact_schema':exact,'unknown_task':not named,'error':call.get('arm_error'),'fused_cache_consumed':call.get('worker_consumed_fused_cache')})
   correct={f:bool(v) and all(x==fields[f] for x in v) for f,v in observed.items()};schema={f:bool(v) and all(v) for f,v in strict.items()}
   row['arms'][arm]={'legacy_numeric_hit':legacy,'original_group_numeric_hit':original_group_hit,'primary_numeric_report_correct':correct[primary],'strict_primary_schema_correct':schema[primary],'all_fields_correct':all(correct.values()),'fields_correct':sum(correct.values()),'uncovered_fields':sum(not v for v in observed.values()),'unknown_tasks':sum(x['unknown_task'] for x in details),'actual_generations':sum(bool(x.get('output_tokens')) for x in calls),'calls':details}
  rows.append(row)
 keys=['legacy_numeric_hit','original_group_numeric_hit','primary_numeric_report_correct','strict_primary_schema_correct','all_fields_correct','fields_correct','uncovered_fields','unknown_tasks','actual_generations']
 result={'expected_cases':30,'expected_fields':62,'finished':sum(r['status']!='missing' for r in rows),'complete':(a.batch/'complete.json').exists(),'actual_task_distribution':dict(collections.Counter(r['actual_tasks'] for r in rows)),'scores':{arm:{k:sum(r['arms'][arm][k] for r in rows) for k in keys} for arm in ARMS},'rows':rows,'scope':'All30 denominator. original_group_numeric_hit uses only output of a task uniquely attributed to original A. Copies cannot supply A credit. legacy_numeric_hit is a supplementary union only. Unknown or ambiguous mappings stay in the denominator; missing/pilot is progress only.'}
 result['phase_complete']=result['complete']
 result['complete']=result['phase_complete'] and result['finished']==30 and a.batch.name.endswith('-full')
 a.out.mkdir(exist_ok=True,parents=True);(a.out/'scores.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 with (a.out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['arm']+keys)
  for arm in ARMS:w.writerow([arm]+[result['scores'][arm][k] for k in keys])
 print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
