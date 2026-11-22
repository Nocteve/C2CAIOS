"""Run fresh r5 comparison; input revision and all raw history retained."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/c2c-core-os/extension30-20261024-r7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(phase):
 manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'));cases=manifest['cases'];out=ROOT/'outputs'/f'extension30-20261024-r7-{phase}';out.mkdir(exist_ok=False)
 (out/'protocol.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 if phase=='pilot':cases=[next(c for c in cases if c['intended_tasks']==k) for k in (1,2,3)]
 for c in cases:
  assert sha(ROOT/'data/c2c-core-os'/c['public_filename'])==c['sha256']
  cmd=[sys.executable,'scripts/run.py','--c2c-core-rich-hidden','--communication-extension','--arith-scaffold','--max-tasks',str(c['intended_tasks']),'--public-filename',c['public_filename'],'--output-dir',str(out/c['case_id'])]
  with (out/(c['case_id']+'.log')).open('w',encoding='utf8') as log:
   started=time.time();child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'event':'start','case_id':c['case_id'],'pid':child.pid,'time':started})+'\n')
   try:code=child.wait(timeout=1800)
   except subprocess.TimeoutExpired:child.kill();child.wait();code='timeout'
   with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'event':'end','case_id':c['case_id'],'exit_code':code,'seconds':time.time()-started})+'\n')
 (out/'complete.json').write_text(json.dumps({'cases':len(cases),'time':time.time()}))
 print('Completed',phase,len(cases),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','full']);a=ap.parse_args();run(a.mode)
