"""Concrete Linux isolation/recovery check using real hashing processes.

This is an OS component check, not Planner autonomy or a model-quality score.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--child-group',type=Path)
    p.add_argument('--allocate-mb',type=int)
    args=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    source=project/'data/c2c-core-os/public.json'
    if args.child_group:
        (args.child_group/'cgroup.procs').write_text(str(os.getpid()))
        # Allocation happens AFTER cgroup membership; it really is charged here.
        body=bytearray(args.allocate_mb*1024*1024)
        for i in range(0,len(body),4096):body[i]=1
        raw=source.read_bytes()
        for _ in range(250000):hashlib.sha256(raw).digest()
        result=dict(pid=os.getpid(),source_sha256=hashlib.sha256(raw).hexdigest(),
                    allocation_bytes=len(body),cgroup=Path('/proc/self/cgroup').read_text())
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'actual-tool-report.json').write_text(json.dumps(result,indent=2))
        return
    if os.name!='posix' or os.getuid()!=0:
        raise SystemExit('Use preserved WSL project Python as root for temporary delegation.')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    base=Path('/sys/fs/cgroup')/('c2cos-course-check-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    base.mkdir();rows=[]
    uid,gid=project.stat().st_uid,project.stat().st_gid
    os.chown(output,uid,gid)
    try:
        (base/'cgroup.subtree_control').write_text('+cpu +memory +pids')
        coordinator=base/'coordinator';coordinator.mkdir()
        # Coordinator remains outside this group. Only the exact child PIDs enter.
        for label,mb in [('oom',160),('recovered',8)]:
            group=base/label;group.mkdir()
            (group/'memory.max').write_text(str(64*1024*1024))
            (group/'memory.swap.max').write_text('0')
            (group/'cpu.max').write_text('50000 100000')
            (group/'pids.max').write_text('8')
            os.chown(group,uid,gid);os.chown(group/'cgroup.procs',uid,gid);os.chown(base/'cgroup.procs',uid,gid)
            def drop():
                (coordinator/'cgroup.procs').write_text(str(os.getpid()))
                os.setgid(gid);os.setuid(uid)
            began=time.monotonic()
            proc=subprocess.Popen([sys.executable,__file__,'--output-dir',str(output/label),
                '--child-group',str(group),'--allocate-mb',str(mb)],preexec_fn=drop,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            stdout,stderr=proc.communicate(timeout=30)
            row=dict(stage=label,pid=proc.pid,exitcode=proc.returncode,wall_seconds=time.monotonic()-began,
                     memory_events=(group/'memory.events').read_text(),cpu_stat=(group/'cpu.stat').read_text(),
                     memory_peak=(group/'memory.peak').read_text().strip(),pids_current=(group/'pids.current').read_text().strip(),
                     stdout=stdout,stderr=stderr,limits=dict(memory_max=67108864,swap_max=0,cpu_max='50000 100000',pids_max=8))
            rows.append(row)
            (output/'attempts.json').write_text(json.dumps(rows,indent=2))
            if not (group/'cgroup.procs').read_text().strip():group.rmdir()
        recovered=json.loads((output/'recovered/actual-tool-report.json').read_text())
        passed=rows[0]['exitcode']==-9 and 'oom_kill 1' in rows[0]['memory_events'] and rows[1]['exitcode']==0 and recovered['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        (output/'result.json').write_text(json.dumps(dict(passed=passed,scope='Actual cgroup OOM and native-process restart with a real file-hashing deliverable; not model autonomy',rows=rows,recovered_artifact=recovered),indent=2))
        print(json.dumps(dict(passed=passed,rows=[dict(stage=r['stage'],pid=r['pid'],exitcode=r['exitcode'])for r in rows])))
    finally:
        (output/'attempts.json').write_text(json.dumps(rows,indent=2))
        for child in base.iterdir():
            if child.is_dir() and not (child/'cgroup.procs').read_text().strip():child.rmdir()
        if not any(c.is_dir() for c in base.iterdir()):base.rmdir()
        (output/'cleanup.json').write_text(json.dumps(dict(exact_subtree=str(base),removed=not base.exists())))


if __name__=='__main__':main()
