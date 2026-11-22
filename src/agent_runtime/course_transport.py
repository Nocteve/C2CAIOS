"""Full-duplex SSH transport; remote KV handles never cross hosts.

Credentials enter once through stdin and are never written to artifacts/config.
Host timing and remote ingress timing are separate clocks.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
from time import perf_counter
import uuid

LIMIT=32*1024*1024


def read_frame(stream):
    line=stream.readline(LIMIT+1)
    if not line:
        raise EOFError("course GPU transport EOF; cache handles invalid")
    if len(line)>LIMIT or not line.endswith('\n'):
        raise ValueError("invalid course frame size")
    return json.loads(line)


def write_frame(stream,value):
    line=json.dumps(value,ensure_ascii=False)+'\n'
    if len(line.encode('utf-8'))>LIMIT:
        raise ValueError("course frame too large")
    stream.write(line);stream.flush()


def serve_stdio(output,session):
    from .course_models import serve
    from .project import ROOT
    protocol=sys.stdout
    sys.stdout=sys.stderr
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    requests=queue.Queue()
    def receive():
        try:
            while True:
                frame=read_frame(sys.stdin)
                if frame['session']!=session:
                    raise ValueError("session mismatch")
                req=frame['payload']
                if 'submitted_monotonic' in req:
                    req['host_submitted_monotonic']=req['submitted_monotonic']
                    req['submitted_monotonic']=perf_counter()
                requests.put(req)
                if req['kind']=='stop':
                    return
        except Exception:
            requests.put({'kind':'stop'})
    class Responses:
        def put(self,value):
            if value.get('kind')=='result':
                raw=(out/value['raw_record']).read_bytes()
                value.update(remote_raw_record_sha256=hashlib.sha256(raw).hexdigest(),
                             remote_raw_record_utf8=raw.decode('utf-8'),remote_output_directory=str(out))
            write_frame(protocol,{'session':session,'payload':value})
    threading.Thread(target=receive,daemon=True).start()
    serve(requests,Responses(),json.loads((ROOT/'configs/cloud.json').read_text()),
          json.loads((ROOT/'configs/cloud.models.lock.json').read_text()),str(out))


def windows_client(config,session):
    import paramiko
    from .project import ROOT
    for stream in (sys.stdin,sys.stdout,sys.stderr):
        stream.reconfigure(encoding='utf-8')
    password=sys.stdin.readline().rstrip('\r\n')
    client=paramiko.SSHClient()
    client.load_host_keys(str(ROOT/config['known_hosts']))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(config['host'],port=config['port'],username=config['user'],password=password,
                       timeout=20,allow_agent=False,look_for_keys=False)
        password=None
        remote=config['project_root']
        command='cd '+shlex.quote(remote)+' && exec .venv/bin/python scripts/run.py --course-gpu-stdio --course-session '+shlex.quote(session)+' --output-dir '+shlex.quote(remote+'/artifacts/correction-20261019/gpu-'+session)
        outgoing,incoming,errors=client.exec_command(command)
        def logs():
            for line in errors:
                sys.stderr.write(line);sys.stderr.flush()
        def send():
            try:
                while True:
                    frame=read_frame(sys.stdin)
                    if frame['session']!=session:
                        raise ValueError('session mismatch')
                    write_frame(outgoing,frame)
                    if frame['payload']['kind']=='stop':
                        return
            except Exception:
                outgoing.channel.shutdown_write()
        threading.Thread(target=logs,daemon=True).start()
        threading.Thread(target=send,daemon=True).start()
        while True:
            frame=read_frame(incoming)
            write_frame(sys.stdout,frame)
            if frame['payload']['kind'] in {'stopped','fatal'}:
                break
        # SSH owner exit status is distinct from receiving a stop request.
        outgoing.channel.settimeout(30)
        status=incoming.channel.recv_exit_status()
        write_frame(sys.stdout,{'session':session,'payload':{'kind':'remote_exit','exitcode':status}})
    finally:
        password=None
        client.close()


def remote_proxy(requests,responses,config,password,output):
    session=uuid.uuid4().hex
    proc=None
    try:
        command=[config['windows_python'],config['windows_project_root']+'\\scripts\\run.py',
                 '--course-ssh-client','--course-remote-config',config['windows_project_root']+'\\configs\\course.remote.json',
                 '--course-session',session,'--output-dir',config['windows_project_root']+'\\artifacts\\correction-20261019']
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,encoding='utf-8',bufsize=1,start_new_session=True)
        proc.stdin.write(password+'\n');proc.stdin.flush();password=None
        sent={}
        def logs():
            with (Path(output)/'remote-stderr.log').open('w',encoding='utf-8') as f:
                for line in proc.stderr:
                    f.write(line);f.flush()
        def send():
            while True:
                req=requests.get()
                if 'id' in req:
                    sent[req['id']]=perf_counter()
                write_frame(proc.stdin,{'session':session,'payload':req})
                if req['kind']=='stop':
                    return
        threading.Thread(target=logs,daemon=True).start()
        threading.Thread(target=send,daemon=True).start()
        while True:
            frame=read_frame(proc.stdout)
            if frame['session']!=session:
                raise ValueError('session mismatch')
            reply=frame['payload']
            if reply.get('kind')=='ready':
                reply.update(session=session,bridge_pid=os.getpid(),ssh_client_pid=proc.pid,
                             deployment='local task cgroups; remote serialized GPU owner; KV stays remote')
            if reply.get('kind')=='result':
                receive=perf_counter()
                raw=reply.pop('remote_raw_record_utf8').encode('utf-8')
                if hashlib.sha256(raw).hexdigest()!=reply['remote_raw_record_sha256']:
                    raise ValueError('raw record hash mismatch')
                (Path(output)/reply['raw_record']).write_bytes(raw)
                reply.update(host_roundtrip_seconds=receive-sent.pop(reply['id']),
                             host_record_write_seconds=perf_counter()-receive)
            responses.put(reply)
            if reply.get('kind')=='remote_exit':
                break
            if reply.get('kind')=='fatal':
                break
        proc.wait(timeout=10)
    except Exception as exc:
        responses.put({'kind':'fatal','error':repr(exc),'session':session,'cache_handles_invalid':True,'remote_owner_exit_confirmed':False})
    finally:
        password=None
        if proc:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
