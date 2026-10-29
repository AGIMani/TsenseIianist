"""Resumeable bounded-validation result transfer; no training or new job submission."""
import hashlib,json,subprocess,time,os,fcntl
from pathlib import Path
root=Path(__file__).resolve().parent
lock=(root/'transfer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
job='10529';remote='/home/houj/projects/pianist_pitch_range_20261028'
(root/'transfer.pid').write_text(str(os.getpid())+'\n')
while True:
 try:
  r=subprocess.run(['ssh','-o','ConnectTimeout=15','volc-a32','sacct -n -P -j '+job+' --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS'],capture_output=True,text=True,timeout=30)
  r.check_returncode();rows=[line.split('|') for line in r.stdout.splitlines() if line]
  main=next((x for x in rows if x[0]==job),None)
  status={'job_id':job,'checked_at':time.time(),'sacct':rows,'transfer_complete':False}
  print(json.dumps(status),flush=True)
  (root/'sync_status.json').write_text(json.dumps(status,indent=2)+'\n')
  if main and main[1] not in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
   dest=root/'acceptance';dest.mkdir(exist_ok=True)
   for origin in ('acceptance/', 'project/logs/hpc/'+job+'.out', 'project/logs/hpc/'+job+'.err',
                  'project/logs/hpc/'+job+'/gpu_probe.json','project/logs/hpc/'+job+'/gpu_utilization.csv'):
    subprocess.run(['rsync','-a','volc-a32:'+remote+'/'+origin,str(dest)+'/'],check=True,timeout=120)
   files=[p.name for p in dest.iterdir() if p.is_file()]
   # Hash remote and local copies independently; report fields never imply success themselves.
   payload='import hashlib,json\nfrom pathlib import Path\nr=Path('+repr(remote)+')\n'
   payload+='paths=list((r/"acceptance").glob("*"))+[r/"project/logs/hpc/10529.out",r/"project/logs/hpc/10529.err",r/"project/logs/hpc/10529/gpu_probe.json",r/"project/logs/hpc/10529/gpu_utilization.csv"]\n'
   payload+='print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}))\n'
   h=subprocess.run(['ssh','volc-a32','python3 -'],input=payload,capture_output=True,text=True,check=True,timeout=30)
   expected=json.loads(h.stdout)
   for name,digest in expected.items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest,name
   (root/'transfer_manifest.json').write_text(json.dumps(expected,indent=2)+'\n')
   report=json.loads((dest/'acceptance.json').read_text())
   status.update(transfer_complete=True,sha256_verified=True,validation_passed=main[1]=='COMPLETED' and main[2]=='0:0' and report.get('passed') is True)
   (root/'sync_status.json').write_text(json.dumps(status,indent=2)+'\n')
   print(json.dumps(status),flush=True);break
 except Exception as error:
  print(repr(error),flush=True)
 time.sleep(60)
