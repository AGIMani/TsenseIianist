"""Capture the unified default command, replacing only the final bash dispatch."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
root=Path('/home/hj/pianist')
out=Path(__file__).resolve().parent/'final_launch_command.json'
with tempfile.TemporaryDirectory() as directory:
 stub=Path(directory)/'bash'
 stub.write_text('#!'+sys.executable+'\nimport os,sys,json\nopen(os.environ["ENTRY_CAPTURE"],"w").write(json.dumps(sys.argv[1:],indent=2)+"\\n")\n')
 stub.chmod(0o755)
 subprocess.run(['/bin/bash',str(root/'all_use_this_train.sh')],env={**os.environ,'PATH':directory+':'+os.environ['PATH'],'ENTRY_CAPTURE':str(out)},check=True)
current=json.loads(out.read_text());old=json.loads((root/'logs/palm_acceleration_20261028/final_launch_command.json').read_text())
def canonical(values):
 return [str(Path(v).resolve()) if v.startswith('/') else v for v in values[values.index('train')+1:]]
x,y=canonical(current),canonical(old)
changes=[(x[i-1],a,b) for i,(a,b) in enumerate(zip(x,y)) if a!=b]
assert len(x)==len(y) and changes==[('--tcp-target-acceleration-limit','4.75','1.90')],changes
print('Only TCP acceleration soft threshold changed; no training submitted.')
