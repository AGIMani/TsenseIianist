"""Per-process memory policy experiment; does not alter the node's sysctls."""
import ctypes,json,os,runpy,sys
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
cpus=sorted(os.sched_getaffinity(0))
if len(cpus)>int(os.environ['SLURM_CPUS_PER_TASK']):raise RuntimeError('Slurm step CPU binding required')
cpu=cpus[0]
node_paths=list(Path(f'/sys/devices/system/cpu/cpu{cpu}').glob('node[0-9]*'))
if len(node_paths)!=1:raise RuntimeError('Ambiguous CPU NUMA node')
node=int(node_paths[0].name[4:])
mask=ctypes.c_ulong(1<<node)
numa=ctypes.CDLL('libnuma.so.1',use_errno=True)
numa.set_mempolicy.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong]
numa.set_mempolicy.restype=ctypes.c_long
if numa.set_mempolicy(2,ctypes.byref(mask),ctypes.sizeof(mask)*8):
    raise OSError(ctypes.get_errno(),'set_mempolicy(MPOL_BIND)')
policy=ctypes.c_int();got=ctypes.c_ulong()
numa.get_mempolicy.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong]
if numa.get_mempolicy(ctypes.byref(policy),ctypes.byref(got),ctypes.sizeof(got)*8,None,0):
    raise OSError(ctypes.get_errno(),'get_mempolicy')
assert policy.value==2 and got.value==mask.value
record=dict(mode='MPOL_BIND',node=node,allowed_cpus=cpus,flags=0,scope='current process and inherited threads; no system setting change')
print('NUMA POLICY',json.dumps(record),flush=True)
if '--output' in sys.argv:
    output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    (output/'numa_policy.json').write_text(json.dumps(record,indent=2)+'\n')
script=sys.argv[1];sys.argv=sys.argv[1:]
sys.path.insert(0,str(Path(script).resolve().parent))
runpy.run_path(script,run_name='__main__')
