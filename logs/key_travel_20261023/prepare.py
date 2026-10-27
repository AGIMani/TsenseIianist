import hashlib,json,tarfile
from pathlib import Path
root=Path.cwd()
parent=Path('/home/houj/projects/pianist_black_key_20261023/source_black_key_shape_v1.tar.gz')
expected='1951eb00657ad1a82a8bc7092f6274dcf4607f26746cc4dfa4d936f89761dd84'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==expected
runtime=root/'runtime';runtime.mkdir(exist_ok=False)
with tarfile.open(parent) as t:t.extractall(runtime)
for name in ['tsensei_v3_2_pianist/models/piano/piano_constants.py','tests/test_key_activation_fraction.py']:
 (runtime/name).write_bytes((root/'overlay'/name).read_bytes())
for name in ['README.md','AGENTS.md','docs/reward_defaults_20261006.md']:
 p=runtime/name;s=p.read_text().replace('标称约10/8mm','标称12/10mm，2026-10-23更新');p.write_text(s)
p=runtime/'AGENTS.md';p.write_text(p.read_text().replace('logs/black_key_model_20261023/source_black_key_shape_v1.tar.gz','logs/key_travel_20261023/source_key_travel_12_10_v1.tar.gz'))
p=runtime/'docs/black_key_shape_20261023.md';s=p.read_text().replace('关节范围、弹簧、阻尼、armature、键程和100%激活保持','弹簧、阻尼、armature和100%激活保持；2026-10-23后续将白/黑键标称行程改为12/10 mm，关节上限分别为atan(12/150)和atan(10/90) rad');p.write_text(s)
for name in ['training','training_mjlab']:
 (runtime/name).mkdir(exist_ok=True)
 (runtime/name/'.venv').symlink_to('/home/houj/projects/tsensei_v3_2_pianist/'+name+'/.venv')
(root/'parent.json').write_text(json.dumps({'path':str(parent),'sha256':expected},indent=2)+'\n')
