import xml.etree.ElementTree as ET, numpy as np, copy
from scipy.spatial.transform import Rotation
p='tsensei_v3_2_pianist/assets/urdf/TsenseI_left_updated_project.urdf'
tr=ET.parse(p); root=tr.getroot()
# Uploaded frame: +X follows the middle finger, +Y spans index->pinky, +Z is opposite project palm direction.
# Project frame: +X is palm direction, +Z follows the middle finger. Origin is the four-hole mounting point.
R=np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
t=np.array([0.0237,0.0065,0.0441380069844])

def fmt(v): return ' '.join(f'{x:.15g}' for x in v)
def transform_origin(o):
 xyz=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' '); rpy=np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')
 xyz2=R@xyz+t
 M=R@Rotation.from_euler('xyz',rpy).as_matrix()
 rpy2=Rotation.from_matrix(M).as_euler('xyz')
 o.attrib['xyz']=fmt(xyz2); o.attrib['rpy']=fmt(rpy2)
for link in root.findall('link'):
 for o in link.findall('.//origin'): transform_origin(o)
for j in root.findall('joint'):
 for o in j.findall('origin'): transform_origin(o)
 a=j.find('axis')
 if a is not None:
  av=R@np.fromstring(a.attrib.get('xyz','0 0 1'),sep=' '); a.attrib['xyz']=fmt(av)
# rota_back: convert the source negative-coordinate interval into a positive interval while preserving the physical direction.
for j in root.findall("joint"):
 if j.attrib.get('name')=='left_hand_thumb_rotaback_joint2':
  lim=j.find('limit'); lim.attrib['lower']='0'; lim.attrib['upper']='1.74'
  a=j.find('axis'); av=-np.fromstring(a.attrib['xyz'],sep=' '); a.attrib['xyz']=fmt(av)
  root.insert(2,ET.Comment('left_hand_thumb_rotaback_joint2 uses positive [0, 1.74] rad; axis sign is reversed to preserve the uploaded negative-coordinate motion direction.'))
# Ensure motor/rota block visual inventory is explicit.
rb=root.find("link[@name='left_hand_thumb_rota_block_link']")
rb.insert(0,ET.Comment('Rota block motor/housing visuals retained from NR_G0040, NR_G0038_housing, NR_G0038_output, NR_G0037, NR_G0003, NR_G0007, and NR_G0011.'))
ET.indent(root,space='  '); tr.write(p,encoding='unicode',xml_declaration=True)
print('written',p)
