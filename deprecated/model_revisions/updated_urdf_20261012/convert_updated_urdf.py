from pathlib import Path
import copy, shutil, xml.etree.ElementTree as ET
src=Path('/home/hj/hand_middle_dip_standard_description/urdf/hand_middle_dip_standard.urdf')
out=Path('/home/hj/pianist/tsensei_v3_2_pianist/assets/urdf/TsenseI_left_updated_project.urdf')
mesh_src=Path('/home/hj/hand_middle_dip_standard_description/meshes')
mesh_dst=out.parent.parent/'meshes_updated'
mesh_dst.mkdir(parents=True,exist_ok=True)
if not mesh_dst.exists() or not any(mesh_dst.rglob('*')):
    shutil.copytree(mesh_src,mesh_dst,dirs_exist_ok=True)
R=ET.parse(src).getroot(); links={x.attrib['name']:x for x in R.findall('link')}; joints={x.attrib['name']:x for x in R.findall('joint')}
# output links: project contract names -> source physical output link
linkmap={'base_link':'hand_base'}
for side, srcfinger in [('index','index'),('mid','middle'),('ring','ring'),('pinky','little')]:
    linkmap[f'left_hand_{side}_MCP_link']=f'{srcfinger}_mcp_carrier'
    linkmap[f'left_hand_{side}_swing_link']=f'{srcfinger}_proximal'
    linkmap[f'left_hand_{side}_PIP_link']=f'{srcfinger}_middle'
    linkmap[f'left_hand_{side}_DIP_link']=f'{srcfinger}_distal'
linkmap.update({'left_hand_thumb_rota_block_link':'thumb_rot1_carrier','left_hand_thumb_rotaback_link2':'thumb_rot2_carrier','left_hand_thumb_MCP_link':'thumb_f2_output','left_hand_thumb_PIP_link':'thumb_f1_output'})
# joint map output -> source joint whose axis/limit/dynamics define motion; origin override source mount or direct
jm={}
for side,sf in [('index','index'),('mid','middle'),('ring','ring'),('pinky','little')]:
    jm[f'left_hand_{side}_MCP_joint']=(f'{sf}_mcp_flexion',f'{sf}_mount_fixed')
    jm[f'left_hand_{side}_swing_joint']=(f'{sf}_mcp_swing',None)
    jm[f'left_hand_{side}_PIP_joint']=(f'{sf}_pip_flexion',None)
    jm[f'left_hand_{side}_DIP_joint']=(f'{sf}_dip_flexion',None)
jm.update({'left_hand_thumb_rota_block_joint':('thumb_rot1','thumb_mount_fixed'),'left_hand_thumb_rotaback_joint2':('thumb_rot2',None),'left_hand_thumb_MCP_joint':('thumb_f2_flexion',None),'left_hand_thumb_PIP_joint':('thumb_f1_flexion',None)})
root=ET.Element('robot',{'name':'TsenseI_A3.2_left_updated_project_compatible'})
# add explanatory metadata as XML comments
root.append(ET.Comment('Converted from hand_middle_dip_standard.urdf. Auxiliary slider/rod joints removed; output coordinates folded into the project 20-DOF contract.'))
root.append(ET.Comment('Root hand_base is renamed base_link. Joint origins and axes remain in the uploaded hand_base coordinate frame; verify global mounting transform before activation.'))
for new,old in linkmap.items():
    e=copy.deepcopy(links[old]); e.attrib['name']=new
    for m in e.findall('.//mesh'):
        fn=m.attrib.get('filename','')
        prefix='package://hand_middle_dip_standard_description/meshes/'
        if fn.startswith(prefix): m.attrib['filename']='../meshes_updated/'+fn[len(prefix):]
    root.append(e)
for new,(motion,mount) in jm.items():
    e=copy.deepcopy(joints[motion]); e.attrib['name']=new
    # project parent/child based on name
    if 'thumb_rota_block' in new: parent,child='base_link','left_hand_thumb_rota_block_link'
    elif 'thumb_rotaback' in new: parent,child='left_hand_thumb_rota_block_link','left_hand_thumb_rotaback_link2'
    elif 'thumb_MCP' in new: parent,child='left_hand_thumb_rotaback_link2','left_hand_thumb_MCP_link'
    elif 'thumb_PIP' in new: parent,child='left_hand_thumb_MCP_link','left_hand_thumb_PIP_link'
    else:
        side=next(s for s in ['index','mid','ring','pinky'] if f'_{s}_' in new)
        stage=new.split('_')[-2] if new.endswith('_joint') else ''
        if '_MCP_joint' in new: parent,child='base_link',f'left_hand_{side}_MCP_link'
        elif '_swing_joint' in new: parent,child=f'left_hand_{side}_MCP_link',f'left_hand_{side}_swing_link'
        elif '_PIP_joint' in new: parent,child=f'left_hand_{side}_swing_link',f'left_hand_{side}_PIP_link'
        elif '_DIP_joint' in new: parent,child=f'left_hand_{side}_PIP_link',f'left_hand_{side}_DIP_link'
        else: raise ValueError(new)
    e.find('parent').attrib['link']=parent; e.find('child').attrib['link']=child
    if mount is not None:
        oldo=joints[mount].find('origin'); o=e.find('origin')
        if oldo is None:
            if o is not None: e.remove(o)
        elif o is None: e.insert(0,copy.deepcopy(oldo))
        else: e.remove(o); e.insert(0,copy.deepcopy(oldo))
    root.append(e)
ET.indent(root,space='  ')
out.write_text(ET.tostring(root,encoding='unicode'))
print(out)
print('links',len(root.findall('link')),'joints',len(root.findall('joint')))
