"""Both current hands use the supplied 952 g inertial data without rescaling."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'tsensei_v3_2_pianist/assets'
spec = importlib.util.spec_from_file_location('sync_hand', ROOT / 'scripts/sync_project_urdf.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.mark.parametrize('side', ['left', 'right'])
def test_supplied_mass_and_inertia_are_preserved_without_hull_estimation(side):
    source = ET.parse(ASSETS/'source/latest_project/urdf'/f'TsenseI_hand_{side}_v3_2_withid.urdf').getroot()
    active = ET.parse(ASSETS/'urdf'/f'TsenseI_{side}.urdf').getroot()
    mass = 0.
    for link in source.findall('link'):
        node = active.find(f"link[@name='{link.get('name')}']")
        original = module.supplied_inertia(link)
        actual = module.supplied_inertia(node)
        mass += actual[0]
        for a,b in zip(original,actual): np.testing.assert_array_equal(a,b)
    assert mass == pytest.approx(.952, abs=1e-12)

def test_missing_or_nonphysical_supplied_inertia_is_rejected():
    link = ET.Element('link',name='bad')
    with pytest.raises(ValueError,match='Missing supplied inertia'): module.supplied_inertia(link)
    inertial=ET.SubElement(link,'inertial')
    ET.SubElement(inertial,'origin',xyz='0 0 0',rpy='0 0 0')
    ET.SubElement(inertial,'mass',value='1')
    ET.SubElement(inertial,'inertia',ixx='1',iyy='1',izz='3',ixy='0',ixz='0',iyz='0')
    with pytest.raises(ValueError,match='Invalid supplied mass/inertia'): module.supplied_inertia(link)
