"""Replay rendering must create key boxes only, including old model metadata."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay_key_highlight',ROOT/'scripts/replay_key_highlight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_all_fingers_color_keys_without_any_fingertip_sites_or_text():
    active=np.zeros(88,dtype=bool)
    keys=SimpleNamespace(keys=np.arange(88),geom_ids=np.arange(88),sizes=np.ones((88,3))*.01,active=lambda d:active)
    data=SimpleNamespace(geom_xpos=np.zeros((88,3)),geom_xmat=np.tile(np.eye(3).ravel(),(88,1)))
    color=module.FingeringHighlight(None,{},keys)
    scene=mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"),maxgeom=100)
    fingers=np.zeros((10,88))
    color.append(scene,data,fingers)
    assert scene.ngeom==0
    for i in range(10):fingers[i,i]=1
    color.append(scene,data,fingers)
    assert scene.ngeom==10
    for i,g in enumerate(scene.geoms[:scene.ngeom]):
        assert g.type==mujoco.mjtGeom.mjGEOM_BOX and not g.label
        np.testing.assert_allclose(g.rgba,module.FINGER_COLORS[i%5])
    # Actual-key green has priority; ambiguous fingering creates no markers.
    active[0]=True;fingers[1,2]=1;scene.ngeom=0
    color.append(scene,data,fingers)
    assert scene.ngeom==8
    assert all(not g.label and g.type==mujoco.mjtGeom.mjGEOM_BOX for g in scene.geoms[:scene.ngeom])


def test_packager_never_copies_legacy_training_display(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('packager',ROOT/'scripts/package_interactive_replay.py')
    packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
    (tmp_path/'scripts').mkdir();(tmp_path/'tsensei_v3_2_pianist').mkdir()
    (tmp_path/'scripts/replay_key_highlight.py').write_bytes((ROOT/'scripts/replay_key_highlight.py').read_bytes())
    (tmp_path/'scripts/interactive_recorded_replay.py').write_text('# current viewer')
    (tmp_path/'tsensei_v3_2_pianist/preview_fingering.py').write_text('old floating label generator')
    folder=tmp_path/'replay';folder.mkdir()
    (folder/'preview_fingering.py').write_text('old cached generator')
    monkeypatch.setattr(packager,'ROOT',tmp_path)
    packager.install_replay_display(folder)
    assert (folder/'preview_fingering.py').read_bytes()==(ROOT/'scripts/replay_key_highlight.py').read_bytes()
    assert (folder/'viewer.py').read_text()=='# current viewer'
