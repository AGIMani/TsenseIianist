"""CPU trajectory contracts: explicit units, geometry, timing, and source reuse."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from online_control.models import build_scene_manifest, forward_kinematics
from trajectory_compare import data


@pytest.fixture(scope="module")
def manifest():
    current = build_scene_manifest()
    by_id = {item["id"]: item for item in current["models"]}
    return {**current, "models": [by_id["arm_left"], by_id["hand_left"]]}


@pytest.fixture
def native(manifest):
    names = data.joint_names(manifest)
    q = np.asarray([[model["initial_positions"][joint["name"]]
                     for model in manifest["models"] for joint in model["joints"]
                     if joint["type"] != "fixed"]] * 2)
    q[1, 10] += .035
    return np.array([7., 7.025]), q, names


def write_csv(path, time, q, names, unit="rad"):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s"] + [name + "_q_" + unit for name in names])
        writer.writerows(np.column_stack((time, q)))


def test_native_fk_has_correct_five_tips_and_retains_clock(tmp_path, manifest, native):
    time, q, names = native
    path = tmp_path / "position.csv"
    write_csv(path, time, q, names)
    parsed = data.parse_trajectory(path, path.name, manifest)
    assert parsed["curve"]["time_s"] == time.tolist()
    assert parsed["metadata"]["duration_s"] == pytest.approx(.025)
    assert parsed["metadata"]["start_s"] == 7.
    assert parsed["curve"]["frame"] == "piano"
    assert parsed["curve"]["unit"] == "mm"
    assert len(parsed["curve"]["positions"]) == 26
    assert not parsed["metadata"]["hardware_zero_conversion_applied"]
    positions = {"arm_left": dict(zip(names[:7], q[0, :7])), "hand_left": dict(zip(names[7:], q[0, 7:]))}
    frames = forward_kinematics(manifest, positions, use_defaults=False)["hand_left"]
    for finger, _ in data.FINGERS:
        tip = parsed["metadata"]["tip_local_positions_m"][finger + "_tip"]
        transform = np.asarray(frames[tip["link"]])
        expected = (transform[:3, :3] @ tip["xyz"] + transform[:3, 3]) * 1000
        np.testing.assert_allclose(parsed["curve"]["positions"][finger + "_tip"][0], expected, atol=1e-10)
    joint = next(item for item in manifest["models"][1]["joints"] if item["type"] != "fixed")
    np.testing.assert_allclose(parsed["curve"]["positions"][joint["name"]][0],
                               np.asarray(frames[joint["child"]])[:3, 3] * 1000)


def test_explicit_degrees_equal_native_radians(tmp_path, manifest, native):
    time, q, names = native
    rad, deg = tmp_path / "rad.csv", tmp_path / "deg.csv"
    write_csv(rad, time, q, names)
    write_csv(deg, time, np.rad2deg(q), names, "deg")
    a, b = data.parse_trajectory(rad, rad.name, manifest), data.parse_trajectory(deg, deg.name, manifest)
    for name in a["curve"]["positions"]:
        np.testing.assert_allclose(a["curve"]["positions"][name], b["curve"]["positions"][name], atol=1e-10)


@pytest.mark.parametrize("time", [[0, 0], [1, 0], [0, float("nan")]])
def test_time_must_be_finite_strictly_increasing(tmp_path, manifest, native, time):
    _, q, names = native
    path = tmp_path / "invalid.csv"
    write_csv(path, time, q, names)
    with pytest.raises(ValueError, match="递增|NaN"):
        data.parse_trajectory(path, path.name, manifest)


def test_no_implicit_foreign_joint_mapping_or_blank_filling(tmp_path, manifest, native):
    time, q, names = native
    path = tmp_path / "hardware.csv"
    write_csv(path, time, q, ["J1"] + names[1:])
    with pytest.raises(ValueError, match="不推测硬件"):
        data.parse_trajectory(path, path.name, manifest)
    q[1, 17] = np.nan
    write_csv(path, time, q, names)
    with pytest.raises(ValueError, match="NaN"):
        data.parse_trajectory(path, path.name, manifest)


def test_json_requires_explicit_units_and_all_joints(tmp_path, manifest, native):
    time, q, names = native
    path = tmp_path / "native.json"
    contents = {"time_s": time.tolist(), "q": q.tolist(), "joint_names": names}
    path.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match="unit"):
        data.parse_trajectory(path, path.name, manifest)
    contents["unit"] = "rad"
    contents["q"] = q[:, :-1].tolist()
    contents["joint_names"] = names[:-1]
    path.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少关节"):
        data.parse_trajectory(path, path.name, manifest)


def test_npz_object_array_and_unknown_mapping_are_rejected(tmp_path, manifest, native):
    time, q, names = native
    path = tmp_path / "unknown.npz"
    np.savez(path, time=time, q=q)
    with pytest.raises(ValueError, match="推测关节映射"):
        data.parse_trajectory(path, path.name, manifest)
    np.savez(path, time=time, q=q, names=np.array(names, dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        data.parse_trajectory(path, path.name, manifest)


def test_npz_expansion_checked_before_array_loading(tmp_path, manifest, native, monkeypatch):
    time, q, names = native
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, time=time, q=q, names=np.asarray(names))
    monkeypatch.setattr(data, "MAX_NPZ_EXPANDED_BYTES", 100)
    with pytest.raises(ValueError, match="解压数组"):
        data.parse_trajectory(path, path.name, manifest)


def test_named_npz_can_reorder_explicit_channels(tmp_path, manifest, native):
    time, q, names = native
    path = tmp_path / "reverse.npz"
    np.savez(path, time=time, q=q[:, ::-1], names=np.asarray(names[::-1]), unit="rad")
    parsed = data.parse_trajectory(path, path.name, manifest)
    assert parsed["q_names"] == names
    assert parsed["metadata"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_plot_thinning_preserves_endpoints_without_interpolating(tmp_path, manifest, native, monkeypatch):
    _, q, names = native
    time = np.arange(9) / 400
    q = np.tile(q[0], (9, 1))
    path = tmp_path / "thin.csv"
    write_csv(path, time, q, names)
    monkeypatch.setattr(data, "MAX_CHART_SAMPLES", 4)
    parsed = data.parse_trajectory(path, path.name, manifest)
    plotted = parsed["curve"]["time_s"]
    assert plotted[0] == time[0] and plotted[-1] == time[-1]
    assert len(plotted) == 4 and set(plotted).issubset(set(time))
    assert parsed["metadata"]["samples"] == 9
    assert parsed["metadata"]["chart_samples"] == 4


@pytest.mark.skipif(not (data.ROOT / "preview/outputs/rollout_prescribed_comparison_20261007/manifest.json").is_file(),
                    reason="historical example exports are local runtime inputs")
def test_catalog_binds_each_score_and_curve_to_matching_video():
    catalog = data.load_catalog()
    assert len(catalog["slots"]) == 3
    expected_f1 = [.966120566247813, .22096850023507286, .9557834027555271]
    for slot, expected in zip(catalog["slots"], expected_f1):
        result = slot["result"]
        assert result["f1"]["value"] == pytest.approx(expected)
        assert result["metadata"]["samples"] == 3651
        assert result["curve"]["time_s"][-1] == 9.125
        for key in ("video_url", "comparison_url", "trajectory_url"):
            path = Path(catalog["media"][result[key].removeprefix("/media/")])
            assert path.is_file()
        assert data.find_cached_export(result["metadata"]["sha256"], slot["mode"], catalog) is result
        wrong_mode = "piano_only" if slot["mode"] == "robot_dynamics" else "robot_dynamics"
        assert data.find_cached_export(result["metadata"]["sha256"], wrong_mode, catalog) is None
    assert "no_arms=True" in catalog["slots"][0]["result"]["provenance"]["arm"]
    assert catalog["slots"][1]["result"]["provenance"]["robot_dynamic_dofs"] == 0


def test_local_dataset_verifies_media_and_rejects_directory_escape(tmp_path, manifest):
    import copy
    directory = tmp_path / 'export'
    directory.mkdir()
    video = directory / 'video.mp4'
    video.write_bytes(b'verified test media')
    raw = {'metadata': {'sha256': 'source'}, 'curve': {}, 'video_url': '/media/v',
           'f1': None, 'metric_status': '缺少原始MIDI'}
    catalog = {'schema': 'trajectory_compare.catalog.v1', 'manifest': manifest,
               'media': {'v': 'video.mp4'}, 'slots': [{'id': 1, 'result': raw}]}
    path = directory / 'dataset.json'
    def package(value):
        path.write_text(json.dumps(value))
        (directory / 'manifest.json').write_text(json.dumps({'files': {
            'dataset.json': data.fingerprint(path), 'video.mp4': data.fingerprint(video)}}))
    package(catalog)
    loaded = data.load_dataset(path)
    assert loaded['media']['v'] == str(video)
    video.write_bytes(b'altered bytes')
    with pytest.raises(ValueError, match='散列'):
        data.load_dataset(path)
    package(catalog)
    escaped = copy.deepcopy(catalog)
    escaped['media']['v'] = '../outside.mp4'
    package(escaped)
    with pytest.raises(ValueError, match='超出|缺失'):
        data.load_dataset(path)


def standard_csv(path, manifest, native):
    from trajectory_compare.contracts import META
    t,q,names=native
    sdk=np.rad2deg(q.copy())
    for j,n in enumerate(names):
        if n.endswith('_swing_joint'):sdk[:,j]=np.rad2deg(q[:,j]+.35)
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s',*META]+[n+'_q_deg' for n in names])
        w.writerow([0,1,'hardware_sdk_v2',207.090,-150.552,99.158,*sdk[0]])
        w.writerow([.005,'','','','','',*sdk[1]])
    return q,names


def test_standard_sdk_csv_calibrates_fixed_base_and_preserves_input_angles(tmp_path,manifest,native):
    import copy
    before=copy.deepcopy(manifest);path=tmp_path/'standard.csv';q,names=standard_csv(path,manifest,native)
    result=data.parse_trajectory(path,path.name,manifest,return_arrays=True)
    np.testing.assert_allclose(result['_motion']['q'],q,atol=1e-15,rtol=0)
    assert result['metadata']['hardware_zero_conversion_applied']
    new=result['_scene_manifest'];assert manifest==before
    np.testing.assert_allclose(np.array(new['models'][0]['root_transform'])[:3,:3],np.array(manifest['models'][0]['root_transform'])[:3,:3],atol=1e-15,rtol=0)
    f=forward_kinematics(new,{'arm_left':dict(zip(names[:7],q[0,:7]))},False)
    L=np.array(f['arm_left']['Link7_L']);actual=L[:3,3]+L[:3,:3]@np.array([0,-.095,0])
    np.testing.assert_allclose(actual,[.207090,-.150552,.099158],atol=1e-14,rtol=0)
    assert not result['metadata']['base_calibration']['extra_yaw_015_added']


def test_standard_blank_hold_only_after_known_first_frame(tmp_path,manifest,native):
    path=tmp_path/'standard.csv';q,names=standard_csv(path,manifest,native)
    rows=list(csv.reader(path.open()));rows[2][-1]=''
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    result=data.parse_trajectory(path,path.name,manifest,return_arrays=True)
    assert result['metadata']['blank_joint_positions_held']==1
    assert result['_motion']['q'][1,-1]==result['_motion']['q'][0,-1]
    rows[1][-1]=''
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    with pytest.raises(ValueError,match='首帧'):data.parse_trajectory(path,path.name,manifest)


def test_standard_calibration_metadata_cannot_change_mid_motion(tmp_path,manifest,native):
    path=tmp_path/'standard.csv';standard_csv(path,manifest,native)
    rows=list(csv.reader(path.open()));rows[2][3]='300'
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    with pytest.raises(ValueError,match='固定标定'):data.parse_trajectory(path,path.name,manifest)


def test_standard_csv_truncated_row_is_a_validation_error(tmp_path,manifest,native):
    path=tmp_path/'truncated.csv';standard_csv(path,manifest,native)
    lines=path.read_text().splitlines();lines[1]=lines[1].rsplit(',',1)[0]
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='列数'):
        data.parse_trajectory(path,path.name,manifest)


def test_standard_reupload_keeps_already_calibrated_scene_sha(tmp_path,manifest,native):
    path=tmp_path/'standard.csv';standard_csv(path,manifest,native)
    first=data.parse_trajectory(path,path.name,manifest,return_arrays=True)
    again=data.parse_trajectory(path,path.name,first['_scene_manifest'],return_arrays=True)
    assert json.dumps(first['_scene_manifest'],sort_keys=True)==json.dumps(again['_scene_manifest'],sort_keys=True)


def test_sdk_zero_min_mid_and_max_have_confirmed_direction(tmp_path,manifest,native):
    path=tmp_path/'sdk_direction.csv';q,names=standard_csv(path,manifest,native)
    rows=list(csv.reader(path.open()));header=rows[0];data_rows=[]
    for i,deg in enumerate([0.,np.rad2deg(.35),np.rad2deg(.7)]):
        row=rows[1].copy();row[0]=str(i*.005)
        for n in names:
            if n.endswith('_swing_joint'):row[header.index(n+'_q_deg')]=str(deg)
        data_rows.append(row)
    with path.open('w',newline='') as f:csv.writer(f).writerows([header,*data_rows])
    parsed=data.parse_trajectory(path,path.name,manifest,return_arrays=True)
    for i,n in enumerate(names):
        if n.endswith('_swing_joint'):np.testing.assert_allclose(parsed['_motion']['q'][:,i],[-.35,0,.35],atol=1e-15,rtol=0)
    assert parsed['metadata']['swing_mapping']['version']==2
    path.write_text(path.read_text().replace('hardware_sdk_v2','hardware_sdk_v1'))
    with pytest.raises(ValueError,match='已停用新导入'):data.parse_trajectory(path,path.name,manifest)
