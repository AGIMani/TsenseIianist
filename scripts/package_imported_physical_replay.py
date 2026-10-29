#!/usr/bin/env python3
"""Package a completed imported-trajectory physics trace; no model loading."""
from pathlib import Path
import argparse,json,shutil,hashlib,wave,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.package_interactive_replay import install_local_launchers,install_replay_display
from scripts.package_planned_replay import audio_events,render_audio,viewer_annotations,actual_intervals

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    recording=a.recording.resolve();inp=root/'input/TurkishB_2221_best24250';out=a.output.resolve()
    report=json.loads((recording/'replay_report.json').read_text())
    validation=json.loads((recording/'tracking_validation.json').read_text())
    if not validation['passed']:
        raise ValueError('Actual trajectory tracking gate failed; do not package this as the repaired replay')
    model_validation=json.loads((recording/'retained_model_validation.json').read_text())
    if (report.get('diagnostic_only') or model_validation.get('diagnostic_only')
            or not model_validation['geometry_and_contact_fields_preserved']
            or not model_validation['passed'] or abs(report['duration_s']-17.)>1e-9
            or any(w['count'] for w in report['warnings'])
            or report['activated_key_count']==0 or report['num_hand_key_contact_samples']==0):
        raise ValueError('Full duration, retained model, physical contact/activation and warning checks must pass')
    out.mkdir(exist_ok=False)
    assert sha(recording/'model.mjb')==report['model_sha256']
    with np.load(recording/'physical_trace.npz',allow_pickle=False) as z:trace={k:z[k] for k in z.files}
    t=trace['time'];dt=float(trace['physics_dt']);assert np.allclose(t,np.arange(len(t))*dt,atol=1e-10,rtol=0)
    actual=trace['key_qpos']>=trace['key_limits'][:,1];assert np.array_equal(actual,trace['actual_keys'])
    assert not actual[0].any()
    events=json.loads((inp/'trajectory_bundle/note_events.json').read_text())
    import pretty_midi
    midi=inp/'music/turkish_march_B_grace_chords.mid';pm=pretty_midi.PrettyMIDI(str(midi))
    raw=sorted((n.start+.5,n.end+.5,n.pitch) for inst in pm.instruments for n in inst.notes)
    aligned=sorted((e['onset_s'],e['offset_s'],e['pitch']) for e in events)
    assert len(raw)==228 and np.allclose(raw,aligned,atol=1e-9,rtol=0)
    target=np.zeros_like(actual);assignment=np.zeros((len(t),10,88),dtype=np.uint8)
    for e in events:
        assert e['hand'] in ('right','left') and 1<=e['finger']<=5 and 21<=e['pitch']<=108
        m=(t>=e['onset_s']-1e-10)&(t<e['offset_s']-1e-10);k=e['pitch']-21
        target[m,k]=True;assignment[m,(0 if e['hand']=='right' else 5)+e['finger']-1,k]=1
    valid=(t>=.5)&(t<16.5)
    tp=int((actual[valid]&target[valid]).sum());fp=int((actual[valid]&~target[valid]).sum());fn=int((~actual[valid]&target[valid]).sum())
    metrics=dict(scope='this new physical replay, 1000Hz samples, original MIDI interval [0.5,16.5); micro aggregation',tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))
    report['metrics']=metrics;report['original_midi_sha256']=sha(midi);report['physical_trace_sha256']=sha(recording/'physical_trace.npz')
    mocap_pos=np.zeros((len(t),0,3));mocap_quat=np.zeros((len(t),0,4));viewer=dict(physics_dt=dt)
    for k,v in {**{k:trace[k] for k in ('qpos','qvel','ctrl')},'mocap_pos':mocap_pos,'mocap_quat':mocap_quat}.items():viewer['initial_'+k]=v[0];viewer[k]=v[1:]
    np.savez_compressed(out/'physics_trace.npz',**viewer)
    np.savez_compressed(out/'score_annotations.npz',**viewer_annotations(t,assignment))
    edges=audio_events(actual,t);audio=render_audio(edges,float(t[-1]),out/'played.wav');write(out/'audio_events.json',edges)
    with wave.open(str(out/'played.wav')) as w:
        samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
        assert abs(w.getnframes()/w.getframerate()-float(t[-1]))<1e-8
        audio['rms']=float(np.sqrt(np.mean(samples*samples)));assert audio['rms']>0
    shutil.copy2(recording/'model.mjb',out/'model.mjb');shutil.copy2(midi,out/'original.mid')
    install_replay_display(out);install_local_launchers(out)
    for n in ('preview_keys.py','key_activation.py'):shutil.copy2(ROOT/'tsensei_v3_2_pianist'/n,out/n)
    shutil.copy2(recording/'target_rate_validation.json',out/'target_rate_validation.json')
    shutil.copy2(recording/'tracking_validation.json',out/'tracking_validation.json')
    shutil.copy2(recording/'retained_model_validation.json',out/'retained_model_validation.json')
    meta=dict(display_contract='piano_key_colors_only_v1',dependency_versions=report['dependency_versions'],native_arms=False,preparation_prefix_seconds=0,key_activation_fraction=1.,original_midi_offset_s=.5,controller=report['controller'],source_recording=str(recording),model_sha256=report['model_sha256'],physics_trace_sha256=sha(out/'physics_trace.npz'))
    write(out/'replay.json',meta);write(out/'report.json',report);write(out/'audio_report.json',audio)
    comparison=dict(original_events_shifted_to_replay=events,actual_events=actual_intervals(actual,t),metrics=metrics,original_midi_sha256=sha(midi),source_trace_sha256=sha(recording/'physical_trace.npz'),sample_hz=1000,time_origin='full replay; original MIDI shifted +0.5s; final original sample held from 16.983333s to 17s',checkpoint=None)
    write(out/'comparison_events.json',comparison)
    np.savez_compressed(out/'key_comparison.npz',time=t,actual=actual,original=target,key_qpos=trace['key_qpos'],key_limits=trace['key_limits'],key_activation_fraction=1.,hand_key_normal_force=trace['hand_key_normal_force'])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(2,1,figsize=(16,7),sharex=True,sharey=True,layout='constrained')
    def draw(ax,ev,color):
        ax.add_collection(PolyCollection([[(s,p-.4),(e,p-.4),(e,p+.4),(s,p+.4)] for s,e,p in ev],facecolors=color,edgecolors='none'))
    draw(axes[0],raw,'#326fba')
    for mask,color in [(actual&target,'#278c6d'),(actual&~target,'#d54a47'),(~actual&target,'#dfbf56')]:draw(axes[1],actual_intervals(mask,t),color)
    axes[0].set_title('Original MIDI — exact note timing, shifted +0.5 s',loc='left');axes[1].set_title('Current-model physical keys — full key travel activation',loc='left')
    visible=[r[2] for r in raw]+(np.flatnonzero(actual.any(0))+21).tolist()
    for ax in axes:
        ax.set_xlim(0,t[-1]);ax.set_ylim(max(20,min(visible)-2),min(109,max(visible)+2));ax.set_ylabel('MIDI pitch');ax.grid(axis='x',alpha=.15)
        ax.axvspan(0,.5,color='gray',alpha=.1);ax.axvspan(16.5,17,color='gray',alpha=.1)
    axes[1].legend(handles=[Patch(color=c,label=n) for c,n in [('#278c6d','correct'),('#d54a47','extra'),('#dfbf56','missed')]],loc='upper right')
    axes[1].set_xlabel('Replay time (seconds); 1000 Hz physics; original MIDI t = replay t - 0.5 s')
    fig.suptitle(f'Turkish B · current A3.2 + six-DOF wrist · new physical micro-F1 {metrics["f1"]:.4f}\nSource: job 2221 / update 24250 saved positions; no policy checkpoint; original F1 does not apply')
    fig.savefig(out/'actual_vs_original_midi.png',dpi=150);plt.close(fig)
    with (out/'README.txt').open('a') as f:
        f.write('\n本次为当前A3.2模型与驱动下的新物理回放，双侧掌根六自由度；不是包内原录制的物理结果。\n原60Hz实际关节位置映射后保留为参考，1000Hz接触求解，键位从未按原记录或MIDI写入。\n拇指第二轴反号；新增Roll沿局部Z轴，借用当前Pitch驱动Kp=8、阻尼0.5、力矩±5；TZ范围扩至[-.025,.04]m。\n没有策略推理；不应用训练动作层的手指耦合，因为输入已含40个手指实际关节坐标。\n不对参考轨迹再次限速/限加速度或按策略范围裁剪；原生物理关节/执行器/力矩限制保持。实际跟踪误差先验收，结果见tracking_validation.json。\n源首帧手位初始化、初速零、琴键初始静止；原MIDI在回放中偏移+0.5s，总长17s。\n实际接触、按键与音频来自同一次新的物理轨迹。这不是安全验收。\n')
    write(out/'manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(output=str(out),metrics=metrics,audio=audio),ensure_ascii=False))

if __name__=='__main__':main()
