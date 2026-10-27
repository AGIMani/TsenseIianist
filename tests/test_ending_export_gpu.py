"""Slurm integration fixture: no training, two-note score, constant hover actor."""
import os
import unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'),'Slurm GPU required')
class EndingExportChecks(unittest.TestCase):
    def test_terminal_export_audio_observer_and_package(self):
        import sys,json,copy,subprocess
        import numpy as np
        import torch
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
        from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
        from tsensei_v3_2_pianist.rl_torch.sac import SAC,SACConfig
        from tsensei_v3_2_pianist.rl_torch.checkpoint import save
        out=root/'results/ending_export_fixture';out.mkdir(parents=True)
        fixture_code="""
from note_seq.protobuf import music_pb2
import sys,copy
seq=music_pb2.NoteSequence();seq.ParseFromString(open(sys.argv[1],'rb').read())
notes=[copy.deepcopy(n) for n in sorted(seq.notes,key=lambda n:n.start_time)[:2]]
del seq.notes[:]
for n in notes:
    n.end_time=n.start_time+.1;seq.notes.add().CopyFrom(n)
seq.total_time=max(n.end_time for n in seq.notes)
open(sys.argv[2],'wb').write(seq.SerializeToString())
"""
        subprocess.run([str(root/'training/.venv/bin/python'),'-c',fixture_code,
            str(root/'training_mjlab/inputs/turkish_march_advanced/a32_contact_refined_20261024/turkish-march-advanced_fingered.proto'),str(out/'fixture.proto')],check=True)
        recipe=make_recipe(height=.09,ty=.04,yaw=.2617993877991494,control_hz=20,physics_hz=1000,
            midi=str(out/'fixture.proto'),initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',
            task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',
            preparation_mode='first_note_v1',preparation_time=2,preparation_lift=0,ending_time=2)
        scene,task,_=build(recipe);scene,task=without_arms(scene,task,scene.parent/'no_arms')
        cfg=effective_config(parser().parse_args(['--no-arms','--disable-wrist-tz','--disable-wrist-yaw',
            '--action-profile','direct_joint_v1','--finger-coupling','--no-coupling-reduce-observation',
            '--finger-coupling-angle-ranges',str(root/'configs/finger_coupling/angle_ranges_10_80.json'),
            '--preparation-mode','first_note_v1','--preparation-time','2','--preparation-lift','0','--ending-time','2',
            '--reward-config',str(root/'configs/rewards/default_20261006.json')]))
        options={k:cfg[k] for k in ('preparation','finger_coupling','control_limits','reward_config','action_profile')}
        env=PianoTaskEnv(scene,task,device='cuda:0',physics_graph=False,disable_wrist_tz=True,disable_wrist_yaw=True,**options)
        metadata=dict(cfg,source_runtime=recipe,scene_sha256=sha256(scene/'manifest.json'),task_sha256=sha256(task/'task.json'),
            preparation_contract=env.scene.bundle.preparation,control_limits_contract=env.scene.control_limits,
            finger_clearance_contract=env.scene.finger_clearance,finger_coupling_contract=env.scene.finger_coupling,
            coupling_observation_contract=env.coupling_observation_contract,arm_follow_contract=None,
            action_contract=env.scene.action_contract,validation_fixture='untrained constant hover; not a training result')
        agent=SAC(env.observation_size,env.action_size,SACConfig(**cfg['sac']),device='cuda:0',action_profile=cfg['action_profile'])
        action=env.scene.policy_action(env.scene.control_reset_action)
        with torch.no_grad():
            for parameter in agent.actor.parameters():parameter.zero_()
            agent.actor.mean.bias.copy_(torch.atanh(((action-agent.actor.action_shift)/agent.actor.action_scale).clamp(-.999999,.999999)))
        save(out/'fixture.pt',agent,metadata,torch.Generator(device='cuda:0'))
        (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
        # Same real renderer and observer used by automatic exports, through done=True.
        import capture_archive_render
        old=sys.argv;sys.argv=['capture_archive_render.py','--archive-root',str(root),'--checkpoint',str(out/'fixture.pt'),
            '--mode','render','--output',str(out/'recording')]
        try:capture_archive_render.main()
        finally:sys.argv=old
        report=json.loads((out/'recording/report.json').read_text())
        with np.load(out/'recording/physics_trace.npz') as raw,np.load(out/'recording/ending.npz') as ending,np.load(out/'recording/audio_substeps.npz') as audio:
            policy_count=round(report['ending_start_policy_seconds']/.001)
            np.testing.assert_array_equal(raw['qpos'][policy_count-1],ending['qpos'][0])
            self.assertEqual(len(raw['qpos']),policy_count+len(ending['time'])-1)
            self.assertEqual(len(audio['key_qpos']),len(raw['qpos'])+round(report['preparation_prefix_seconds']/.001))
            self.assertEqual(report['video_frames'],round((report['preparation_prefix_seconds']+len(raw['qpos'])*.001)*report['fps'])+1+round(report['fps']))
        import export_score_annotations
        from types import SimpleNamespace
        export_score_annotations.recover(SimpleNamespace(archive_root=root,config=out/'config.json',recording=out/'recording',output=out/'data'))
        with np.load(out/'data/score_annotations.npz') as annotation:
            count=round(report['ending_seconds']*report['fps'])
            self.assertEqual(annotation['goal'][-count:].sum(),0)
            self.assertEqual(annotation['fingers'][-count:].sum(),0)
        import package_interactive_replay
        result=package_interactive_replay.package(out,out/'recording',dict(checkpoint_sha256=sha256(out/'fixture.pt'),archive_sha256=sha256(root.parent/'source.tar.gz'),videos=[dict(native_arms=False)]))
        (out/'validation.json').write_text(json.dumps(dict(package=result,report=report,scope='two-note untrained fixture; no learning'),indent=2)+'\n')

        # Fault injection at the real terminal integration preserves the WHOLE run.
        from unittest.mock import patch
        from tsensei_v3_2_pianist.mjlab_env import ending as ending_module
        from tsensei_v3_2_pianist.mjlab_env.first_note_preparation import PreparationFailure
        original_generate=ending_module.generate
        def fail_arrival(*args,**kwargs):
            contract,trace=original_generate(*args,**kwargs)
            contract['report']['valid']=False
            raise PreparationFailure('injected arrival failure',contract,trace)
        old=sys.argv;sys.argv=['capture_archive_render.py','--archive-root',str(root),'--checkpoint',str(out/'fixture.pt'),
            '--mode','render','--output',str(out/'failed_recording')]
        try:
            with patch.object(ending_module,'generate',side_effect=fail_arrival),self.assertRaisesRegex(PreparationFailure,'injected arrival'):
                capture_archive_render.main()
        finally:sys.argv=old
        failure=json.loads((out/'failed_recording/physics_trace.json').read_text())
        self.assertFalse(failure['complete'])
        with np.load(out/'failed_recording/physics_trace.npz') as raw,np.load(out/'failed_recording/audio_substeps.npz') as audio:
            self.assertEqual(len(audio['key_qpos']),len(raw['qpos'])+round(report['preparation_prefix_seconds']/.001))
        self.assertTrue((out/'failed_recording/rollout.npz').is_file())
