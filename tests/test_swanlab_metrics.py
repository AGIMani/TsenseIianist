"""Reporting preserves metric populations, weights, ordering and privacy."""
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "swanlab_metrics", Path(__file__).parents[1] / "scripts" / "swanlab_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def config():
    return {"num_envs": 512, "reward_config": {"terms": {
        "sustain_reward": {"enabled": True, "weight": 1},
        "self_collision_reward": {"enabled": True, "weight": .2},
        "energy_reward": {"enabled": True, "weight": 1},
        "finger_speed_reward": {"enabled": False, "weight": 0},
    }}}


def episode(tp, fp, fn):
    return dict(music_true_positive_count=tp, music_false_positive_count=fp,
                music_false_negative_count=fn)


class SwanLabMetricsTest(unittest.TestCase):
    def test_arm_reward_and_tracking_diagnostics_keep_units_counts_and_privacy(self):
        cfg = config()
        cfg['reward_config']['terms']['arm_tracking_error_penalty'] = dict(enabled=True, weight=1.,
            params=dict(error_scale_m=.001, private_path='/private/no-upload'))
        cfg['reward_config']['terms']['arm_motion_limit_penalty'] = dict(enabled=True, weight=1.,
            params=dict(soft_limit_ratio=.95, full_penalty_excess=.1, maximum_penalty_per_step=5.))
        diagnostics = dict(requested_peak_m=.003, ik_target_peak_m=.0005,
            over_1mm_substeps=3, sampled_substeps=10, within_1mm_substep_fraction=.7,
            over_1mm_entry_events=2, over_1mm_control_steps=1, sampled_control_steps=1,
            over_1mm_side_substeps=dict(right=2, left=2, private=99), private='no-upload')
        values = metrics.scalars_from_record(dict(arm_tracking=diagnostics,
            eval=dict(arm_tracking=diagnostics),
            reward_weighted_terms=dict(arm_tracking_error_penalty=-4., arm_motion_limit_penalty=-5.)), cfg)
        for group in ('arm_tracking', 'evaluation_arm_tracking'):
            self.assertEqual(values[group+'/requested_peak_m'], .003)
            self.assertEqual(values[group+'/within_1mm_substep_fraction'], .7)
            self.assertEqual(values[group+'/over_1mm_substeps'], 3)
            self.assertEqual(values[group+'/over_1mm_right_substeps'], 2)
        for term, expected in (('arm_tracking_error_penalty', -4.), ('arm_motion_limit_penalty', -5.)):
            name = next(key for key in metrics.metric_order(cfg) if key.endswith('_'+term))
            self.assertEqual(values[name], expected)
        public = metrics.public_config(cfg)
        self.assertEqual(public['reward_config']['terms']['arm_tracking_error_penalty']['params'],
            dict(error_scale_m=.001))
        self.assertNotIn('private', json.dumps(values))
        self.assertNotIn('private', json.dumps(public))

    def test_current_key_formula_and_exact_sampling_contract_are_published(self):
        root=Path(__file__).parents[1]
        reward=json.loads((root/'configs/rewards/default_20261006.json').read_text())
        source=dict(reward_config=reward,score_timing_contract=dict(
            reward_contract=reward['contract'],reward_target='original_midi_physical_substeps',
            reward_state='post_physics_substep_mean',key_correct_signal='continuous_gaussian_key_travel',
            key_wrong_signal='physical_key_activation_count',control_dt=.05,physics_dt=.005,
            key_activation_weight=.5, private_path='/private/do-not-upload',
            exact_events=dict(schema='exact_note_events_v1',interval='start_inclusive_end_exclusive',
                physical_sample_time='post_step',observation='next_on_off_seconds_v1',
                observation_size=176,absent_event=-1.,observation_horizon=.5,
                source_path='/private/do-not-upload')))
        source['reward_config']['performance'].update(target_key_activation_weight=.5,
            substep_rewards=True,target_key_substep_reward=True)
        public=metrics.public_config(source)
        self.assertEqual(public['reward_config']['contract'],'physical_key_travel_exact_v1')
        self.assertIs(public['reward_config']['performance']['exact_note_timing'],True)
        timing=public['score_timing_contract']
        self.assertEqual(timing['key_correct_signal'],'continuous_gaussian_key_travel')
        self.assertEqual(timing['key_wrong_signal'],'physical_key_activation_count')
        self.assertEqual(timing['physics_dt'],.005)
        self.assertEqual(timing['exact_events']['physical_sample_time'],'post_step')
        encoded=json.dumps(public)
        for removed in ('activation_weight','substep_rewards','target_key_substep_reward','/private/'):
            self.assertNotIn(removed,encoded)
        source['score_timing_contract']['key_correct_signal']='/private/do-not-upload'
        self.assertNotIn('key_correct_signal',metrics.public_config(source)['score_timing_contract'])

    def test_joint_motion_penalty_is_reported_without_reweighting(self):
        cfg = config()
        cfg['reward_config']['terms']['joint_motion_limit_penalty'] = {'enabled': True, 'weight': .5}
        values = metrics.scalars_from_record({
            'reward_weighted_terms': {'joint_motion_limit_penalty': -2.5},
            'reward_raw_terms': {'joint_motion_limit_penalty': -5.}}, cfg)
        main = [name for name in metrics.metric_order(cfg) if name.endswith('_joint_motion_limit_penalty')]
        self.assertEqual(len(main), 1)
        self.assertEqual(values[main[0]], -2.5)
        self.assertEqual(values['reward_raw/joint_motion_limit_penalty'], -5.)

    def test_release_diagnostics_stay_separate_from_primary_f1(self):
        row = {'eval': {'f1': .9, 'release_tail_frames': 8,
            'release_tail_wrong_key_frames_max': 2, 'release_tail_end_active_keys_max': 0,
            'release_tail_wrong_key_key_substeps': 13, 'release_tail_complete': True}}
        values = metrics.scalars_from_record(row, config())
        self.assertEqual(values['metrics/01_F1'], .9)
        self.assertEqual(values['evaluation/release_tail_frames'], 8.)
        self.assertEqual(values['evaluation/release_tail_wrong_key_frames_max'], 2.)
        self.assertEqual(values['evaluation/release_tail_end_active_keys_max'], 0.)
        self.assertEqual(values['evaluation/release_tail_wrong_key_key_substeps'], 13.)
        self.assertEqual(values['evaluation/release_tail_complete'], 1.)
        for absent in (None, 'true'):
            row['eval']['release_tail_complete'] = absent
            self.assertNotIn('evaluation/release_tail_complete', metrics.scalars_from_record(row, config()))

    def test_f1_repair_parameters_report_resolved_and_requested_start_separately(self):
        source = {'requested_learning_start_transition_1_based': 5000,
            'learning_start_transition_1_based': 95232,
            'replay_prefill': {'episodes': 1, 'episode_steps': 186, 'required_vector_steps': 186,
                'required_transitions': 95232, 'requested_learning_start_transition_1_based': 5000,
                'effective_learning_start_transition_1_based': 95232, 'resolved': True,
                'source_path': '/private/replay'},
            'reward_config': {'performance': {'anticipatory_current_priority': True}},
            'source_runtime': {'environment': {'release_tail_time': .4, 'score_validation': 'strict_v1'}}}
        public = metrics.public_config(source)
        self.assertEqual(public['requested_learning_start_transition_1_based'], 5000)
        self.assertEqual(public['learning_start_transition_1_based'], 95232)
        self.assertEqual(public['replay_prefill']['required_transitions'], 95232)
        self.assertIs(public['replay_prefill']['resolved'], True)
        self.assertNotIn('source_path', public['replay_prefill'])
        self.assertIs(public['reward_config']['performance']['anticipatory_current_priority'], True)
        self.assertEqual(public['environment']['release_tail_time'], .4)
        self.assertEqual(public['environment']['score_validation'], 'strict_v1')

    def test_model_profile_names_match_source_namespaces(self):
        # The task profile tsensei_v1 is not a valid drive profile.
        for profile in ('legacy_v1', 'shadow_v1'):
            published = metrics.public_config({'source_runtime': {'model': {
                'drive_profile': profile, 'collision_profile': 'shadow_primitive_v1'}}})
            self.assertEqual(published['model']['drive_profile'], profile)
        published = metrics.public_config({'source_runtime': {'model': {
            'drive_profile': 'tsensei_v1'}}})
        self.assertNotIn('drive_profile', published['model'])
        for profile in ('legacy_mesh_v1', 'shadow_primitive_v1'):
            published = metrics.public_config({'source_runtime': {'model': {
                'collision_profile': profile}}})
            self.assertEqual(published['model']['collision_profile'], profile)
        published = metrics.public_config({'source_runtime': {'model': {
            'collision_profile': 'tsensei_v1'}}})
        self.assertNotIn('collision_profile', published['model'])

    def test_main_order_survives_alphabetic_chart_sort(self):
        order = metrics.metric_order(config())
        self.assertEqual(order, ["metrics/01_F1", "metrics/02_recall", "metrics/03_precision",
            "metrics/04_energy_reward", "metrics/05_self_collision_reward", "metrics/06_sustain_reward"])
        self.assertEqual(order, sorted(order))

    def test_main_metrics_all_use_existing_full_episode_frame_averages(self):
        row = {"eval": {"f1": .8, "recall": .81, "precision": .82,
            "music_only_f1": .7, "music_only_f1_min": .69,
            "music_only_micro_f1": .9, "episode_statistics": [episode(10, 2, 8), episode(30, 6, 12)]},
            "episodes": {"mean_f1": .3}}
        values = metrics.scalars_from_record(row, config())
        self.assertEqual(list(values)[:3], list(metrics.PRIMARY))
        self.assertEqual([values[key] for key in metrics.PRIMARY], [.8, .81, .82])
        self.assertNotIn("evaluation/music_only_f1_min", values)
        self.assertFalse(any("music_only_micro" in key for key in values))
        self.assertEqual(values["train_episodes/mean_f1"], .3)

    def test_weighted_reward_is_never_multiplied_again_or_replaced_with_raw(self):
        row = {"reward_weighted_terms": {"self_collision_reward": -.6, "energy_reward": -.4},
               "reward_terms": {"self_collision_reward": -99},
               "reward_raw_terms": {"self_collision_reward": -3}}
        values = metrics.scalars_from_record(row, config())
        self.assertEqual(values["metrics/05_self_collision_reward"], -.6)
        self.assertEqual(values["reward_raw/self_collision_reward"], -3)
        self.assertNotIn("metrics/06_sustain_reward", values)
        # Historical weighted terms use the legacy field, not config weights.
        values = metrics.scalars_from_record({"reward_terms": {"self_collision_reward": -.6}}, config())
        self.assertEqual(values["metrics/05_self_collision_reward"], -.6)

    def test_missing_or_invalid_eval_never_fills_main_from_training_or_music(self):
        rows = [{}, {"eval": {"music_only_f1": .9}, "episodes": {"mean_f1": .8}},
                {"eval": {"f1": None, "precision": "0.9", "recall": True}},
                {"eval": {"f1": float("nan"), "precision": float("inf"), "recall": -float("inf")}}]
        for row in rows:
            with self.subTest(row=row):
                values = metrics.scalars_from_record(row, config())
                self.assertFalse(set(values).intersection(metrics.PRIMARY))
                json.dumps(values, allow_nan=False)

    def test_nonfinite_training_and_reward_values_are_omitted(self):
        values = metrics.scalars_from_record({
            "train": {"actor_loss": float("inf"), "critic_loss": float("nan"), "entropy": 2},
            "reward_weighted_terms": {"energy_reward": float("nan")},
            "reward_terms": {"energy_reward": -.3}}, config())
        self.assertEqual(values, {"train/entropy": 2.})

    def test_missing_counts_or_empty_predictions_do_not_create_fake_micro_scores(self):
        for episodes in ([], [episode(2, 0, 1), {}], [episode(1, -1, 0)], [episode(1.5, 0, 1)]):
            values = metrics.scalars_from_record({"eval": {"episode_statistics": episodes}}, config())
            self.assertFalse(any("micro_" in key for key in values))
        values = metrics.scalars_from_record({"eval": {"episode_statistics": [episode(0, 0, 3)]}}, config())
        self.assertFalse(any("micro_" in key for key in values))
        values = metrics.scalars_from_record({"eval": {"episode_statistics": [episode(0, 0, 0)]}}, config())
        self.assertFalse(any("micro_" in key for key in values))

    def test_substep_micro_scores_have_separate_population_names(self):
        row = episode(4, 1, 1)
        row.update(music_only_substep_true_positive_count=20,
                   music_only_substep_false_positive_count=5,
                   music_only_substep_false_negative_count=15)
        values = metrics.scalars_from_record({"eval": {"episode_statistics": [row]}}, config())
        self.assertFalse(any("micro_" in key for key in values))

    def test_merge_same_step_keeps_train_episode_eval_and_latest_present_value(self):
        rows = [
            {"vector_step": 100, "reward_weighted_terms": {"energy_reward": -.5},
             "train": {"actor_loss": 2}},
            {"vector_step": 100, "eval": {"f1": .8, "recall": .9, "precision": .7}},
            {"vector_step": 100, "episodes": {"mean_f1": .4}},
            {"vector_step": 200, "train": {"actor_loss": 1}},
            {"vector_step": 100, "train": {"actor_loss": 3}, "eval": {"f1": None}},
        ]
        merged = metrics.merge_same_step_records(rows, config())
        self.assertEqual([step for step, _ in merged], [100, 200])
        values = merged[0][1]
        self.assertEqual(list(values)[:4], [*metrics.PRIMARY, "metrics/04_energy_reward"])
        self.assertEqual(values["metrics/01_F1"], .8)
        self.assertEqual(values["train/actor_loss"], 3)
        self.assertEqual(values["train_episodes/mean_f1"], .4)

    def test_invalid_steps_are_not_invented(self):
        for step in (None, True, 1.5, -1, "1"):
            with self.subTest(step=step), self.assertRaises(ValueError):
                metrics.merge_same_step_records([{"vector_step": step}], config())

    def test_public_config_allowlist_excludes_secrets_source_paths_and_scores(self):
        private = "fixture-secret-do-not-upload"
        source = config()
        source.update(api_key=private, credentials={"token": private},
            cli={"seed": 42, "api_key": private, "midi": "/private/score.mid"},
            source_sha256={"/private/module.py": private},
            execution_environment={"hostname": private}, device="/private/device",
            sac={"actor_lr": .0003, "actor_hidden": [256, 256, 256], "api_key": private,
                 "gamma": .8, "critic_lr": float("nan")},
            preparation={"mode": "policy_v1", "duration": 1., "path": private},
            source_runtime={"external_score": {"notes": [private]},
                "environment": {"midi": "/private/score.mid", "control_hz": 20,
                                "physics_hz": 200, "task_profile": "shadow_v1", "api_key": private},
                "pose": {"height": .1}, "model": {"drive_profile": "shadow_v1", "source": private}})
        source["reward_config"].update(token=private, performance={
            "anticipatory_fingering_time": .15, "substep_rewards": True, "api_key": private})
        source["reward_config"]["terms"]["energy_reward"].update(api_key=private,
            params={"coefficient": -.01, "reference": private, "mode": private})
        source["reward_config"]["terms"][private] = {"enabled": True, "weight": 99}
        public = metrics.public_config(source)
        payload = json.dumps(public, allow_nan=False)
        for forbidden in (private, "/private/", "source_sha256", "external_score", "api_key", "hostname"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(public["sac"]["actor_hidden"], [256, 256, 256])
        self.assertEqual(public["sac"]["actor_lr"], .0003)
        self.assertEqual(public["environment"]["control_hz"], 20)
        self.assertEqual(public["seed"], 42)
        self.assertEqual(public["reward_config"]["terms"]["energy_reward"]["weight"], 1)
        self.assertEqual(public["reward_config"]["performance"]["anticipatory_fingering_time"], .15)
        # The input, including private data, remains unchanged on disk/in memory.
        self.assertEqual(source["sac"]["api_key"], private)


if __name__ == "__main__":
    unittest.main()
