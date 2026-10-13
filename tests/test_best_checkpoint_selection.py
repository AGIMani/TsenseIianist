import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch

spec=importlib.util.spec_from_file_location('best_export',Path(__file__).resolve().parents[1]/'preview/scripts/export_best_mjlab.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BestCheckpointTest(unittest.TestCase):
    def exact_run(self, root):
        selection, best, metadata = self.modern_run(root)
        selection['best_metric'] = 'exact_original_music_micro_f1_min'
        value = best['evaluation']
        row = dict(completed=1, truncated=0, note_timing_available=True,
            note_timing_interval='start_inclusive_end_exclusive', note_timing_sample_time='post_step',
            note_timing_music_start_s=1., note_timing_music_end_s=8.,
            substep_key_diagnostics_coverage=1., substep_key_coverage=1.,
            substep_key_diagnostics_valid=True, exact_original_music_true_positive_count=990,
            exact_original_music_false_positive_count=0, exact_original_music_false_negative_count=10,
            exact_original_music_micro_f1=1980/1990)
        value.update(episode_statistics=[deepcopy(row) for _ in range(3)], note_timing_available=True,
            note_timing_episodes=3, substep_key_sample_hz=400., substep_key_diagnostics_coverage=1.,
            substep_key_coverage=1., substep_key_diagnostics_valid=True,
            exact_original_music_true_positive_count=2970, exact_original_music_false_positive_count=0,
            exact_original_music_false_negative_count=30, exact_original_music_micro_f1=1980/1990)
        best.update(metric=selection['best_metric'], score=1980/1990)
        (root/'config.json').write_text(json.dumps(dict(evaluation_selection=selection)))
        (root/'best.json').write_text(json.dumps(best))
        torch.save(dict(metadata=metadata), root/'best.pt')
        return best, metadata

    def modern_run(self, root, *, threshold=0):
        selection=dict(save_best=True,best_metric='music_only_f1_min',max_key_underside_substeps=threshold)
        evaluation=dict(music_only_f1_min=.95,f1=.94,return_=100,evaluation_episodes=3,
            completed_episodes=3,truncated_episodes=0,key_underside_substeps_max=0,
            key_contact_diagnostics_coverage=1.,key_contact_substep_coverage=1.,
            key_contact_diagnostics_valid=True,key_contact_physics_sample_hz=200.)
        best=dict(metric='music_only_f1_min',score=.95,vector_step=100,updates=256,
            evaluation=evaluation,best_eligible=True,best_ineligible_reasons=[],max_key_underside_substeps=threshold)
        metadata=dict(evaluation_selection=selection,vector_step=100,updates=256,
            last_evaluation=evaluation,last_evaluation_vector_step=100,
            evaluation_selection_state=dict(config=selection,best=best))
        (root/'config.json').write_text(json.dumps(dict(evaluation_selection=selection)))
        (root/'best.json').write_text(json.dumps(best))
        torch.save(dict(metadata=metadata),root/'best.pt')
        # Deliberately higher but unsafe legacy score must never win.
        (root/'metrics.jsonl').write_text(json.dumps(dict(vector_step=200,
            eval=dict(f1=.999,return_=200,key_underside_substeps_max=30)))+'\n')
        (root/'step-200.pt').touch()
        return selection,best,metadata

    def test_best_existing_and_completed_final(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            rows=[dict(vector_step=i,eval={'f1':f,'return':r}) for i,f,r in [(1000,.9,100),(2000,.8,200),(3000,.99,300),(3500,.95,150)]]
            (p/'metrics.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n{"vector_step":')
            (p/'step-1000.pt').touch();(p/'step-2000.pt').touch();(p/'final.pt').touch()
            self.assertEqual(Path(module.select_best(p)['checkpoint']).name,'step-1000.pt')
            (p/'summary.json').write_text(json.dumps({'status':'completed','vector_steps':3500}))
            self.assertEqual(Path(module.select_best(p)['checkpoint']).name,'final.pt')

    def test_no_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'metrics.jsonl').write_text('{"vector_step":10,"train":{}}\n')
            with self.assertRaises(ValueError):module.select_best(p)

    def test_authoritative_eligible_best_beats_higher_unsafe_logged_score(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);self.modern_run(p)
            result=module.select_best(p)
            self.assertEqual(Path(result['checkpoint']).name,'best.pt')
            self.assertEqual(result['evaluation']['vector_step'],100)
            self.assertIn('eligible minimum-episode',result['criterion'])
            self.assertIn('key_underside',result['criterion'])
            self.assertTrue(result['authoritative'])

    def test_new_run_without_best_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(module.time,'sleep'):
            p=Path(directory);self.modern_run(p);(p/'best.pt').unlink()
            with self.assertRaisesRegex(ValueError,'refusing legacy/unsafe fallback'):module.select_best(p)

    def test_inconsistent_metadata_or_json_never_falls_back(self):
        for field in ('vector_step','selection','eligible','coverage'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory,patch.object(module.time,'sleep'):
                p=Path(directory);selection,best,metadata=self.modern_run(p)
                if field=='vector_step':metadata['vector_step']=99
                elif field=='selection':metadata['evaluation_selection']=dict(selection,save_best=False)
                elif field=='eligible':best['best_eligible']=False
                else:best['evaluation']['key_contact_substep_coverage']=.5
                (p/'best.json').write_text(json.dumps(best));torch.save(dict(metadata=metadata),p/'best.pt')
                with self.assertRaisesRegex(ValueError,'refusing legacy/unsafe fallback'):module.select_best(p)

    def test_writer_lag_retries_then_uses_matching_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);_,best,_=self.modern_run(p)
            old=deepcopy(best);old['vector_step']=99
            (p/'best.json').write_text(json.dumps(old))
            def writer_finishes(_): (p/'best.json').write_text(json.dumps(best))
            with patch.object(module.time,'sleep',side_effect=writer_finishes) as wait:
                result=module.select_best(p)
            self.assertEqual(wait.call_count,1)
            self.assertEqual(result['best_record']['vector_step'],100)

    def test_previous_music_best_without_new_underside_gate_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);selection,best,metadata=self.modern_run(p,threshold=None)
            for key in ('best_eligible','best_ineligible_reasons','max_key_underside_substeps'):best.pop(key)
            selection.pop('max_key_underside_substeps')
            (p/'config.json').write_text(json.dumps(dict(evaluation_selection=selection)))
            (p/'best.json').write_text(json.dumps(best));torch.save(dict(metadata=metadata),p/'best.pt')
            self.assertEqual(Path(module.select_best(p)['checkpoint']).name,'best.pt')

    def test_disabled_authoritative_saving_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);selection,_,_=self.modern_run(p)
            selection['save_best']=False
            (p/'config.json').write_text(json.dumps(dict(evaluation_selection=selection)))
            with self.assertRaisesRegex(ValueError,'saving is disabled'):module.select_best(p)

    def test_exact_authoritative_best_revalidates_worst_episode_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            self.exact_run(p)
            result = module.select_best(p)
            self.assertIn('exact_original_music_micro_f1_min', result['criterion'])
            self.assertEqual(result['best_record']['score'], 1980/1990)
        for change in ('coverage', 'score', 'missing_rows'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory, patch.object(module.time, 'sleep'):
                p = Path(directory)
                best, metadata = self.exact_run(p)
                if change == 'coverage':
                    best['evaluation']['episode_statistics'][1]['substep_key_coverage'] = .5
                elif change == 'score':
                    best['score'] = .999
                else:
                    best['evaluation'].pop('episode_statistics')
                (p/'best.json').write_text(json.dumps(best))
                torch.save(dict(metadata=metadata), p/'best.pt')
                with self.assertRaisesRegex(ValueError, 'refusing legacy/unsafe fallback'):
                    module.select_best(p)


if __name__=='__main__':unittest.main()
