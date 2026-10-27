"""CPU-only checks for explicitly authorized process-job lifecycle reporting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('process_backend',Path(__file__).resolve().parents[1]/'scripts/hpc/process_backend.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ProcessBackendTest(unittest.TestCase):
    def test_authorization_is_host_project_and_gpu_scoped(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);cfg=dict(authorized=True,hostname='node4-test',projects=[str(p)],gpu=0)
            (p/'process_backend.json').write_text(json.dumps(cfg))
            with patch.object(m.socket,'gethostname',return_value='node4-test'),patch.dict(m.os.environ,{},clear=True):
                self.assertEqual(m.authorize(p),cfg)
                cfg['projects']=[];(p/'process_backend.json').write_text(json.dumps(cfg))
                with self.assertRaises(RuntimeError):m.authorize(p)
    def test_dead_worker_cannot_remain_running(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.json';m.save(p,dict(state='RUNNING',pid=123456789))
            with patch.object(m.os,'kill',side_effect=ProcessLookupError):
                self.assertEqual(m.status(p)['state'],'FAILED')
    def test_terminal_status_is_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.json';m.save(p,dict(state='COMPLETED',pid=123456789,exit_code='0:0'))
            self.assertEqual(m.status(p)['exit_code'],'0:0')
if __name__=='__main__':unittest.main()
