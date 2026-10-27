import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('gpu_status', Path(__file__).parents[1]/'scripts/hpc/gpu_status.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class InventoryTest(unittest.TestCase):
    def test_allocated_idle_utilization_is_still_owned(self):
        nodes='NodeName=n State=MIXED Gres=gpu:h100:4(S:0) AllocTRES=cpu=8,gres/gpu=2'
        jobs='JobId=42 JobName=train UserId=alice(1) JobState=RUNNING\n Nodes=n CPU_IDs=0-7 GRES=gpu:h100:2(IDX:0,2)'
        n,j=m.parse(nodes,jobs)
        self.assertEqual(set(n[0]['owners']),{0,2})
        self.assertEqual(n[0]['owners'][0][0]['user'],'alice')
        self.assertTrue(n[0]['mapping_complete'])
    def test_hidden_jobs_and_drain_not_free(self):
        n,_=m.parse('NodeName=n State=IDLE+DRAIN Gres=gpu:a100:8(S:0-1) AllocTRES=gres/gpu=2','')
        self.assertFalse(n[0]['mapping_complete'])
        self.assertFalse(n[0]['schedulable_state'])
    def test_array_job_identity(self):
        n,j=m.parse('NodeName=n State=MIXED Gres=gpu:a100:2 AllocTRES=gres/gpu=1',
                    'JobId=7632 ArrayJobId=7632 ArrayTaskId=615 JobName=a UserId=bob(2) JobState=RUNNING\n Nodes=n GRES=gpu:a100:1(IDX:1)')
        self.assertEqual(n[0]['owners'][1][0]['job_id'],'7632_615')
    def test_ranges(self):
        self.assertEqual(m.indices('0-2,5'),[0,1,2,5])
