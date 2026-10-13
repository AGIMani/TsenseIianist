"""Dispatch only independent pending work and retain the running experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/hpc'))
from parallel_twinkle_ablation import free_allocations, dispatchable


def test_adopt_c_start_d_then_e_on_first_finished_pair():
    stages=[dict(label='C',status='running',allocation=1364),dict(label='D',status='pending'),dict(label='E',status='pending')]
    choices=list(zip(dispatchable(stages),free_allocations(stages,[1364,1457])))
    assert [(s['label'],a) for s,a in choices]==[('D',1457)]
    stages[1].update(status='running',allocation=1457)
    assert free_allocations(stages,[1364,1457])==[]
    for first,pair in [(0,1364),(1,1457)]:
        stages[first]['status']='completed'
        assert [(s['label'],a) for s,a in zip(dispatchable(stages),free_allocations(stages,[1364,1457]))]==[('E',pair)]
        stages[first]['status']='running'


def test_failed_group_prevents_launch_but_leaves_other_pair_busy():
    stages=[dict(label='C',status='failed',allocation=1364),dict(label='D',status='running',allocation=1457),dict(label='E',status='pending')]
    assert dispatchable(stages)==[]
    assert free_allocations(stages,[1364,1457])==[1364]
