"""Validate pinned ending runtime against concurrently updated default Ty=.020m."""
import runpy
from pathlib import Path
from unittest.mock import patch

def test_current_default_origin_ty_002():
    from tsensei_v3_2_pianist.mjlab_env import source_runtime
    original=source_runtime.make_recipe
    def current_default(**kwargs):
        kwargs['ty']=.020
        return original(**kwargs)
    case=runpy.run_path(str(Path.cwd()/'tests/test_ending_gpu.py'))['EndingGPUChecks']()
    with patch.object(source_runtime,'make_recipe',side_effect=current_default):
        case.test_actual_terminal_reset_budget_and_first_control_step()
