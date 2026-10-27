import pytest

from scripts.compare_fullscore_analyses import compare


def source(f1,tp,fp,fn,same):
    return dict(entire_score_recorded=True,original_midi_sha256='midi',model_sha256='model',archive_sha256='archive',
        physics_hz=1000,control_hz=20,recorded_end_s=1.,original_score_end_s=.999,
        physical_trace_sha256=str(f1),score_sha256='score'+str(f1),global_metrics=dict(
        samples=1001,micro_f1=f1,tp=tp,fp=fp,fn=fn,same_hand_contact_samples=same,cross_hand_contact_samples=0))


def test_comparison_retains_regression_and_confounded_controller_label():
    left=source(.5,50,20,80,10);right=source(.4,40,20,100,5)
    bars=[dict(bar=1,start_s=0.,end_s=1.,**left['global_metrics'])]
    nextbars=[dict(bar=1,start_s=0.,end_s=1.,**right['global_metrics'])]
    result=compare(left,right,bars,nextbars)
    assert result['global_delta']['micro_f1']==pytest.approx(-.1)
    assert result['global_delta']['same_hand_contact_samples']==-5
    assert result['regressed_bar_count']==1
    assert result['biggest_improvements']==[]
    assert result['causal_attribution_supported'] is False


def test_different_midi_or_bar_partitions_refuse_comparison():
    left=source(.5,50,20,80,10);right=source(.4,40,20,100,5)
    bars=[dict(bar=1,start_s=0.,end_s=1.,**left['global_metrics'])]
    right['original_midi_sha256']='other'
    with pytest.raises(ValueError,match='original_midi_sha256'):compare(left,right,bars,bars)
    right['original_midi_sha256']='midi'
    with pytest.raises(ValueError,match='bar partition'):compare(left,right,bars,[{**bars[0],'end_s':.5}])
