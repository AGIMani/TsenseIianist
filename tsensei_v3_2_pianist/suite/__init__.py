"""Standalone training environment factory."""
from dm_control import composer
from tsensei_v3_2_pianist import music
from tsensei_v3_2_pianist.music import library
from tsensei_v3_2_pianist.suite.tasks.piano_with_tsensei_hands import PianoWithTsenseiHands


def load(midi="Toy", seed=0, task_kwargs=None):
    """Return a dm_env environment with joint-position targets and sustain action."""
    kwargs = dict(task_kwargs or {})
    score = library.toy() if midi == "Toy" else music.load(midi)
    return composer.Environment(
        task=PianoWithTsenseiHands(midi=score, **kwargs),
        random_state=seed, strip_singleton_obs_buffer_dim=True,
    )
