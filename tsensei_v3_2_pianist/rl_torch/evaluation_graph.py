"""Capture read-only evaluation statistics; never capture policy or physics steps."""
from collections import defaultdict

import torch


class EvaluationMetricGraph:
    """Replay the identical tensor operations against explicitly copied inputs.

    ``state`` lists every mutable accumulator, including prior motor targets.
    Warmup and capture restore them before processing the first real sample.
    No random calls, policy decisions, reward calculation or resets live here.
    """

    def __init__(self, callback, info, state, *, keys):
        self.keys = set(keys)
        self.inputs = {k: v.detach().clone() for k, v in info.items() if k in self.keys}
        if not self.inputs or any(not isinstance(v, torch.Tensor) or not v.is_cuda
                                  for v in self.inputs.values()):
            raise ValueError('Evaluation metric graph requires CUDA tensor inputs')
        self.device = next(iter(self.inputs.values())).device
        if any(v.device != self.device for v in (*self.inputs.values(), *state)):
            raise ValueError('Evaluation metrics and state must share one CUDA device')
        self.signature = {k: (v.shape, v.dtype, v.device) for k, v in self.inputs.items()}
        self.graph = torch.cuda.CUDAGraph()
        saved = [v.detach().clone() for v in state]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    callback(self.inputs)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph, stream=stream):
                callback(self.inputs)
        finally:
            torch.cuda.synchronize(self.device)
            for value, original in zip(state, saved):
                value.copy_(original)

    def sample(self, info):
        selected = {k: v for k, v in info.items() if k in self.keys}
        if ({k: (v.shape, v.dtype, v.device) for k, v in selected.items()} != self.signature):
            raise ValueError('Evaluation diagnostic fields changed during graph replay')
        groups = defaultdict(lambda: ([], []))
        for name, value in selected.items():
            dest, source = groups[value.dtype]
            dest.append(self.inputs[name]); source.append(value)
        for dest, source in groups.values():
            torch._foreach_copy_(dest, source)
        self.graph.replay()
