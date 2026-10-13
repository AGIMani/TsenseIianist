"""CUDA execution layout for the existing Torch fused Adam arithmetic.

The installed PyTorch 2.10 header supplies adam_math verbatim (BSD-3-Clause,
PyTorch contributors); no optimizer formula is reimplemented here. This adapter
uses smaller chunks and shares the original scalar bias corrections per block.
It retains float64 intermediate arithmetic in Torch's fused Adam, FP32 state,
the original step tensors and checkpoint layout. No TF32/AMP is introduced.
"""
import os
from pathlib import Path
import sys

import torch


_EXTENSION = None
_DECLARATION = '''
void adam_step(std::vector<torch::Tensor> p, std::vector<torch::Tensor> g,
               std::vector<torch::Tensor> m, std::vector<torch::Tensor> v,
               std::vector<torch::Tensor> steps, double lr, double beta1,
               double beta2, double eps);
'''
_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/native/cuda/fused_adam_utils.cuh>

namespace {
constexpr int max_tensors = 64;
constexpr int threads = 256;
constexpr int chunk = threads * at::native::kILP;
struct Metadata {
  float* p[max_tensors]; float* g[max_tensors]; float* m[max_tensors];
  float* v[max_tensors]; float* steps[max_tensors];
  int64_t sizes[max_tensors]; int ends[max_tensors];
};

__global__ void adam_layout(Metadata data, double lr, double beta1,
                            double beta2, double eps) {
  int tensor = 0;
  while (blockIdx.x >= data.ends[tensor]) ++tensor;
  const int previous_end = tensor ? data.ends[tensor-1] : 0;
  const int64_t offset = int64_t(blockIdx.x - previous_end) * chunk;
  __shared__ float bias1;
  __shared__ float bias2_sqrt;
  if (threadIdx.x == 0) {
    // Identical double operations and final opmath_t cast as Torch's functor.
    const double correction1 = 1 - at::native::pow_(beta1, *data.steps[tensor]);
    const double correction2 = 1 - at::native::pow_(beta2, *data.steps[tensor]);
    bias1 = correction1;
    bias2_sqrt = std::sqrt(correction2);
  }
  __syncthreads();
  float args[4][at::native::kILP];
  #pragma unroll
  for (int item = 0; item < at::native::kILP; ++item) {
    const int64_t index = offset + threadIdx.x * at::native::kILP + item;
    const bool valid = index < data.sizes[tensor];
    args[0][item] = valid ? data.p[tensor][index] : 0;
    args[1][item] = valid ? data.g[tensor][index] : 0;
    args[2][item] = valid ? data.m[tensor][index] : 0;
    args[3][item] = valid ? data.v[tensor][index] : 0;
  }
  at::native::adam_math<float, float, 4, at::native::ADAM_MODE::ORIGINAL, false>(
      args, lr, beta1, beta2, 0., eps, false, nullptr, nullptr, bias1, bias2_sqrt);
  #pragma unroll
  for (int item = 0; item < at::native::kILP; ++item) {
    const int64_t index = offset + threadIdx.x * at::native::kILP + item;
    if (index < data.sizes[tensor]) {
      data.p[tensor][index] = args[0][item];
      data.m[tensor][index] = args[2][item];
      data.v[tensor][index] = args[3][item];
    }
  }
}
}

void adam_step(std::vector<torch::Tensor> p, std::vector<torch::Tensor> g,
               std::vector<torch::Tensor> m, std::vector<torch::Tensor> v,
               std::vector<torch::Tensor> steps, double lr, double beta1,
               double beta2, double eps) {
  TORCH_CHECK(!p.empty() && p.size() <= max_tensors, "Unsupported Adam group size");
  TORCH_CHECK(p.size()==g.size() && p.size()==m.size() && p.size()==v.size()
              && p.size()==steps.size(), "Inconsistent Adam tensor lists");
  c10::cuda::CUDAGuard guard(p[0].device());
  Metadata data;
  int blocks = 0;
  for (size_t i=0; i<p.size(); ++i) {
    for (const auto& value : {p[i], g[i], m[i], v[i], steps[i]}) {
      TORCH_CHECK(value.is_cuda() && value.scalar_type()==torch::kFloat32
                    && value.is_contiguous() && value.device()==p[0].device(),
                  "Adam execution requires contiguous FP32 CUDA tensors");
    }
    TORCH_CHECK(p[i].numel()==g[i].numel() && p[i].numel()==m[i].numel()
                  && p[i].numel()==v[i].numel() && steps[i].numel()==1,
                "Inconsistent Adam tensor sizes");
    data.p[i]=p[i].data_ptr<float>(); data.g[i]=g[i].data_ptr<float>();
    data.m[i]=m[i].data_ptr<float>(); data.v[i]=v[i].data_ptr<float>();
    data.steps[i]=steps[i].data_ptr<float>(); data.sizes[i]=p[i].numel();
    blocks += (p[i].numel()+chunk-1)/chunk;
    data.ends[i]=blocks;
  }
  adam_layout<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      data, lr, beta1, beta2, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault('MAX_JOBS', '2')
        previous_path = os.environ.get('PATH', '')
        os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + previous_path
        try:
            _EXTENSION = load_inline(
                name='tsensei_adam_execution_v1', cpp_sources=_DECLARATION,
                cuda_sources=_CUDA, functions=['adam_step'],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--expt-extended-lambda'], verbose=False)
        finally:
            os.environ['PATH'] = previous_path
    return _EXTENSION


class AdamExecution:
    """Use already initialized standard Adam state without replacing it."""
    def __init__(self):
        self.extension = extension()

    @torch.no_grad()
    def __call__(self, optimizer):
        for group in optimizer.param_groups:
            if (group.get('amsgrad') or group.get('maximize') or group.get('differentiable')
                    or group.get('weight_decay', 0) or group.get('decoupled_weight_decay', False)
                    or isinstance(group['lr'], torch.Tensor)):
                raise ValueError('Unsupported optimizer option for unchanged Adam execution')
            params = [parameter for parameter in group['params'] if parameter.grad is not None]
            if not params:
                continue
            grads = [parameter.grad for parameter in params]
            values = [optimizer.state[parameter] for parameter in params]
            steps = [value['step'] for value in values]
            torch._foreach_add_(steps, 1)
            # Separate launch groups do not affect independent parameter updates.
            for start in range(0, len(params), 64):
                stop = start+64
                self.extension.adam_step(
                    params[start:stop], grads[start:stop],
                    [value['exp_avg'] for value in values[start:stop]],
                    [value['exp_avg_sq'] for value in values[start:stop]], steps[start:stop],
                    group['lr'], *group['betas'], group['eps'])
