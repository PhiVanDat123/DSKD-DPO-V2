# MIT License
#
# (C) 2020 Mehran Maghoumi
# Modified 2025 by ChatGPT for CUDA safety and stability

from numba import config
config.CUDA_DEFAULT_PTX_CC = (8, 0)

import numpy as np
import torch
import torch.cuda
from numba import jit, prange, cuda
from torch.autograd import Function
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma

    for p in range(n_passes):
        J = max(0, min(p - tid, max_j - 1))
        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))
        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):
            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = float(gamma)
        bandwidth = float(bandwidth)

        B, N, M = D.shape
        max_threads = torch.cuda.get_device_properties(dev).max_threads_per_block
        threads_per_block = int(min(max(N, M), max_threads))
        n_passes = 2 * threads_per_block - 1

        if max(N, M) > max_threads:
            print(f"[SoftDTW Warning] Sequence length ({max(N, M)}) > max threads per block ({max_threads}). "
                  f"Consider CPU fallback or chunking.")

        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=torch.float32) * math.inf
        R[:, 0, 0] = 0
        D_cuda = D.detach().to(torch.float32)

        compute_softdtw_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_cuda),
            gamma, bandwidth, N, M, n_passes,
            cuda.as_cuda_array(R)
        )

        ctx.save_for_backward(D, R.clone(), torch.tensor(gamma, device=dev), torch.tensor(bandwidth, device=dev))
        result = R[:, -2, -2].to(dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        D, R, gamma, bandwidth = ctx.saved_tensors
        gamma = float(gamma.item())
        bandwidth = float(bandwidth.item())
        dev = D.device
        dtype = D.dtype
        B, N, M = D.shape
        max_threads = torch.cuda.get_device_properties(dev).max_threads_per_block
        threads_per_block = int(min(max(N, M), max_threads))
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D.to(torch.float32)
        R_local = R.to(torch.float32).clone()
        R_local[:, :, -1] = -math.inf
        R_local[:, -1, :] = -math.inf
        R_local[:, -1, -1] = R_local[:, -2, -2]
        E = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
        E[:, -1, -1] = 1

        compute_softdtw_backward_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R_local),
            1.0 / gamma, bandwidth, N, M, n_passes,
            cuda.as_cuda_array(E)
        )

        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E.to(dtype), None, None

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B, N, M = D.shape
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B, N, M = D_.shape
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                if 0 < bandwidth < abs(i - j):
                    continue
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a, b, c = np.exp(a0), np.exp(b0), np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev, dtype = D.device, D.dtype
        D_ = D.detach().cpu().float().numpy()
        R = torch.tensor(compute_softdtw(D_, gamma, bandwidth)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, torch.tensor(gamma), torch.tensor(bandwidth))
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_, R_ = D.cpu().float().numpy(), R.cpu().float().numpy()
        E = torch.tensor(compute_softdtw_backward(D_, R_, gamma.item(), bandwidth.item())).to(D.device).type(D.dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    def __init__(self, use_cuda=True, gamma=1.0, normalize=False, bandwidth=None):
        super().__init__()
        self.use_cuda = use_cuda
        self.gamma = gamma
        self.normalize = normalize
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)

    def _get_func_dtw(self, x, y):
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by and dx == dy
        use_cuda = self.use_cuda
        if use_cuda and (lx > 1024 or ly > 1024):
            print(f"SoftDTW: sequence length > 1024 → fallback to CPU.")
            use_cuda = False
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        n, m, d = x.size(1), y.size(1), x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return ((x - y) ** 2).sum(3)

    def forward(self, X, Y):
        func_dtw = self._get_func_dtw(X, Y)
        D_xy = self._euclidean_dist_func(X, Y)
        return func_dtw(D_xy, self.gamma, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0)
    sdtw_cuda = SoftDTW(True, gamma=1.0)
    n_iters = 4
    print(f"Profiling batch={batch_size}, len=({seq_len_a},{seq_len_b}), dims={dims}")
    for i in range(n_iters):
        a = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b = torch.rand((batch_size, seq_len_b, dims))
        t_gpu, t_cpu = 0, 0
        if torch.cuda.is_available():
            ag, bg = a.cuda(), b.cuda()
            f_gpu = sdtw_cuda(ag, bg)
            torch.cuda.synchronize()
        f_cpu = sdtw(a, b)
        print(f"Iter {i+1}: CPU={f_cpu.mean().item():.4f}, GPU={f_gpu.mean().item() if torch.cuda.is_available() else 0:.4f}")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1234)
    profile(128, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(512, 256, 256, 2, tol_backward=1e-3)
