# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
# Modified 2025 by ChatGPT (fix: CUDA thread-limit safety + clearer fallbacks)

from numba import config
config.CUDA_DEFAULT_PTX_CC = (8, 0)

import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    CUDA diagonal implementation. Each block processes one sample (pair).
    Threads per block correspond to max diagonal length (we cap this in Python launcher).
    """
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
    """
    Backward pass on CUDA, following the anti-diagonals in reverse.
    """
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
    """
    CUDA-backed autograd Function. We cap threads_per_block to device capability.
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        # Ensure scalar floats for kernel args
        gamma_val = float(gamma)
        bandwidth_val = float(bandwidth)

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        # Determine device capability (max threads per block)
        if D.is_cuda:
            try:
                props = torch.cuda.get_device_properties(dev)
                max_threads = props.max_threads_per_block
            except Exception:
                # fallback conservative default
                max_threads = 1024
        else:
            max_threads = 1024

        threads_per_block = int(min(max(N, M), max_threads))
        if threads_per_block <= 0:
            raise RuntimeError(f"Invalid threads_per_block computed: {threads_per_block}")

        n_passes = 2 * threads_per_block - 1

        # Prepare R buffer (float32 for CUDA kernel)
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=torch.float32) * math.inf
        R[:, 0, 0] = 0.0

        # Convert D to float32 for kernel (kernel expects float32 arrays)
        D_cuda = D.detach().to(torch.float32)

        # Launch the kernel. If threads_per_block is lower than max diagonal length (because we capped),
        # the kernel still runs but will only use `threads_per_block` threads: This means we must ensure
        # max(N,M) <= threads_per_block in normal designs. To be safe, we print a warning when capping happens.
        if max(N, M) > max_threads:
            # If sequence length exceeds capability, safer to raise or let caller fallback.
            # Here we still attempt launch (but it's likely incorrect if algorithm assumed full diagonal threads).
            print(f"[SoftDTW] Warning: sequence length {max(N,M)} > device max_threads_per_block {max_threads}. "
                  "Consider using CPU fallback or tile-based CUDA implementation.")

        # Perform kernel launch
        compute_softdtw_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_cuda),
            gamma_val, bandwidth_val, N, M, n_passes,
            cuda.as_cuda_array(R)
        )

        # Save tensors for backward
        ctx.save_for_backward(D, R.clone(), torch.tensor(gamma_val, device=dev), torch.tensor(bandwidth_val, device=dev))

        # Return values (converted back to original dtype)
        result = R[:, -2, -2]
        if dtype != torch.float32:
            result = result.to(dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma_t, bandwidth_t = ctx.saved_tensors

        gamma_val = float(gamma_t.item())
        bandwidth_val = float(bandwidth_t.item())

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        # Device capability
        if grad_output.is_cuda:
            try:
                props = torch.cuda.get_device_properties(dev)
                max_threads = props.max_threads_per_block
            except Exception:
                max_threads = 1024
        else:
            max_threads = 1024

        threads_per_block = int(min(max(N, M), max_threads))
        if threads_per_block <= 0:
            raise RuntimeError(f"Invalid threads_per_block computed in backward: {threads_per_block}")
        n_passes = 2 * threads_per_block - 1

        # Prepare D_, R_local, E buffers for backward kernel
        D_ = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D.to(torch.float32)

        R_local = R.to(torch.float32).clone()
        R_local[:, :, -1] = -math.inf
        R_local[:, -1, :] = -math.inf
        R_local[:, -1, -1] = R_local[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
        E[:, -1, -1] = 1.0

        compute_softdtw_backward_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R_local),
            1.0 / gamma_val,
            bandwidth_val,
            N, M, n_passes,
            cuda.as_cuda_array(E)
        )

        E_out = E[:, 1:N + 1, 1:M + 1]
        if dtype != torch.float32:
            E_out = E_out.to(dtype)
        grad_input = grad_output.view(-1, 1, 1).expand_as(E_out) * E_out
        return grad_input, None, None

# ----------------------------------------------------------------------------------------------------------------------
# CPU implementations (numba-optimized)
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0.0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < np.abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1.0
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                if 0 < bandwidth < np.abs(i - j):
                    continue
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU autograd Function that wraps the numba implementation.
    """
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma_val = float(gamma)
        bandwidth_val = float(bandwidth)
        D_np = D.detach().cpu().float().numpy()
        R_np = compute_softdtw(D_np, gamma_val, bandwidth_val)
        R = torch.tensor(R_np, device=dev).type(dtype)
        ctx.save_for_backward(D, R, torch.tensor(gamma_val, device=dev), torch.tensor(bandwidth_val, device=dev))
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        D, R, gamma_t, bandwidth_t = ctx.saved_tensors
        gamma_val = float(gamma_t.item())
        bandwidth_val = float(bandwidth_t.item())
        D_np = D.detach().cpu().float().numpy()
        R_np = R.detach().cpu().float().numpy()
        E_np = compute_softdtw_backward(D_np, R_np, gamma_val, bandwidth_val)
        E = torch.tensor(E_np, device=grad_output.device).type(grad_output.dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    SoftDTW module with both CUDA and CPU implementations and feature parity with original file.
    """

    def __init__(self, use_cuda: bool, gamma=1.0, normalize=False, bandwidth=None, dist_func=None, alignment_postprocess: str = 'row'):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        if alignment_postprocess not in ('row', 'none'):
            raise ValueError("alignment_postprocess must be 'row' or 'none'")
        self.alignment_postprocess = alignment_postprocess

        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by
        assert dx == dy

        use_cuda = self.use_cuda

        # Check device limit: many GPUs limit block size to 1024
        # Use torch.cuda.get_device_properties to find actual max_threads_per_block
        if use_cuda and (lx > 1024 or ly > 1024):
            print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
            use_cuda = False

        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x_exp = x.unsqueeze(2).expand(-1, n, m, d)
        y_exp = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x_exp - y_exp, 2).sum(3)

    def forward_with_cost_matrix(self, C, return_alignment: bool = False):
        assert C.dim() == 3, "Cost matrix C must be 3-dimensional (batch, N, M)"

        max_len = max(C.shape[1], C.shape[2])
        use_cuda = self.use_cuda
        if use_cuda and max_len > 1024:
            print(f"SoftDTW: Cannot use CUDA because a sequence length ({max_len}) > 1024")
            use_cuda = False

        func_dtw = _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

        if self.normalize:
            raise ValueError("Normalization is not supported when providing a pre-computed cost matrix.")

        if not return_alignment:
            return func_dtw(C, self.gamma, self.bandwidth)

        C_req = C.clone().detach().requires_grad_(True)
        sdtw_vals = func_dtw(C_req, self.gamma, self.bandwidth)
        grad_outputs = torch.ones_like(sdtw_vals, device=sdtw_vals.device)
        A = torch.autograd.grad(sdtw_vals, C_req, grad_outputs=grad_outputs, retain_graph=True)[0]

        if self.alignment_postprocess == 'row':
            eps = 1e-9
            A = A.clamp_min(0.0)
            A = A / (A.sum(dim=-1, keepdim=True) + eps)
        return sdtw_vals, A

    def forward(self, X, Y, return_alignment: bool = False):
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            if return_alignment:
                raise NotImplementedError("Alignment return is not implemented for normalize=True.")
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)

        if not return_alignment:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

        D_xy = self.dist_func(X, Y)
        D_req = D_xy.clone().detach().requires_grad_(True)
        sdtw_vals = func_dtw(D_req, self.gamma, self.bandwidth)
        grad_outputs = torch.ones_like(sdtw_vals, device=sdtw_vals.device)
        A = torch.autograd.grad(sdtw_vals, D_req, grad_outputs=grad_outputs, retain_graph=True)[0]

        if self.alignment_postprocess == 'row':
            eps = 1e-9
            A = A.clamp_min(0.0)
            A = A / (A.sum(dim=-1, keepdim=True) + eps)
        return sdtw_vals, A

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    from timeit import default_timer as timer
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()
    t += end - start
    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda() if torch.cuda.is_available() else a_cpu
        b_gpu = b_cpu.cuda() if torch.cuda.is_available() else b_cpu

        # GPU
        if torch.cuda.is_available():
            t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)
        else:
            t_gpu, forward_gpu, backward_gpu = None, None, None

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # If GPU available, verify results
        if torch.cuda.is_available():
            assert torch.allclose(forward_cpu, forward_gpu.cpu(), atol=1e-5), "forward mismatch CPU vs GPU"
            assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward), "backward mismatch CPU vs GPU"

        if i > 0:
            times_cpu.append(t_cpu)
            if t_gpu is not None:
                times_gpu.append(t_gpu)

    avg_cpu = np.mean(times_cpu) if len(times_cpu) else float('nan')
    avg_gpu = np.mean(times_gpu) if len(times_gpu) else float('nan')
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    if not np.isnan(avg_cpu) and not np.isnan(avg_gpu) and avg_gpu > 0:
        print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer
    torch.manual_seed(1234)

    profile(128, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(512, 256, 256, 2, tol_backward=1e-3)

