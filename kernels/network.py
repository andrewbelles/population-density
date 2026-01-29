#!/usr/bin/env python3 
# 
# network.py  Andrew Belles  Jan 29th, 2026 
# 
# Triton Kernels specifically for SpatialBackbone and aggregation of pooled features from 
# backbone 
# 

import torch, triton 

import triton.language as tl 

# --------------------------------------------------------- 
# Masked Group Norm Kernels 
# --------------------------------------------------------- 

@triton.jit 
def _masked_group_norm_stats_kernel(
    x_ptr, m_ptr, mean_ptr, rstd_ptr, count_ptr, 
    N, C, H, W, G, 
    eps,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    n   = pid // G 
    Cg  = C // G 
    HW  = H * W 
    K   = Cg * HW 

    acc_sum   = tl.zeros((), dtype=tl.float32)
    acc_cnt   = tl.zeros((), dtype=tl.float32)
    acc_var   = tl.zeros((), dtype=tl.float32)

    group_offset = pid * K  
    mask_base    = n * HW  

    for k in range(0, K, BLOCK): 
        offset = k + tl.arange(0, BLOCK)
        mask_k  = offset < K 

        hw_idx = offset % HW 
        mask_ptr = m_ptr + mask_base + hw_idx 

        x = tl.load(x_ptr + group_offset + offset, mask=mask_k, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr, mask=mask_k, other=0.0)
        m = m > 0 
        x = tl.where(m, x, 0.0)

        acc_sum   += tl.sum(x, axis=0)
        acc_cnt   += tl.sum(m, axis=0)

    denom = tl.maximum(acc_cnt, 1.0)
    mean  = acc_sum / denom 

    tl.store(mean_ptr + pid, mean)
    tl.store(count_ptr + pid, acc_cnt)

    # second pass for variance 
    for k in range(0, K, BLOCK):
        offset = k + tl.arange(0, BLOCK)
        mask_k  = offset < K 

        hw_idx = offset % HW 
        mask_ptr = m_ptr + mask_base + hw_idx 

        x = tl.load(x_ptr + group_offset + offset, mask=mask_k, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr, mask=mask_k, other=0.0)
        m = m > 0 

        diff = tl.where(m, x - mean, 0.0)
        acc_var += tl.sum(diff * diff, axis=0)

    var  = acc_var / denom 
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit 
def _masked_group_norm_apply_kernel(
    x_ptr, m_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, y_ptr, 
    N, C, H, W, G, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    n   = pid // C 
    c   = pid % C 
    Cg  = C // G 
    g   = c // Cg 
    HW  = H * W 

    seek_stat = n * G + g
    mean = tl.load(mean_ptr + seek_stat)
    rstd = tl.load(rstd_ptr + seek_stat)

    w    = tl.load(w_ptr + c) if w_ptr is not None else 1.0 
    b    = tl.load(b_ptr + c) if b_ptr is not None else 0.0 

    plane_offset = pid * HW 
    mask_offset  = n * HW 

    for k in range(0, HW, BLOCK): 
        offset = k + tl.arange(0, BLOCK) 
        mask_k = offset < HW 

        x = tl.load(x_ptr + plane_offset + offset, mask=mask_k, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + mask_offset + offset, mask=mask_k, other=0.0)
        m = m > 0 

        y = (x - mean) * rstd 
        y = y * w + b 
        y = tl.where(m, y, 0.0)
        
        tl.store(y_ptr + plane_offset + offset, y, mask=mask_k)

@triton.jit 
def _masked_group_norm_backward_stats_kernel(
    x_ptr, dy_ptr, m_ptr, mean_ptr, rstd_ptr, w_ptr, 
    s1_ptr, s2_ptr,
    N, C, H, W, G, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    n   = pid // G 
    g   = pid % G 
    Cg  = C // G 
    HW  = H * W 
    K   = Cg * HW 

    group_offset = pid * K 
    mask_base    = n * HW 

    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)

    acc_s1  = tl.zeros((), dtype=tl.float32)
    acc_s2  = tl.zeros((), dtype=tl.float32)
    acc_cnt = tl.zeros((), dtype=tl.float32)

    for k in range(0, K, BLOCK): 
        offset = k + tl.arange(0, BLOCK)
        mask_k  = offset < K 

        c_rel    = offset // HW 
        global_c = g * Cg + c_rel 

        hw_idx   = offset % HW 
        mask_ptr = m_ptr + mask_base + hw_idx 

        x  = tl.load(x_ptr + group_offset + offset, mask=mask_k, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + group_offset + offset, mask=mask_k, other=0.0).to(tl.float32) 
        m  = tl.load(mask_ptr, mask=mask_k, other=0.0)
        m  = m > 0 

        x  = tl.where(m, x, 0.0)
        dy = tl.where(m, dy, 0.0)

        x_hat = (x - mean) * rstd 
        wgt   = tl.load(w_ptr + global_c, mask=mask_k, other=1.0) if w_ptr is not None else 1.0 

        dy_gamma = dy * wgt 

        acc_s1  += tl.sum(dy_gamma, axis=0)
        acc_s2  += tl.sum(dy_gamma * x_hat, axis=0)

    tl.store(s1_ptr + pid, acc_s1)
    tl.store(s2_ptr + pid, acc_s2)

@triton.jit 
def _masked_group_norm_backward_dx_kernel(
    x_ptr, dy_ptr, m_ptr, mean_ptr, rstd_ptr, w_ptr, count_ptr, 
    s1_ptr, s2_ptr, dx_ptr, dgamma_ptr, dbeta_ptr, 
    N, C, H, W, G, 
    BLOCK: tl.constexpr
):

    pid = tl.program_id(0)
    n   = pid // C 
    c   = pid % C 
    Cg  = C // G 
    g   = c // Cg 
    HW  = H * W 

    stat_idx = n * G + g 
    mean     = tl.load(mean_ptr + stat_idx)
    rstd     = tl.load(rstd_ptr + stat_idx)
    s1       = tl.load(s1_ptr + stat_idx)
    s2       = tl.load(s2_ptr + stat_idx)
    cnt      = tl.load(count_ptr + stat_idx)
    denom    = tl.maximum(cnt, 1.0)
    wgt      = tl.load(w_ptr + c).to(tl.float32) if w_ptr is not None else 1.0

    plane_offset = pid * HW 
    mask_offset  = n * HW 

    acc_dgamma = tl.zeros((), dtype=tl.float32)
    acc_dbeta  = tl.zeros((), dtype=tl.float32)

    for k in range(0, HW, BLOCK): 
        offset = k + tl.arange(0, BLOCK) 
        mask_k = offset < HW 

        x  = tl.load(x_ptr + plane_offset + offset, mask=mask_k, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + plane_offset + offset, mask=mask_k, other=0.0).to(tl.float32)
        m  = tl.load(m_ptr + mask_offset + offset, mask=mask_k, other=0.0)
        m  = m > 0 

        x  = tl.where(m, x, 0.0)
        dy = tl.where(m, dy, 0.0)

        x_hat = (x - mean) * rstd 
        dy_gamma = dy * wgt 

        dx = (dy_gamma - s1 / denom - x_hat * s2 / denom) * rstd 
        dx = tl.where(m, dx, 0.0)

        tl.store(dx_ptr + plane_offset + offset, dx, mask=mask_k)

        acc_dgamma += tl.sum(dy * x_hat, axis=0)
        acc_dbeta  += tl.sum(dy, axis=0)

    tl.atomic_add(dgamma_ptr + c, acc_dgamma)
    tl.atomic_add(dbeta_ptr + c, acc_dbeta)

# --------------------------------------------------------- 
# Masked Group Norm Triton Calls + Autograd Function 
# --------------------------------------------------------- 

def masked_group_norm_triton(x, mask, weight, bias, num_groups, eps=1e-4, return_stats=False): 
    assert x.is_cuda and mask.is_cuda 

    if not x.is_contiguous(): 
        x = x.contiguous() 

    if mask is not None and not mask.is_contiguous(): 
        mask = mask.contiguous() 

    if mask.ndim == 4: 
        mask = mask[:, 0]

    N, C, H, W = x.shape 
    G = num_groups
    assert C % G == 0 

    mean  = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    rstd  = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    count = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    grid  = (N * G,)

    _masked_group_norm_stats_kernel[grid](
        x, mask, mean, rstd, count, 
        N, C, H, W, G, 
        eps,
        BLOCK=1024
    )

    y = torch.empty_like(x)
    grid2 = (N * C,)
    _masked_group_norm_apply_kernel[grid2](
        x, mask, mean, rstd, weight, bias, y, 
        N, C, H, W, G, 
        BLOCK=256 
    )

    if return_stats: 
        return y, mean, rstd, count 
    return y 

def masked_group_norm_backward_triton(
    x, dy, mask, weight, mean, rstd, count, num_groups 
): 
    assert x.is_cuda and dy.is_cuda and mask.is_cuda 

    if not x.is_contiguous(): 
        x = x.contiguous() 

    if mask is not None and not mask.is_contiguous(): 
        mask = mask.contiguous() 

    if not dy.is_contiguous(): 
        dy = dy.contiguous() 

    if mask.ndim == 4: 
        mask = mask[:, 0]

    N, C, H, W = x.shape 
    G = num_groups

    x_fp32  = x.float() 
    dy_fp32 = dy.float() 
    w_fp32  = weight.float() if weight is not None else None 

    grid = (N * G,)
    s1   = torch.empty(grid, device=x.device, dtype=torch.float32)
    s2   = torch.empty(grid, device=x.device, dtype=torch.float32)

    _masked_group_norm_backward_stats_kernel[grid](
        x_fp32, dy_fp32, mask, mean, rstd, w_fp32, s1, s2,
        N, C, H, W, G, 
        BLOCK=1024
    )
    
    dx     = torch.empty_like(x_fp32)
    dgamma = torch.zeros((C,), device=x.device, dtype=torch.float32)
    dbeta  = torch.zeros((C,), device=x.device, dtype=torch.float32)
    grid2  = (N * C,)

    _masked_group_norm_backward_dx_kernel[grid2](
        x_fp32, dy_fp32, mask, mean, rstd, w_fp32, count, s1, s2, dx, dgamma, dbeta, 
        N, C, H, W, G,
        BLOCK=256
    )

    dx = dx.to(dtype=x.dtype)
    return dx, dgamma, dbeta 

class MaskedGroupNormTritonFn(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, x, mask, weight, bias, num_groups, eps): 
        y, mean, rstd, count = masked_group_norm_triton(
            x, mask, weight, bias, num_groups, eps, return_stats=True
        )
        ctx.save_for_backward(x, mask, weight, mean, rstd, count)
        ctx.num_groups = num_groups 
        return y 

    @staticmethod 
    def backward(ctx, dy): 
        x, mask, weight, mean, rstd, count = ctx.saved_tensors 
        dx, dgamma, dbeta = masked_group_norm_backward_triton(
            x, dy, mask, weight, mean, rstd, count, ctx.num_groups
        )
        return dx, None, dgamma, dbeta, None, None 

# ---------------------------------------------------------
# Masked Pooling Head Kernels   
# ---------------------------------------------------------

@triton.jit 
def _masked_pooling_kernel(
    x_ptr, m_ptr, out_ptr, 
    N, C, H, W, 
    p_val, eps, 
    BLOCK: tl.constexpr
): 
    pid = tl.program_id(0)
    n   = pid // C 
    c   = pid % C 
    HW  = H * W 

    x_offset    = pid * HW 
    mask_offset = n * HW 

    acc_sum_pos = tl.zeros((), dtype=tl.float32)
    acc_pow     = tl.zeros((), dtype=tl.float32)
    acc_ent     = tl.zeros((), dtype=tl.float32)
    acc_cnt     = tl.zeros((), dtype=tl.float32)
    acc_max     = tl.full((), float("-inf"), dtype=tl.float32)

    for k in range(0, HW, BLOCK): 
        offset = k + tl.arange(0, BLOCK)
        mask_k = offset < HW 

        x_raw = tl.load(
            x_ptr + x_offset + offset, mask=mask_k, other=float("-inf")).to(tl.float32)
        m     = tl.load(m_ptr + mask_offset + offset, mask=mask_k, other=0.0)
        m     = m > 0 

        x     = tl.where(m, x_raw, 0.0)
        x_max = tl.where(m, x_raw, float("-inf"))

        x_pos = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        x_pos = tl.where(m, x_pos, 0.0)

        x_clamp = tl.maximum(x, eps)
        x_clamp = tl.where(m, x_clamp, 0.0)

        acc_cnt     += tl.sum(tl.where(m, 1.0, 0.0), axis=0)
        acc_max      = tl.maximum(acc_max, tl.max(x_max, axis=0))
        acc_sum_pos += tl.sum(x_pos, axis=0)

        acc_pow     += tl.sum(tl.exp(tl.log(x_clamp) * p_val) * tl.where(m, 1.0, 0.0), axis=0)

        log_x        = tl.where(x_pos > eps, tl.log(x_pos), 0.0)
        acc_ent     += tl.sum(x_pos * log_x, axis=0)

    denom    = tl.maximum(acc_cnt, eps)
    has_data = acc_cnt > 0.5 
    mean     = acc_sum_pos / denom 

    acc_sq_diff = tl.zeros((), dtype=tl.float32)

    for k in range(0, HW, BLOCK): 
        offset = k + tl.arange(0, BLOCK)
        mask_k = offset < HW 

        x_raw = tl.load(
            x_ptr + x_offset + offset, mask=mask_k, other=float("-inf")).to(tl.float32)
        m     = tl.load(m_ptr + mask_offset + offset, mask=mask_k, other=0.0)
        m     = m > 0 

        x     = tl.where(m, x_raw, 0.0) 
        x_pos = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        x_pos = tl.where(m, x_pos, 0.0)

        diff  = tl.where(m, x_pos - mean, 0.0)
        acc_sq_diff += tl.sum(diff * diff, axis=0)

    res_logsum = tl.log(1.0 + acc_sum_pos)

    res_gem    = tl.exp(tl.log(acc_pow / denom) / p_val)
    res_gem    = tl.where(has_data, res_gem, 0.0)

    res_max    = tl.where(has_data, acc_max, 0.0)

    res_var = acc_sq_diff / denom 
    res_var = tl.where(has_data, res_var, 0.0)

    A = tl.log(acc_sum_pos + eps)
    B = acc_ent / (acc_sum_pos + eps) 
    res_ent = A - B 
    res_ent = tl.where(has_data, res_ent, 0.0)

    base_out = n * (5 * C) + c 

    tl.store(out_ptr + base_out + (0 * C), res_logsum) # redundant but explicitly the 0th offset
    tl.store(out_ptr + base_out + (C), res_gem)
    tl.store(out_ptr + base_out + (2 * C), res_max)
    tl.store(out_ptr + base_out + (3 * C), res_ent)
    tl.store(out_ptr + base_out + (4 * C), res_var)

@triton.jit 
def _masked_pooling_backward_kernel(
    dx_ptr, dy_ptr, x_ptr, m_ptr, 
    N, C, H, W, 
    p_val, eps, 
    BLOCK: tl.constexpr
): 

    pid = tl.program_id(0)
    n   = pid // C 
    c   = pid % C 
    HW  = H * W 

    x_offset    = pid * HW 
    mask_offset = n * HW 

    base_out    = n * (5 * C) + c 
    dy_logsum   = tl.load(dy_ptr + base_out)
    dy_gem      = tl.load(dy_ptr + base_out + C)
    dy_max      = tl.load(dy_ptr + base_out + 2*C)
    dy_ent      = tl.load(dy_ptr + base_out + 3*C)
    dy_var      = tl.load(dy_ptr + base_out + 4*C)

    acc_sum_pos = tl.zeros((), dtype=tl.float32)
    acc_sq_pos  = tl.zeros((), dtype=tl.float32)
    acc_pow     = tl.zeros((), dtype=tl.float32)
    acc_ent     = tl.zeros((), dtype=tl.float32)
    acc_cnt     = tl.zeros((), dtype=tl.float32)
    acc_max     = tl.full((), float("-inf"), dtype=tl.float32)

    for k in range(0, HW, BLOCK): 
        offset = k + tl.arange(0, BLOCK)
        mask_k = offset < HW 

        x_raw = tl.load(
            x_ptr + x_offset + offset, mask=mask_k, other=float("-inf")).to(tl.float32)
        m     = tl.load(m_ptr + mask_offset + offset, mask=mask_k, other=0.0)
        m     = m > 0 

        x     = tl.where(m, x_raw, 0.0)
        x_max = tl.where(m, x_raw, float("-inf"))

        x_pos = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        x_pos = tl.where(m, x_pos, 0.0)

        x_clamp = tl.maximum(x, eps)
        x_clamp = tl.where(m, x_clamp, 0.0)

        acc_cnt     += tl.sum(tl.where(m, 1.0, 0.0), axis=0)
        acc_max      = tl.maximum(acc_max, tl.max(x_max, axis=0))
        acc_sum_pos += tl.sum(x_pos, axis=0)
        acc_sq_pos  += tl.sum(x_pos * x_pos, axis=0)
        acc_pow     += tl.sum(tl.exp(tl.log(x_clamp) * p_val) * tl.where(m, 1.0, 0.0), axis=0)
        log_x        = tl.where(x_pos > eps, tl.log(x_pos), 0.0)
        acc_ent     += tl.sum(x_pos * log_x, axis=0)

    denom = tl.maximum(acc_cnt, eps)
    S     = acc_sum_pos + eps 
    Y_gem = tl.exp(tl.log(acc_pow / denom) / p_val) 
    H_val = tl.log(S) - (acc_ent / S)

    for k in range(0, HW, BLOCK): 
        offset = k + tl.arange(0, BLOCK)
        mask_k = offset < HW 

        x_raw = tl.load(
            x_ptr + x_offset + offset, mask=mask_k, other=float("-inf")).to(tl.float32)
        m     = tl.load(m_ptr + mask_offset + offset, mask=mask_k, other=0.0)
        m     = m > 0 

        x     = tl.where(m, x_raw, 0.0)

        sig_x = 1.0 / (1.0 + tl.exp(-x))
        sig_x = tl.where(m, sig_x, 0.0)

        x_pos = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        x_pos = tl.where(m, x_pos, 0.0)

        grad_logsum = (1.0 / (1.0 + S)) * sig_x * dy_logsum 

        x_clamp  = tl.maximum(x, eps)
        term_pow = tl.exp(tl.log(x_clamp) * (p_val - 1.0)) * tl.where(m, 1.0, 0.0)
        grad_gem = dy_gem * Y_gem * term_pow / (acc_pow + eps)

        is_max   = (x_raw >= acc_max) & m 
        grad_max = tl.where(is_max, dy_max, 0.0)

        mean_val = S / denom 
        grad_var = dy_var * (2.0 / denom) * (x_pos - mean_val) * sig_x 

        log_x    = tl.where(x_pos > eps, tl.log(x_pos), 0.0)
        grad_ent = dy_ent * (1.0 / S) * (H_val - 1.0 - log_x) * sig_x 

        total_dx = grad_logsum + grad_gem + grad_max + grad_var + grad_ent 
        total_dx = tl.where(m, total_dx, 0.0)

        tl.store(dx_ptr + x_offset + offset, total_dx, mask=mask_k)
    

# ---------------------------------------------------------
# Masked Pooling Head Triton Calls    
# ---------------------------------------------------------

def masked_pooling_triton(x, mask, p=3.0, eps=1e-6): 
    assert x.is_cuda and mask.is_cuda 

    if not x.is_contiguous(): 
        x = x.contiguous() 

    if not mask.is_contiguous(): 
        mask = mask.contiguous() 

    if mask.ndim == 4: 
        mask = mask[:, 0]

    N, C, H, W = x.shape 

    y = torch.empty((N, 5 * C), device=x.device, dtype=torch.float32)

    grid = (N * C,)

    _masked_pooling_kernel[grid](
        x, mask, y, 
        N, C, H, W,
        float(p), float(eps),
        BLOCK=1024
    )

    return y  

def masked_pooling_backward_triton(x, mask, dy, p, eps): 

    assert x.is_cuda and mask.is_cuda 

    if not x.is_contiguous(): 
        x = x.contiguous() 

    if not mask.is_contiguous(): 
        mask = mask.contiguous() 

    if mask.ndim == 4: 
        mask = mask[:, 0]

    if not dy.is_contiguous(): 
        dy = dy.contiguous() 

    if mask.ndim == 4: 
        mask = mask[:, 0]

    N, C, H, W = x.shape 

    dx = torch.empty_like(x)

    grid = (N * C,)
    _masked_pooling_backward_kernel[grid](
        dx, dy, x, mask, 
        N, C, H, W, 
        float(p), float(eps),
        BLOCK=1024
    )

    return dx 

class MaskedPoolingTritonFn(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, x, mask, p, eps):
        y = masked_pooling_triton(x, mask, p, eps)
        ctx.save_for_backward(x, mask)
        ctx.p   = p
        ctx.eps = eps 

        return y 

    @staticmethod 
    def backward(ctx, dy): 
        x, mask = ctx.saved_tensors 
        p   = ctx.p 
        eps = ctx.eps 

        dx  = masked_pooling_backward_triton(x, mask, dy, p, eps)
        return dx, None, None, None

def masked_pooling(x, mask, p=3.0, eps=1e-6): 
    y = MaskedPoolingTritonFn.apply(x, mask, p, eps)
    if y is None: 
        raise RuntimeError

    N, total_C = y.shape 
    C    = total_C // 5 

    log_sum = y[:, :C]
    gem     = y[:, C:2*C]
    mx      = y[:, 2*C:3*C]
    entropy = y[:, 3*C:4*C]
    var     = y[:, 4*C:5*C]

    return log_sum, gem, mx, entropy, var
