#!/usr/bin/env python3 
# 
# resources.py  Andrew Belles  Jan 20th, 2026 
# 
# Dynamic resource control to respect distribute computing network 
# shared pool of resources  
# 

import os, torch 

from dataclasses import dataclass 

@dataclass 
class ComputeStrategy: 
    device: str 
    n_jobs: int 
    gpu_id: int | None 
    greedy: bool 

    @classmethod 
    def create(cls, greedy: bool = False): 
        '''
        Caller Provides: 
            greedy, true forces all cores/GPUs and false honors env 
        '''
        base = cls.from_env()

        # Caller specified greedy
        if greedy: 
            return cls(
                device=base.device,
                n_jobs=-1,
                gpu_id=base.gpu_id, 
                greedy=True 
            )

        # Read strategy was greedy, but caller specified otherwise 
        if base.greedy:
            return cls(
                device=base.device,
                n_jobs=1,
                gpu_id=base.gpu_id, 
                greedy=False 
            )

        # Copy from_env call
        return cls(
            device=base.device,
            n_jobs=base.n_jobs,
            gpu_id=base.gpu_id, 
            greedy=False 
        )

    @classmethod 
    def from_env(cls): 
        raw = os.environ.get("TOPG_JOBS", "1")
        try: 
            jobs = int(raw)
        except ValueError: 
            jobs = 1 

        greedy = jobs <= 0 
        n_jobs = -1 if greedy else max(jobs, 1)

        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        use_gpu  = torch.cuda.is_available() and cuda_vis not in ("", "-1")
        device   = "cuda" if use_gpu else "cpu"
        gpu_id   = 0 if device == "cuda" else None 

        return cls(
            device=device,
            n_jobs=n_jobs,
            gpu_id=gpu_id,
            greedy=greedy
        )

