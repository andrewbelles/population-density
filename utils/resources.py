#!/usr/bin/env python3 
# 
# resources.py  Andrew Belles  Jan 20th, 2026 
# 
# Dynamic resource control to respect distribute computing network 
# shared pool of resources  
# 

import os, torch 

from dataclasses import dataclass 

from collections import OrderedDict 


class LRUCache: 
    def __init__(
        self, 
        *,
        max_bytes: int | None = None, 
        max_items: int | None = None 
    ): 
        self.max_bytes = max_bytes 
        self.max_items = max_items 
        self.total     = 0 
        self.data      = OrderedDict()

    def get(self, key): 
        if key not in self.data: 
            return None 
        val, size = self.data.pop(key)
        self.data[key] = (val, size)
        return val 

    def put(self, key, value): 
        size = self._sizeof(value)
        self.data[key] = (value, size)
        self.total += size 
        self._evict() 

    def _sizeof(self, value) -> int: 
        size = 0 
        for arr in value: 
            if hasattr(arr, "nbytes"): 
                size += int(arr.nbytes)
        return size 

    def _evict(self): 
        while True: 
            too_many = self.max_items is not None and len(self.data) > self.max_items 
            too_big  = self.max_bytes is not None and self.total > self.max_bytes 
            if not (too_many or too_big): 
                break 
            _, (_, size) = self.data.popitem(last=False)
            self.total -= size 

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

    def visible_devices(self) -> list[int]: 
        if self.device != "cuda": 
            return []
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if raw in ("", "-1"):
            return []
        tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
        return list(range(len(tokens))) 
