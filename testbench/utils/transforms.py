#!/usr/bin/env python3 
# 
# transforms.py  Andrew Belles  Jan 7th, 2026 
# 
# Transform Helper functions for dataset transformation prescaling in Cross Validator 
# 
# 

from typing import Iterable

def compose_transform_factories(*factories): 
    def _factory(feature_names): 
        transforms = []
        for f in factories: 
            if f is None: 
                continue 
            out = f(feature_names)
            if callable(out): 
                v = out() 
            else: 
                v = out 
            if not isinstance(v, Iterable): 
                raise TypeError("factory must return iterable")
            transforms.extend(v)
        return transforms 
    return _factory

def apply_transforms(X, transforms):
    for t in transforms: 
        if hasattr(t, "fit_transform"):
            X = t.fit_transform(X)
        else: 
            t.fit(X)
            X = t.transform(X)
    return X 
