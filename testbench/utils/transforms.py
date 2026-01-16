#!/usr/bin/env python3 
# 
# transforms.py  Andrew Belles  Jan 7th, 2026 
# 
# Transform Helper functions for dataset transformation prescaling in Cross Validator 
# 
# 

def apply_transforms(X, transforms):
    for t in transforms: 
        if hasattr(t, "fit_transform"):
            X = t.fit_transform(X)
        else: 
            t.fit(X)
            X = t.transform(X)
    return X 
