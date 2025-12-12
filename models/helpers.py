#!/usr/bin/env python3 
# 
# helpers.py  Andrew Belles  Dec 11th, 2025 
# 
# Module of helper functions for models to utilize, mostly for path concatenation
# 
# 

import os 

def project_path(*args):
    root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *args)

