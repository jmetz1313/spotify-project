# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 19:12:42 2025

@author: Ryan
"""

# utils.py

import numpy as np

def get_embedding_twitter(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def str_to_embedding(s):
    s = s.strip('[]')
    return np.array([float(x) for x in s.split()])

def get_index_by_title(title):
    matches = combined_df[combined_df['Song'].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError("❌ Song not found. Double-check the title.")
    return matches.index[0]
