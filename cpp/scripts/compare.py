import numpy as np
import sys

def cos_sim(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_sim = dot_product / (norm_A * norm_B)
    return cosine_sim

A = np.fromfile(sys.argv[1], dtype=np.int32)
B = np.fromfile(sys.argv[2], dtype=np.int32)
np.testing.assert_allclose(A, B)
print(cos_sim(A, B))