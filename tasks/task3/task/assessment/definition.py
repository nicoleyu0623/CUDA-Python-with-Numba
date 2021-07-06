# Use the 'File' menu above to 'Save' after pasting in your own mm_shared function definition.

import numpy as np
from numba import cuda, types
@cuda.jit
def mm_shared(a, b, c):
    column, row = cuda.grid(2)
    sum = 0
    stride_column,stride_row=cuda.gridsize(2)

    a_cache = cuda.shared.array(block_size, types.int32)
    b_cache = cuda.shared.array(block_size, types.int32)
    tx=cuda.threadIdx.x
    ty=cuda.threadIdx.y
   
    for i in range(row,a.shape[0],stride_row):
        a_cache[ty,tx]=a[i,tx]   
    cuda.syncthreads()
    for j in range(column,b.shape[1],stride_column):
        b_cache[ty,tx]=b[ty,j]
    cuda.syncthreads()
    for i in range(b_cache.shape[0]):
        sum += a_cache[ty][i] * b_cache[i][tx]

    c[row][column] = sum
