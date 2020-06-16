# Use the 'File' menu above to 'Save' after pasting in your own mm_shared function definition.
import numpy as np
from numba import cuda, types

@cuda.jit
def mm_shared(a, b, c):
    column, row = cuda.grid(2)
    sum = 0
    
    a_cache = cuda.shared.array(block_size, types.int32)
    b_cache = cuda.shared.array(block_size, types.int32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    for i in range(a.shape[0]):
        # Preload data into shared memory
        a_cache[tx][ty] = a[column][ty + i * N]
        b_cache[tx][ty] = b[tx + i * N][ row]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(a.shape[1]):
            sum += a_cache[tx][j] * b_cache[j][ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    c[column][row] = sum

