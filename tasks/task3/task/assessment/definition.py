# Use the 'File' menu above to 'Save' after pasting in your own mm_shared function definition.
import numpy as np
from numba import cuda, types

@cuda.jit
def mm_shared(A, B, C):
    x, y = cuda.grid(2)
    sum = 0
    
    a_cache = cuda.shared.array(block_size, types.int32)
    b_cache = cuda.shared.array(block_size, types.int32)


    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = M/N  # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    for i in range(bpg):
        # Preload data into shared memory
        a_cache[tx, ty] = A[x, ty + i * N]
        b_cache[tx, ty] = B[tx + i * N, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(N):
            sum += a_cache[tx, j] * b_cache[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = sum
