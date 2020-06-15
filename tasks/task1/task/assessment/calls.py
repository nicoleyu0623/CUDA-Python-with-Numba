# Use the 'File' menu above to 'Save' after pasting in your 3 function calls.

%%timeit
# Feel free to modify the 3 function calls in this cell
# normalized = normalize(greyscales)
# weighted = weigh(normalized, weights)
# SOLUTION = activate(weighted)
from numba import cuda

do_n = cuda.device_array(shape=(n,), dtype=np.float32)
do_w = cuda.device_array(shape=(n,), dtype=np.float32)
do_a = cuda.device_array(shape=(n,), dtype=np.float32)

normalize(d_greyscales, out=do_n)
weigh(do_n, d_weights, out=do_w)
activate(do_w, out=do_a)
SOLUTION = do_a.copy_to_host()
print(SOLUTION)
