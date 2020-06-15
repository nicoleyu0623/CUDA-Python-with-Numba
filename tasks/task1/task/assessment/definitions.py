# Use the 'File' menu above to 'Save' after pasting in your imports, data, and function definitions.

# Remember that we can't use numpy math function on the GPU...
# from numpy import exp
import math

# Consider modifying the 3 values in this cell to optimize host <-> device memory movement
normalized = np.empty_like(greyscales)
weighted = np.empty_like(greyscales)
activated = np.empty_like(greyscales)

d_greyscales = cuda.to_device(greyscales)
d_weights = cuda.to_device(weights)

d_normalized = cuda.to_device(noise)
d_weighted = cuda.to_device(weighted)
d_activated = cuda.to_device(normalized)

# Modify these 3 function calls to run on the GPU
@vectorize(['float32(float32)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32, float32)'], target='cuda')
def weigh(values, weights):
    return values * weights
 
@vectorize(['float32(float32)'], target='cuda')
def activate(values):
#     return ( np.exp(values) - np.exp(-values) ) / ( np.exp(values) + np.exp(-values) )
    return ( math.exp(values) - math.exp(-values) ) / ( math.exp(values) + math.exp(-values) )