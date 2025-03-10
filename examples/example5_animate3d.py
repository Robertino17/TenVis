import tenvis
import numpy as np

array = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)
for i in range(18):
    array[i, i + 2, :, :] = 0
    array[i, i + 1, :, :] = 0
    array[i, i, :, :] = 0
t = tenvis.Tensor(array)
#animation our array along axis0 
t.animate_3d(0, colorbar=False, max_value=100, interval=2000, alpha=0.8, file_name="example5_animate3d.gif")
