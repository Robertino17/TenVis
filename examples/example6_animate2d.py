import tenvis
import numpy as np
import matplotlib.pyplot as plt

array = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)
for i in range(19):
    if i % 2 == 0:
        array[i, 5, 5, :] = 255
t = tenvis.Tensor(array)
#animation the slice(axis2, index=5) along axis0 
t.animate_2d(2, 5, 0, file_name="example6_animate2d.gif")
