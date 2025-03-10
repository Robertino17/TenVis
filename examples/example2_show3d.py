import tenvis
import numpy as np
import matplotlib.pyplot as plt

#make the array with some values
array = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)

t = tenvis.Tensor(array)

#lets see a slice along the axis2 with index = 7 with a log scale
t.show_3d(2, 7, log=True)
