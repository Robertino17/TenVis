import tenvis
import numpy as np
import matplotlib.pyplot as plt

#make the array with some values
array = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)

array[:, 5, 5:7, :] = 255
array[:, 5, :, 15:17] = 255

t = tenvis.Tensor(array)

#lets see a slice along the axis1 with index = 5 with filtration and slightly transparent dots
t.show_3d(1, 5, min_value=230, alpha=0.9, file_name="example3_show3d.png") # you will see two white planes