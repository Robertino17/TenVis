import tenvis
import numpy as np

#make the array with some values
array = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)
array[7, 10, :, 15] = 25
t = tenvis.Tensor(array)

#lets see a slice along the axis1 with index = 10 and along the axis3 with index = 15
t.show_2d(1, 10, 3, 15, file_name="example4_show2d.png")

#now see 2d projection with a log scale
array2 = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)
array2[2, 5, 7, :] = 25
t = tenvis.Tensor(array2)
t.show_2d(0, 2, 1, 5, log=True)
