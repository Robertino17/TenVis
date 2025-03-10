import tenvis
import numpy as np

#make the array with some values
array = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)

#change the array a little
array[1, :, 3:10, :] = 0 
t = tenvis.Tensor(array, copy=False)

#lets see a slice along the axis0 with index = 1
t.show_3d(0, 1) # you will see a huge black part, because values that have coordinate axis2 between 3 and 9 are 0

#now change the array again
array[1, :, :, 17: ] = 0
t.show_3d(0, 1) #you will see another black part, because copy=False

