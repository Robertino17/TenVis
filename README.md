# TenVis
TenVis - is a very simple library for visualizing 4D tensors 20x20x20x20
## Install
```bash
git clone https://github.com/Robertino17/TenVis.git
pip install -e.
```
## Usage
Using the library is very simple : 
```python
import tenvis
t = tenvis.Tensor(your_numpy_array)
t.show_3d(1, 2)
```
In this example, we can see a slice along the axis1 with index = 2.
For more information, please check [report_task](report_task.pdf) and [examples](examples/)
