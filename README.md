# TenVis
```bash
git clone https://github.com/Robertino17/TenVis.git
pip install -e.
```
Using the library is very simple : 
```python
import tenvis
t = tenvis.Tensor(your_numpy_array)
t.show_3d(1, 2)
```
In this example, we can see a slice along the axis1 with index = 2.
For more information, please check [report_task](report_task.pdf) and [examples](examples/)
