import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class Tensor:
    """
    A class for visualizing 4D tensor 20x20x20x20

    
    Attributes
    ---------
    tensor : np.ndarray
        Array for visualization.
    """

    def __init__(self, tensor, copy=True):

        """
        Initialize the Tensor object.

        Parameters
        ----------
        tensor : np.ndarray
            The 4D tensor to visualize.
        copy : bool
            If True, a copy of the tensor is made. If False, the tensor is used as-is.
        """

        if not isinstance(tensor, (np.ndarray)):
            raise TypeError(f"Expected numpy.ndarray, but got {type(tensor)}")
        
        if tensor.shape != (20, 20, 20, 20):
            raise TypeError(f"Expected shape = (20, 20, 20, 20), but got {tensor.shape}")
        
        self.tensor = np.copy(tensor) if copy else tensor
    
    def show_3d(self, axis, index, size=(24, 13.5), max_value=255, min_value=0,
                alpha=1, log=False, colorbar=True, file_name=None, show=True):
        """
        Visualize 3D slice of the tensor

        Parameters
        ----------
        axis : int
            Axis for the slice.
        index : int
            The index by which the slice of the selected axis is assigned.
        size : tuple
            Size of the plot.
        max_value : int
            Maximum value for filtration.
        min_value : int
            Minimum value for filtration.
        alpha : float
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        log : bool
            Use a log scale.
        colorbar : bool
            Use a colorbar.
        file_name : str
            The name to save the file. If None, the file will not be saved.
        show : bool
            Show a plot.
        """
        if axis not in {0, 1, 2, 3}:
            raise ValueError("Axis must be 0, 1, 2, 3")
        
        if max_value < min_value:
            raise ValueError("min_value must be less than max_value")
        
        #make indexes for 3d projection
        slices = [slice(None)] * 4
        slices[axis] = index

        #make coordinates for values 
        condition = (self.tensor[tuple(slices)] >= min_value) & (self.tensor[tuple(slices)] <= max_value)
        Z = np.where(condition)[0]
        Y = np.where(condition)[1]
        X = np.where(condition)[2]
        colors = self.tensor[tuple(slices)][condition]

        #choose the axes names
        axes_names = {
            0: ("axis 1", "axis 2", "axis 3"),
            1: ("axis 0", "axis 2", "axis 3"),
            2: ("axis 0", "axis 1", "axis 3"),
            3: ("axis 0", "axis 1", "axis 2"),
        }

        Z_name, Y_name, X_name = axes_names[axis]

        #make a 3d projection
        fig = plt.figure(figsize=size)
        fig.tight_layout()
        ax = fig.add_subplot(projection='3d')

        norm = "log" if log else None
        
        scatter = ax.scatter(X, Y, Z, s=100, edgecolors="black", linewidths=0.5,
                             c=colors, cmap="gray", alpha=alpha, norm=norm)
        
        if colorbar:
            bar = fig.colorbar(scatter, shrink=0.7, aspect=10)
            bar.ax.tick_params(labelsize=15)
        
        #adjusting the axes
        ax.set_xlabel(X_name, fontsize=15)
        ax.set_ylabel(Y_name, fontsize=15)
        ax.set_zlabel(Z_name, fontsize=15)

        ax.set_xlim(0, 19)
        ax.set_ylim(0, 19)
        ax.set_zlim(0, 19)

        ax.set_title(f"Axis {axis} = {index}", y=0.995, fontsize=15)

        ax.tick_params(labelsize=12)
        plt.locator_params(axis='both', nbins=20)
        plt.tight_layout()

        if file_name:
            if os.path.exists(file_name):
                print(f"Unable to save the file {file_name}, because it already exists.")
            else:
                print(f"Saving {file_name}")
                plt.savefig(file_name, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    
    def show_2d(self, axis1, index1, axis2, index2, size=(15, 12), log=False,
                colorbar=True, file_name=None, show=True):

        """
        Visualize 2D slice of the tensor

        Parameters
        ----------
        axis1 : int
            First axis for slice.
        index1 : int
            The index by which the slice of the selected first axis is assigned.
        axis2 : int
            Second axis for slice.
        index2 : int
            The index by which the slice of the selected second axis is assigned.
        size : tuple
            Size of the plot.
        log : bool
            Use a log scale.
        colorbar : bool
            Use a colorbar.
        file_name : str
            The name to save the file. If None, the file will not be saved.
        show : bool
            Show a plot.
        """

        if axis1 not in {0, 1, 2, 3}:
            raise ValueError("First axis must be 0, 1, 2, 3")
        
        if axis2 not in {0, 1, 2, 3}:
            raise ValueError("Second axis must be 0, 1, 2, 3")
        
        if axis1 == axis2:
            raise ValueError("First axis and second axis must be different")
        
        #make indexes for 2d projection
        slices = [slice(None)] * 4
        slices[axis1] = index1
        slices[axis2] = index2

        #make coordinates for values
        Y = np.array([int(i / 20) for i in range(400)])
        X = np.array([i for i in range(19)] * 20)
        colors = self.tensor[tuple(slices)].reshape(20, 20)

        #choose the axes names
        Y_name, X_name = tuple(set(range(4)) - {axis1, axis2})
        X_name = "Axis " + str(X_name)
        Y_name = "Axis " + str(Y_name)

        #make a 2d projection
        fig = plt.figure(figsize=size)
        fig.tight_layout()
        ax = fig.add_subplot()

        norm = "log" if log else None

        heatmap = ax.imshow(colors, cmap="gray", norm=norm)

        if colorbar:
            bar = fig.colorbar(heatmap, shrink=0.7, aspect=10)
            bar.ax.tick_params(labelsize=15)

        #adjusting the axes
        ax.set_xlabel(X_name, fontsize=15)
        ax.set_ylabel(Y_name, fontsize=15)

        ax.set_title(f"Axis {axis1} = {index1}, Axis {axis2} = {index2}")

        ax.tick_params(labelsize=12)
        plt.locator_params(axis='both', nbins=20)

        if file_name:
            if os.path.exists(file_name):
                print(f"Unable to save the file {file_name}, because it already exists.")
            else:
                print(f"Saving {file_name}")
                plt.savefig(file_name, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def animate_2d(self, axis1, index1, axis2, size=(15, 12), log=False, colorbar=True,
                   file_name=None, interval=1000, show=True):

        """
        Animation 2D slices along the chosen axis.

        Parameters
        ----------
        axis1 : int
            Axis for the slice.
        index1 : int
            The index by which the slice of the selected axis is assigned.
        axis2 : int
            The axis along which the animation will take place.
        size : tuple
            Size of the plot.
        log : bool
            Use a log scale.
        colorbar : bool
            Use a colorbar.
        file_name : str
            The name to save the file. If None, the file will not be saved.
        interval : int
            Delay between frames in milliseconds.
        show : bool
            Show a plot.
        """

        if axis1 not in {0, 1, 2, 3}:
            raise ValueError("First axis must be 0, 1, 2, 3")
        
        if axis2 not in {0, 1, 2, 3}:
            raise ValueError("Second axis must be 0, 1, 2, 3")
        
        if axis1 == axis2:
            raise ValueError("First axis and second axis must be different")
        
        #make indexes for 2d projection
        slices = [slice(None)] * 4
        slices[axis1] = index1

        #we need to do it for the correct colors 
        max_value = self.tensor[tuple(slices)].max()
        min_value = self.tensor[tuple(slices)].min()

        slices[axis2] = 0

        #make coordinates for values
        Y = np.array([int(i / 20) for i in range(400)])
        X = np.array([i for i in range(19)] * 20)
        colors = self.tensor[tuple(slices)].reshape(20, 20)

        #choose the axes names
        Y_name, X_name = tuple(set(range(4)) - {axis1, axis2})
        X_name = "Axis " + str(X_name)
        Y_name = "Axis " + str(Y_name)

        #make a 2d projection
        fig = plt.figure(figsize=size)
        fig.tight_layout()
        ax = fig.add_subplot()

        norm = "log" if log else None

        heatmap = ax.imshow(colors, cmap="gray", vmin=min_value, vmax=max_value, norm=norm)

        if colorbar:
            bar = fig.colorbar(heatmap, shrink=0.7, aspect=10)
            bar.ax.tick_params(labelsize=15)

        #adjusting the axes
        ax.set_xlabel(X_name, fontsize=15)
        ax.set_ylabel(Y_name, fontsize=15)

        ax.tick_params(labelsize=12)
        plt.locator_params(axis='both', nbins=20)

        #function for the next slice
        def next_slice(i):
            slices[axis2] = i
            colors = self.tensor[tuple(slices)].reshape(20, 20)
            heatmap.set_data(colors)
            ax.set_title(f"Axis {axis1} = {index1}, Axis {axis2} = {i}")
            return heatmap,
    

        ani = FuncAnimation(fig, func=next_slice, frames=self.tensor.shape[axis2], interval=interval)

        if show:
            plt.show()
        else:
            plt.close(fig)

        if file_name:
            if os.path.exists(file_name):
                print(f"Unable to save the file {file_name}, because it already exists.")
            else:
                print(f"Saving {file_name}")
                ani.save(file_name, fps=(1000/interval))

    def animate_3d(self, axis, size=(24, 13.5), max_value=255, min_value=0,
                alpha=1, log=False, colorbar=True, file_name=None, interval=1000, show=True):
        
        """
        Animation 3D slices along the chosen axis.

        Parameters
        ----------
        axis : int
            The axis along which the animation will take place.
        size : tuple
            Size of the plot.
        max_value : int
            Maximum value for filtration.
        min_value : int
            Minimum value for filtration.
        alpha : float
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        log : bool
            Use a log scale.
        colorbar : bool
            Use a colorbar.
        file_name : str
            The name to save the file. If None, the file will not be saved.
        interval : int
            Delay between frames in milliseconds.
        show : bool
            Show a plot.
        """
        
        if axis not in {0, 1, 2, 3}:
            raise ValueError("Axis must be 0, 1, 2, 3")
        
        if max_value < min_value:
            raise ValueError("min_value must be less than max_value")
        

        condition_max_min = (self.tensor >= min_value) & (self.tensor <= max_value)
        maximum = self.tensor[condition_max_min].max()
        minimum = self.tensor[condition_max_min].min()
        
        #make indexes for 3d projection
        slices = [slice(None)] * 4
        slices[axis] = 0

        #make coordinates for values 
        condition = (self.tensor[tuple(slices)] >= min_value) & (self.tensor[tuple(slices)] <= max_value)
        Z = np.where(condition)[0]
        Y = np.where(condition)[1]
        X = np.where(condition)[2]
        colors = self.tensor[tuple(slices)][condition]

        #choose the axes names
        axes_names = {
            0: ("axis 1", "axis 2", "axis 3"),
            1: ("axis 0", "axis 2", "axis 3"),
            2: ("axis 0", "axis 1", "axis 3"),
            3: ("axis 0", "axis 1", "axis 2"),
        }

        Z_name, Y_name, X_name = axes_names[axis]

        #make a 3d projection
        fig = plt.figure(figsize=size)
        fig.tight_layout()
        ax = fig.add_subplot(projection='3d')

        norm = "log" if log else None
        
        scatter = ax.scatter(X, Y, Z, s=100, edgecolors="black", linewidths=0.5,
                             c=colors, cmap="gray", alpha=alpha, norm=norm, vmin=minimum, vmax=maximum)
        
        if colorbar:
            bar = fig.colorbar(scatter, shrink=0.7, aspect=10)
            bar.ax.tick_params(labelsize=15)
        
        #adjusting the axes
        ax.set_xlabel(X_name, fontsize=15)
        ax.set_ylabel(Y_name, fontsize=15)
        ax.set_zlabel(Z_name, fontsize=15)

        ax.set_xlim(0, 19)
        ax.set_ylim(0, 19)
        ax.set_zlim(0, 19)


        ax.tick_params(labelsize=12)
        plt.locator_params(axis='both', nbins=20)
        plt.tight_layout()

        def next_slice(i):
            slices[axis] = i
            condition = (self.tensor[tuple(slices)] >= min_value) & (self.tensor[tuple(slices)] <= max_value)
            Z = np.where(condition)[0]
            Y = np.where(condition)[1]
            X = np.where(condition)[2]
            colors = self.tensor[tuple(slices)][condition]

            scatter._offsets3d = (X, Y, Z)
            scatter.set_array(colors)
            ax.set_title(f"Axis {axis} = {i}", y=0.995)
            return scatter,

        ani = FuncAnimation(fig, func=next_slice, frames=self.tensor.shape[axis], interval=1000)

        if show:
            plt.show()
        else:
            plt.close(fig)

        if file_name:
            if os.path.exists(file_name):
                print(f"Unable to save the file {file_name}, because it already exists.")
            else:
                print(f"Saving {file_name}")
                ani.save(file_name, fps=(1000/interval))



    
