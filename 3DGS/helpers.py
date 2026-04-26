# pulled from the 2DGS A4 assignment, in case we need it :P
from datetime import datetime
import math
from typing import Union

from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt

import numpy as np

import torch


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


#@title Domain2D: Not sure if I will need this but we will start here
class Domain2D:
    """
    Helper function for handling a 2D domain, discretized on a grid of given resolution.
    """
    x_dim: tuple[float]
    y_dim: tuple[float]
    res_x: float
    res_y: float
    xx: torch.Tensor
    yy: torch.Tensor

    def __init__(self, x_dim, y_dim, res_x, res_y, device='cpu'):
        self.x_dim, self.y_dim = x_dim, y_dim
        self.res_x, self.res_y = res_x, res_y

        self.xx, self.yy = torch.meshgrid(
            torch.linspace(self.x_dim[0], self.x_dim[1], self.res_x, device=device),
            torch.linspace(self.y_dim[0], self.y_dim[1], self.res_y, device=device),
            indexing='xy'
        )

    def __str__(self):
        return (f"Domain2D: {self.x_dim}x{self.y_dim} "
                f"discretized on a {self.res_x}x{self.res_y} grid.")
    
    def get_extent(self):
        extent = (*self.x_dim, *self.y_dim)
        return extent


#@title Timer
class Timer:
    """
    Super simple utility class for displaying elapsed time.
    """

    def __init__(self):
        self.init_time = datetime.now()
        # Step 1: Print the current time
        print("Current time:", self.init_time.strftime(TIME_FORMAT))

    def print_time(self, reset_time = False):
        # Print the time and the time elapsed since init_time
        current_time = datetime.now()
        elapsed_time = current_time - self.init_time

        print("Current time:", current_time.strftime(TIME_FORMAT))
        print("Time elapsed:", str(elapsed_time))

        # Optionally overwrite initial time
        if reset_time:
            self.init_time = current_time

    def get_elapsed_time(self) -> str:
        delta_time = datetime.now() - self.init_time
        return "{:02}:{:02}:{:02}".format(
                delta_time.seconds // 3600,
                (delta_time.seconds % 3600) // 60,
                (delta_time.seconds % 60) // 1,
                )


#@title Plotting

def plot_tensor_image(
        image,
        title: str = "",
        _plt=plt,
        extent=None,
):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    if image.shape[-1] == 3:
        image = np.clip(image, 0, 1)
        # plot (R,G,B) image
        #  Clip image colors
        
    _plt.imshow(
        image,
        extent=extent,  # (left, right, bottom, top)
        origin="lower"
    )

    if _plt == plt:
        _plt.title(title)
    else:
        _plt.title.set_text(title)


def draw_outline(params, _plt=plt):
    centers, sigmas, thetas = params["centers"], params["sigmas"], params["thetas"]
    # Convert to basic numpy array if necessary,
    # bringing tensors to the cpu and detach from computing graph
    if isinstance(centers, torch.Tensor):
        centers = centers.cpu().detach().numpy()
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.cpu().detach().numpy()
    if isinstance(thetas, torch.Tensor):
        thetas = thetas.cpu().detach().numpy()

    if not isinstance(_plt, plt.Axes):
        _plt = _plt.gca()

    # plot centers as red dots
    _plt.scatter(x=centers[:, 0], y=centers[:, 1], c='r', s=1)

    # draw ellipses
    for i, c in enumerate(centers):
        _plt.add_patch(
            Ellipse(
                (c[0], c[1]),
                width=sigmas[i][0] * 3.5,
                height=sigmas[i][1] * 3.5,
                angle=thetas[i] * (180.0 / np.pi),
                edgecolor='red',
                facecolor='none',
                linewidth=1,
                alpha=1.0
            )
        )


def assemble_plot_data(data, outline_params):
    """
    Make the data ready for plotting with the `plot` function.
    Calculates the number of rows and columns to be plotted based on the supplied data.

    :param data: image data, or iterable of image data
    :param outline_params: gaussian parameters used for plotting an overlay over the reconstructed image
    :return: number of columns (int), rows (int), assembled data (np.ndarray) and outline parameters (list)
    """

    def is_iterable(x):
        return isinstance(x, list) or isinstance(x, tuple)

    def is_2d_iterable(x):
        return isinstance(x[0], list) or isinstance(x[0], tuple)

    def get_numpy_data(d):
        # Takes a single piece of plottable data, and makes it a numpy array
        if isinstance(d, torch.Tensor):
            d = d.cpu().detach().numpy()
        d = np.array(d)
        return d

    # Create a 2D array of plottable data [[row_1_1, row_1_2 ...],[row_2_1, row_2_2, ...], ...]
    # Each piece of data is uniformly converted to a numpy array
    new_data = []
    if is_iterable(data):
        if is_2d_iterable(data):
            for i in range(len(data)):
                curr_row = []
                for j in range(len(data[i])):
                    curr_data = get_numpy_data(data[i][j])
                    curr_row.append(curr_data)
                new_data.append(curr_row)
        else:
            single_row = []
            for i in range(len(data)):
                curr_data = get_numpy_data(data[i])
                single_row.append(curr_data)
            new_data.append(single_row)
    else:
        # Single piece of data, but we still create a 2D array
        new_data.append([get_numpy_data(data)])

    data = new_data

    # Calculate number of rows and columns in the plot
    nrows = len(data)
    ncols = len(data[0])

    # Handle outline_params
    if isinstance(outline_params, dict):
        # If plotting only a single piece of data
        assert ncols == nrows == 1, "Non-list outline params is only allowed for plotting a single data."
        outline_params = [[outline_params]]
    if outline_params is not None:
        if not all(p is None for p in outline_params):
            if not isinstance(outline_params[0], list):
                # If not already a 2D array, then
                # wrap outline_params to be a 2D array for [row][col] indexing
                outline_params = [outline_params]
        else:
            outline_params = None

    # We could assert that outline_params should be None, or having the same shape as the data

    return ncols, nrows, data, outline_params


def assemble_titles(title, ncols, nrows):
    """
    :param title: string or list of strings
    :param ncols: number of columns in the plot
    :param nrows: number of rows in the plot
    :return: a 2D list of titles corresponding to a (ncols, nrows) plot.
    """
    if isinstance(title, (tuple, list, dict)):
        if isinstance(title[0], (tuple, list, dict)):
            # Title is a 2D array of titles for each subplot, individually
            assert len(title) == nrows and len(title[0]) == ncols
        else:
            # Wrap 1D list of titles to be a 2D array for a single row
            title = [title]
    else:
        # Same title for each subplot
        title = [[title] * ncols] * nrows

    return title


def plot(
        data: Union[torch.Tensor, list, tuple, np.ndarray],
        title: Union[str, list] = "",
        figsize=(16, 6),
        extent=None,
        domain: Domain2D = None,  # if domain is not None, then it overwrite extent, xx and yy
        outline_params=None,
        show_plot=True,
):
    """
    Main plotting function.
    """
    ncols, nrows, data, outline_params = assemble_plot_data(data, outline_params)
    titles = assemble_titles(title, ncols, nrows)

    # Shape of axs, and existence of dimensions is dependent on number of rows and columns. If figure is (1,1)
    # or either ncols or nrows is 1, then at least 1 of the dimensions will be missing from the axs list.
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # Make sure that axs is a 2D list of all subplot axes which can be indexed as `axs[row_i][col_j]`.
    if nrows == ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [axs]  # axs is a 1D list
    elif ncols == 1:
        axs = [[axs[i]] for i in range(len(axs))]  # axs is a 1D list, but we have to reshape it

    # Set extent, xx and yy from domain if it was supplied
    if domain is not None:
        extent = domain.get_extent()

    for i in range(nrows):
        for j in range(ncols):
            curr_data = data[i][j]
            curr_ax = axs[i][j]
            curr_title = titles[i][j]

            if len(curr_data.shape) > 1:
                # Plot scalar field or image data
                plot_tensor_image(
                    image=curr_data,
                    title=curr_title,
                    _plt=curr_ax,
                    extent=extent,
                )
                if outline_params is not None and outline_params[i][j] is not None:
                    draw_outline(params=outline_params[i][j], _plt=curr_ax)
                curr_ax.set_aspect('equal')
            elif len(curr_data.shape) == 1:
                # Plot 1D data, e.g. loss curve
                curr_ax.plot(curr_data)
                curr_ax.set_xticks(range(0, len(curr_data), math.ceil(len(curr_data) / 8)))
                curr_ax.set_title('Learning curve')
                curr_ax.set_xlabel('Epoch')
                curr_ax.set_ylabel('Loss')

    if show_plot:
        plt.show()

    return fig
