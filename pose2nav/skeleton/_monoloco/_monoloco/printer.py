"""
Class for drawing frontal, bird-eye-view and multi figures
"""
# pylint: disable=attribute-defined-outside-init
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np

def draw_orientation(ax, centers, sizes, angles, colors, mode):
    """
    Draw orientation for both the frontal and bird eye view figures
    """

    if mode == 'front':
        length = 5
        fill = False
        alpha = 0.6
        zorder_circle = 0.5
        zorder_arrow = 5
        linewidth = 1.5
        edgecolor = 'k'
        radiuses = [s / 1.2 for s in sizes]
    else:
        length = 1.3
        linewidth = 2.3
        head_width = 0.3
        radiuses = [0.2] * len(centers)
        fill = True
        alpha = 1
        zorder_circle = 2
        zorder_arrow = 1

    for idx, theta in enumerate(angles):
        radius = radiuses[idx]
        color = colors[idx]

        if mode == 'front':
            x_arr = centers[idx][0] + (length + radius) * math.cos(theta)
            z_arr = length + centers[idx][1] + (length + radius) * math.sin(theta)
            delta_x = math.cos(theta)
            delta_z = math.sin(theta)
            head_width = max(10, radiuses[idx] / 1.5)

        else:
            edgecolor = colors[idx]
            x_arr = centers[idx][0]
            z_arr = centers[idx][1]
            length += 0.007 * centers[idx][1]  # increase arrow length
            delta_x = length * math.cos(theta)
            # keep into account kitti convention
            delta_z = - length * math.sin(theta)

        circle = Circle(centers[idx], radius=radius, color=color,
                        fill=fill, alpha=alpha, zorder=zorder_circle)
        arrow = FancyArrow(x_arr, z_arr, delta_x, delta_z, head_width=head_width, edgecolor=edgecolor,
                           facecolor=color, linewidth=linewidth, zorder=zorder_arrow, label='Orientation')
        ax.add_patch(circle)
        ax.add_patch(arrow)
        if mode == 'bird':
            ax.legend(handles=[arrow])


def social_distance_colors(colors, dic_out):
    # Prepare color for social distancing
    colors = ['r' if flag else colors[idx] for idx,flag in enumerate(dic_out['social_distance'])]
    return colors

def draw_uncertainty(ax, centers, stds):
    for idx, std in enumerate(stds):
        std = stds[idx]
        theta = math.atan2(centers[idx][1], centers[idx][0])
        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)
        x = (centers[idx][0] - delta_x, centers[idx][0] + delta_x)
        z = (centers[idx][1] - delta_z, centers[idx][1] + delta_z)
        ax.plot(x, z, color='g', linewidth=2.5)


class BirdEyeViewPlot:
    def __init__(self, z_max):
        self.z_max = z_max
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 8))
        self.fig.set_tight_layout(True)
        x_max = z_max / 1.5

        self.ax.plot([0, x_max], [0, z_max], 'k--')
        self.ax.plot([0, -x_max], [0, z_max], 'k--')
        self.ax.set_ylim(0, z_max + 1)
        self.ax.set_xlim(-x_max - 1, x_max + 1)
        self.ax.set_xlabel("X (left/right, m)")
        self.ax.set_ylabel("Z (forward, m)")
        self.ax.set_title("Top-Down View")
        self.ax.grid(True)

    def get_axis(self):
        return self.ax

    def render(self):
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(self.fig)
        return img