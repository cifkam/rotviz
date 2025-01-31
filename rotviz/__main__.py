import importlib.resources
import json
from argparse import ArgumentParser

import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import trimesh
from matplotlib import pyplot as plt

from .meshview import MeshViewer


def get_asset_path(filename: str):
    return importlib.resources.files("rotviz").joinpath("assets", filename)

def plot_sphere(ax, r=0.98, edgecolor="black", color="w"):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)*r
    y = np.sin(u)*np.sin(v)*r
    z = np.cos(v)*r
    ax.plot_surface(x, y, z, color=color, edgecolor=edgecolor)


def fade_colors_with_distance(pts, base_colors, ax):
    x,y,z = pts.T
    # Get camera angles
    azim = np.radians(ax.azim)  # Convert to radians
    elev = np.radians(ax.elev)

    # Compute the viewpoint direction (unit vector)
    view_dir = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ])

    # Compute the "distance" of each point from the viewer (dot product)
    distances = x * view_dir[0] + y * view_dir[1] + z * view_dir[2]

    # Normalize distances for transparency effect
    alpha_values = distances - np.min(distances)
    alpha_values /= np.max(alpha_values)
    #min_dist, max_dist = np.min(distnces), np.max(distances)
    #alpha_values = (distances - min_dist) / (max_dist - min_dist)
    
    @np.vectorize
    def enhance(x):
        return 3*x**2 - 2*x**3
        
    alpha_values = enhance(alpha_values) 

    # Generate colors with dynamic opacity
    adjusted_colors = [mcolors.to_rgba(c, alpha=a) for c, a in zip(base_colors, alpha_values)]
    return adjusted_colors


if __name__ == '__main__':
    parser = ArgumentParser(description='Annotate the coarse pose of the object in the image')
    parser.add_argument('--data', type=str, help='Path to the folder containing the data', required=True)
    parser.add_argument('--cmap', '-c', type=str, help='Colormap to use for coloring', default='viridis')
    parser.add_argument('--pt-size', '-s', type=int, help='Size of the scattered points', default=50)

    parser.add_argument('--mesh', '-m', type=str, help='Mesh path', default=None)
    parser.add_argument('--focal-length', '-f', type=float, help='Focal length of the camera', default=None)
    parser.add_argument('--window-size', '-w', type=int, nargs=2, help='Size of the mesh window', default=(500, 500))
    parser.add_argument('--scale', type=float, help='Scale of the mesh', default=0.2)
    parser.add_argument('--speed', type=float, help='Speed of the rotation with mouse', default=0.6)
    parser.add_argument('--hide-axes', action='store_true', help='Hide the xyz axes')
    parser.add_argument('--inverse', '-i', action='store_true', help='Invert the rotation matrices')
    args = parser.parse_args()

    if args.mesh is None:
        mesh_path = get_asset_path('cube.ply')
        args.mesh = str(mesh_path)

    with open(args.data, 'r') as f:
        data = json.load(f)

    rotations = np.array([np.array(x['T_WorldFromCamera'])[:3,:3] for x in data])
    if args.inverse:
        rotations = rotations.transpose(0, 2, 1)

    scores = np.array([x['score'] for x in data])
    scores -= scores.min()
    scores /= scores.max()
    colors = matplotlib.colormaps[args.cmap](scores)

    mesh = trimesh.load_mesh(args.mesh)
    mesh.apply_scale(args.scale)

    viewer = MeshViewer(mesh,
                        window_name='Mesh',
                        window_size=args.window_size,
                        focal_length=args.focal_length,
                        mouse_speed=args.speed, 
                        axes_scale=0.3,
                        ambient_light=2.0)
    
    pose = viewer.pose
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def setup_ax():
        plot_sphere(ax, edgecolor=(0,0,0,0.06), color=(1,1,1,0.1))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.grid(False)
        
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((0.975, 0.975, 0.975, 1.0))
        
    
    def compute_points():
        # Multiply the rotation matrices by the camera z-axis vector
        cam_axis = viewer.pose[2,:3]/np.linalg.norm(viewer.pose[2,:3])
        vectors = rotations @ cam_axis
        return vectors
    
    def plot(event=None, base_colors=colors):
        pts = compute_points()
        c = fade_colors_with_distance(pts, base_colors, ax)

        # Redraw scatter plot with updated colors
        ax.clear()
        ax.scatter(*pts.T, c=c, s=args.pt_size, edgecolors=(0,0,0,0.15))
        setup_ax()
        plt.draw()
    

    fig.canvas.mpl_connect('motion_notify_event', plot)
    plot()
    plt.show(block=False)

    for pose in viewer.run_yield(delay=50, always_yield=True):        
        if pose is not None:
            ax.clear()
            plot()
        
        if not plt.get_fignums():
            break
        
        fig.canvas.flush_events()

    cv2.destroyAllWindows()
    plt.close()