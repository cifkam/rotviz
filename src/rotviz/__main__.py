import trimesh
from argparse import ArgumentParser
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import cv2
import json
from .meshview import MeshViewer
from pathlib import Path


#"""
def plot_sphere(ax, r=0.98, edgecolor="black", color="w"):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)*r
    y = np.sin(u)*np.sin(v)*r
    z = np.cos(v)*r
    ax.plot_surface(x, y, z, color=color, edgecolor=edgecolor)
"""
def plot_sphere_v2(ax, r, lat_res, long_res, rstride,cstride, edgecolor="black", color="w"):
     latitude  = np.linspace(0, 2*np.pi, lat_res)
     longitude = np.linspace(-np.pi, np.pi, long_res)
     x_values = np.outer(np.sin(latitude), np.sin(longitude)) * r
     y_values = np.outer(np.cos(latitude), np.sin(longitude)) * r
     z_values = np.outer(np.ones(lat_res), np.cos(longitude)) * r
 
     ax.plot_wireframe(x_values,y_values,z_values,rstride=rstride,cstride=cstride, edgecolor=edgecolor, color=color)

def plot_sphere(ax, r=0.98, edgecolor="black", color="w"):
    plot_sphere_v2(ax, r, 30, 12, 0, 2, color=edgecolor)
    plot_sphere_v2(ax, r, 12, 30, 1, 0, color=edgecolor)
#"""


def sample_rotations(n_poses):
    from scipy.spatial.transform import Rotation as Rot
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    
    Q = np.empty(shape=(n_poses,4), dtype=float)
    rotations = []
    mesh_poses = []
    for i in range(n_poses):
        s = i+0.5
        r = np.sqrt(s/n_poses)
        R = np.sqrt(1.0-s/n_poses)
        alpha = 2.0 * np.pi * s / phi
        beta = 2.0 * np.pi * s / psi
        Q[i,0] = r*np.sin(alpha)
        Q[i,1] = r*np.cos(alpha)
        Q[i,2] = R*np.sin(beta)
        Q[i,3] = R*np.cos(beta)

        rotations.append(Rot.from_quat(Q[i]).as_matrix())

        #mesh_pose = np.eye(4)
        #mesh_pose[:3, 3] = np.array([0, 0, 1.1])
        #mesh_pose[:3, :3] = self.rotations[-1]
        #self.mesh_poses.append(mesh_pose)
    return np.array(rotations)


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
    min_dist, max_dist = np.min(distances), np.max(distances)
    alpha_values = (distances - min_dist) / (max_dist - min_dist)

    # Enhance fading effect
    #alpha_values = alpha_values ** 2
    #alpha_values = np.sqrt(alpha_values)

    # Generate colors with dynamic opacity
    adjusted_colors = [mcolors.to_rgba(c, alpha=a) for c, a in zip(base_colors, alpha_values)]
    return adjusted_colors


if __name__ == '__main__':
    parser = ArgumentParser(description='Annotate the coarse pose of the object in the image')
    parser.add_argument('--data', type=str, help='Path to the folder containing the data', required=True)

    parser.add_argument('--mesh', '-m', type=str, help='Path to the folder containing the data', default=None)
    parser.add_argument('--focal-length', '-f', type=float, help='Focal length of the camera', default=None)
    parser.add_argument('--window-size', '-w', type=int, nargs=2, help='Size of the window', default=(500, 500))
    parser.add_argument('--scale', '-s', type=float, help='Scale of the mesh', default=0.2)
    parser.add_argument('--speed', type=float, help='Speed of the rotation with mouse', default=0.6)
    parser.add_argument('--hide-axes', action='store_true', help='Hide the axes')
    parser.add_argument('--inverse', '-i', action='store_true', help='Inverse the rotation')
    
    args = parser.parse_args()

    if args.mesh is None:
        mesh_path = Path(__file__).parent / 'cube.ply'
        args.mesh = str(mesh_path)

    with open(args.data, 'r') as f:
        data = json.load(f)
    rotations = np.array([np.array(x['T_WorldFromCamera'])[:3,:3] for x in data])
    if args.inverse:
        rotations = rotations.transpose(0, 2, 1)
    scores = np.array([x['score'] for x in data])
    scores -= scores.min()
    scores /= scores.max()
    colors = matplotlib.colormaps['viridis'](scores)


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
    
    
    def compute_points():
        # Multiply the rotation matrices by the camera z-axis vector
        cam_axis = viewer.pose[2,:3]/np.linalg.norm(viewer.pose[2,:3])
        vectors = rotations @ cam_axis
        return vectors

    def plot(event=None, colors=colors):
        pts = compute_points()
        colors = fade_colors_with_distance(pts, colors, ax)

        # Redraw scatter plot with updated colors
        ax.clear()
        ax.scatter(*pts.T, c=colors, s=50, edgecolors=(0,0,0,0.2))
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