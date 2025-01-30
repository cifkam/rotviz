import trimesh
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import cv2

from .meshview import MeshViewer



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


if __name__ == '__main__':
    parser = ArgumentParser(description='Annotate the coarse pose of the object in the image')
    parser.add_argument('--mesh', '-m', type=str, help='Path to the folder containing the data', required=True)
    parser.add_argument('--focal-length', '-f', type=float, help='Focal length of the camera', default=None)
    parser.add_argument('--window-size', '-w', type=int, nargs=2, help='Size of the window', default=(900, 900))
    parser.add_argument('--scale', '-s', type=float, help='Scale of the mesh', default=0.3)
    parser.add_argument('--speed', type=float, help='Speed of the rotation with mouse', default=0.6)
    parser.add_argument('--hide-axes', action='store_true', help='Hide the axes')
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.mesh)
    mesh.apply_scale(args.scale)

    viewer = MeshViewer(mesh,
                        window_name='Mesh',
                        window_size=args.window_size,
                        focal_length=args.focal_length,
                        mouse_speed=args.speed, 
                        axes_scale=None if args.hide_axes else args.scale)
    
    pose = viewer.pose
    fig = plt.figure()
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
    
    def plot():
        #cam_axis = pose[2,:3]/np.linalg.norm(pose[2,:3])
        #ax.scatter(*cam_axis, c='red', alpha=1.0)
        ax.scatter(*pose[:3,:3], c=['red', 'green', 'blue'], alpha=1.0)
        setup_ax()

    plot()
    plt.show(block=False)

    for pose in viewer.run_yield(always_yield=True):        
        if pose is not None:
            ax.clear()
            plot()
            fig.canvas.draw()
        
        if not plt.get_fignums():
            break
        fig.canvas.flush_events()


    cv2.destroyAllWindows()
    plt.close()