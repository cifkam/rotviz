import cv2
import os
import numpy as np
import pyrender
import trimesh
import pinocchio as pin
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from scipy.spatial.transform import Rotation, Slerp
from pyrender.constants import RenderFlags


class Renderer:
    def __init__(self, width, height, mesh, focal_length=None, ambient_light=8.0):
        self.width = width
        self.height = height
        self.renderer = pyrender.OffscreenRenderer(width, height)
        if focal_length is None:
            self.focal_length = np.sqrt(width**2 + height**2)
        else:
            self.focal_length = focal_length
        self.scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
            ambient_light=np.array([ambient_light] * 4)
        )
        self.camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=width/2, cy=height/2)
        self.renderer = pyrender.OffscreenRenderer(width, height)

        if isinstance(mesh, str) or isinstance(mesh, Path):
            self.mesh_trimesh = trimesh.load_mesh(mesh)
        elif isinstance(mesh, trimesh.Trimesh):
            self.mesh_trimesh = mesh
        else:
            raise ValueError("Invalid mesh type")

        self.opencv2opengl = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]])

        self.mesh = pyrender.Mesh.from_trimesh(self.mesh_trimesh)
        self.mesh_node = self.scene.add(self.mesh, name='mesh', pose=np.eye(4))
        self.camera_node = self.scene.add(self.camera, pose=self.opencv2opengl, name='camera')

    def set_focal_length(self, focal_length):
        self.focal_length = focal_length
        self.camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=self.width/2, cy=self.height/2)
        #self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self.scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
            ambient_light=np.array([8.0] * 4)
        )
        self.mesh_node = self.scene.add(self.mesh, name='mesh', pose=np.eye(4))
        self.camera_node = self.scene.add(self.camera, pose=self.opencv2opengl, name='camera')

    
    def render(self, pose):
        self.scene.set_pose(self.mesh_node, pose)
        self.scene.set_pose(self.camera_node, self.opencv2opengl)
        color, depth = self.renderer.render(self.scene, flags=RenderFlags.SKIP_CULL_FACES)
        return color[..., ::-1], depth
    
    def project(self, point):
        return (int(point[0] / point[2] * self.focal_length + self.width / 2),
                int(point[1] / point[2] * self.focal_length + self.height / 2))


def oriented_angle(a,b):
    return np.arctan2(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1])


class MeshViewer:
    def __init__(self, mesh, window_name='Mesh', window_size=(768,768), focal_length=None, axes_scale=None, mouse_speed=1.0, rotate_mouse_button='left', ambient_light=8.0, verbose=True):
        assert rotate_mouse_button in ['left', 'middle', None]
        self.rotate_mouse_button = rotate_mouse_button

        self.verbose = verbose

        self.mesh = mesh
        self.window_name = window_name
        self.window_size = window_size
        self.renderer = Renderer(mesh=mesh, width=window_size[1], height=window_size[0], focal_length=focal_length, ambient_light=ambient_light)
        self.axes_scale = axes_scale
        self.mouse_speed = mouse_speed

        self.depth = 1.5
        self.pose = np.eye(4)
        self.pose[2,3] = self.depth
        #self.pose[:3,:3] = np.round(pin.exp(np.array([-np.pi/2,0,0])))
        self.cam_axis = self.pose[2,:3]/np.linalg.norm(self.pose[2,:3])
        
        self.depth_multiplier = [1, 5, 10]
        self.rotation_multiplier = np.deg2rad([1, 2.5, 5, 10])
        self.last_depth_multipier_idx = 0
        self.last_rotation_multipier_idx = 0

        self.yaw, self.pitch, self.roll = 0, 0, 0
        self.lbtn_down = False
        self.mbtn_down = False
        self.mouse_prev_pos = np.array([0, 0])

    
    def _update_pose(self):
        new_pose = np.eye(4)
        #pose[0, 3] = (x - self.renderer.width / 2) * depth / self.renderer.focal_length
        #pose[1, 3] = (y - self.renderer.height / 2) * depth / self.renderer.focal_length
        new_pose[2, 3] = self.depth
        new_pose[:3, :3] = pin.exp(np.array([self.roll, self.pitch, self.yaw])) @ self.pose[:3, :3]
        self.pose = new_pose
        self.cam_axis = self.pose[2,:3]/np.linalg.norm(self.pose[2,:3])
    
    def _draw(self):
        rendered_color, depth = self.renderer.render(self.pose)
        img = rendered_color.copy()
        
        if self.axes_scale is not None:
            # Define the axes
            s = self.axes_scale
            x_axis = np.array([s, 0, 0, 1])
            y_axis = np.array([0, s, 0, 1])
            z_axis = np.array([0, 0, s, 1])
            origin = np.array([0, 0, 0, 1])

            # Transform the axes using the current pose
            x_axis_transformed = self.pose @ x_axis
            y_axis_transformed = self.pose @ y_axis
            z_axis_transformed = self.pose @ z_axis
            origin_transformed = self.pose @ origin
            
            # Project the 3D points to 2D
            origin_2d = self.renderer.project(origin_transformed)
            x_axis_2d = self.renderer.project(x_axis_transformed)
            y_axis_2d = self.renderer.project(y_axis_transformed)
            z_axis_2d = self.renderer.project(z_axis_transformed)

            # Draw the axes on the rendered image
            cv2.line(img, origin_2d, x_axis_2d, (0, 0, 255), 2)
            cv2.line(img, origin_2d, y_axis_2d, (0, 255, 0), 2)
            cv2.line(img, origin_2d, z_axis_2d, (255, 0, 0), 2)

        
        # Show the camera axis vector
        cam_axis_text = ', '.join(['{0:.3f}'.format(self.cam_axis[i]) for i in range(3)])
        cv2.putText(img, f"[{cam_axis_text}]", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the image
        cv2.imshow(self.window_name, img)


    def run(self, delay=100):
        for _ in self.run_yield(delay=delay):
            pass

    def run_yield(self, delay=100, always_yield=False):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[1], self.window_size[0])
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

        self._draw()
        while True:
            self.yaw, self.pitch, self.roll = 0, 0, 0

            last_pose = self.pose.copy()
            
            key = cv2.waitKey(delay) & 0xFF
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('u'):
                self.last_depth_multipier_idx = (self.last_depth_multipier_idx + 1) % len(self.depth_multiplier)
                if self.verbose:
                    print('Depth multiplier:', self.depth_multiplier[self.last_depth_multipier_idx])
            elif key == ord('y'):
                self.last_rotation_multipier_idx = (self.last_rotation_multipier_idx + 1) % len(self.rotation_multiplier)
                if self.verbose:
                    print("Rotation multiplier:", np.round(np.rad2deg(self.rotation_multiplier[self.last_rotation_multipier_idx]), 1))
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                break
            else:
                if key == ord('s'):
                    self.roll = 1*self.rotation_multiplier[self.last_rotation_multipier_idx]
                elif key == ord('w'):
                    self.roll = -1*self.rotation_multiplier[self.last_rotation_multipier_idx]
                elif key == ord('a'):
                    self.pitch = 1*self.rotation_multiplier[self.last_rotation_multipier_idx]
                elif key == ord('d'):
                    self.pitch = -1*self.rotation_multiplier[self.last_rotation_multipier_idx]
                elif key == ord('e'):
                    self.yaw = 1*self.rotation_multiplier[self.last_rotation_multipier_idx]
                elif key == ord('q'):
                    self.yaw = -1*self.rotation_multiplier[self.last_rotation_multipier_idx]
                if key == ord('z'):
                    self.depth *= 1 + 0.01*self.depth_multiplier[self.last_depth_multipier_idx]
                elif key == ord('x'):
                    self.depth /= 1 + 0.01*self.depth_multiplier[self.last_depth_multipier_idx]
                elif key == ord('f'): # reset the pose
                    #self.pose = np.eye(4)
                    #self.pose[2,3] = self.depth
                    self.pose[:3,:3] = np.eye(3) #np.round(pin.exp(np.array([-np.pi/2,0,0])))

                if self.verbose and key in [ord(x) for x in ['s', 'w', 'a', 'd', 'e', 'q', 'f']]:
                    print('Camera forward axis:', self.cam_axis)
                self._update_pose()
                self._draw()
            
            if not np.allclose(last_pose, self.pose):
                yield self.pose
            elif always_yield:
                yield None
        return
            


    def _on_mouse_click(self, event, x, y, flags, param):
        # do not use global variables, but class attributes
        #global depth, m_prev_pos, m_btn_down, l_btn_down, mesh_pose
        self.yaw, self.pitch, self.roll = 0, 0, 0

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_prev_pos = np.array([x, y])
            self.lbtn_down = True

        if event == cv2.EVENT_LBUTTONUP:
            self.lbtn_down = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.mouse_prev_pos = np.array([x, y])
            self.mbtn_down = True

        elif event == cv2.EVENT_MBUTTONUP:
            self.mbtn_down = False
        
        elif event == cv2.EVENT_MOUSEMOVE:

            if (self.rotate_mouse_button == 'left' and self.lbtn_down) or (self.rotate_mouse_button == 'middle' and self.mbtn_down):
                m_delta = self.mouse_prev_pos - np.array([x, y])

                if np.allclose(m_delta, 0):
                    return

                # rotation around camera x,y
                xy_axis = np.array([-m_delta[1], m_delta[0], 0])
                xy_angle = np.linalg.norm(m_delta)/100.0
                xy_axis = xy_axis/np.linalg.norm(xy_axis)

                # rotation around camera z
                origin = self.renderer.project(self.pose @ np.array([0, 0, 0, 1]))
                a = self.mouse_prev_pos - origin
                b = np.array([x, y]) - origin

                z_angle = oriented_angle(a, b)
                z_axis = np.array([0, 0, 1])

                self.roll, self.pitch, _ = xy_axis*xy_angle
                _, _, self.yaw = z_axis*z_angle

                # weighting based on the angle between the mouse movement and origin-to-mouse vector
                mouse_dist = np.linalg.norm(b)
                if mouse_dist < 15: # completely ignore yaw when the mouse is close to the origin
                    w = 1
                else:
                    dot = np.dot(b, m_delta) / (np.linalg.norm(b) * np.linalg.norm(m_delta)) # (-1, 1)
                    w = np.abs(dot)
                    w = w**1.5 # make it a bit easier to rotate just around the "camera forward axis"
                     # smooth transition from w=1 to w=0 as the mouse moves away from the origin
                    alpha = 1/(1+np.exp(  (-mouse_dist+50)/20  )) # shifted and scaled sigmoid
                    w = w*alpha + 1*(1-alpha)

                self.roll  *= self.mouse_speed * w
                self.pitch *= self.mouse_speed * w
                self.yaw   *= self.mouse_speed * (1-w)

                self.mouse_prev_pos = np.array([x, y])

                self._update_pose()
                self._draw()


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
    
    # Run the viewer and block until the window is closed
    viewer.run(delay=1000)

    # Run the viewer and print the pose on each iteration
    #for pose in viewer.run_yield(delay=1000):
    #    print(pose)
    #    print()