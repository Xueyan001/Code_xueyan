import mujoco
import mujoco.viewer
import numpy as np
from IPython.display import clear_output
import mediapy as media
import matplotlib.pyplot as plt
import cv2
import imageio
import time
import itertools


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ### start 1
# ### camera matrix
# def compute_camera_matrix(renderer, data):
#   """Returns the 3x4 camera matrix."""
#   # If the camera is a 'free' camera, we get its position and orientation
#   # from the scene data structure. It is a stereo camera, so we average over
#   # the left and right channels. Note: we call `self.update()` in order to
#   # ensure that the contents of `scene.camera` are correct.
#   renderer.update_scene(data)
#   pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
#   z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
#   y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
#   rot = np.vstack((np.cross(y, z), y, z))
#   fov = model.vis.global_.fovy

#   # Translation matrix (4x4).
#   translation = np.eye(4)
#   translation[0:3, 3] = -pos

#   # Rotation matrix (4x4).
#   rotation = np.eye(4)
#   rotation[0:3, 0:3] = rot

#   # Focal transformation matrix (3x4).
#   focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
#   focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

#   # Image matrix (3x3).
#   image = np.eye(3)
#   image[0, 2] = (renderer.width - 1) / 2.0
#   image[1, 2] = (renderer.height - 1) / 2.0
#   return image @ focal @ rotation @ translation

# #@title Project from world to camera coordinates {vertical-output: true}

# with mujoco.Renderer(model) as renderer:
#   renderer.disable_segmentation_rendering()
#   # reset the scene
#   renderer.update_scene(data)

#   # Get the world coordinates of the box corners
#   box_pos = data.geom_xpos[model.geom('red_box').id]
#   box_mat = data.geom_xmat[model.geom('red_box').id].reshape(3, 3)
#   box_size = model.geom_size[model.geom('red_box').id]
#   offsets = np.array([-1, 1]) * box_size[:, None]
#   xyz_local = np.stack(list(itertools.product(*offsets))).T
#   xyz_global = box_pos[:, None] + box_mat @ xyz_local

#   # Camera matrices multiply homogenous [x, y, z, 1] vectors.
#   corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
#   corners_homogeneous[:3, :] = xyz_global

#   # Get the camera matrix.
#   m = compute_camera_matrix(renderer, data)

#   # Project world coordinates into pixel space. See:
#   # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
#   xs, ys, s = m @ corners_homogeneous
#   # x and y are in the pixel coordinate system.
#   x = xs / s
#   y = ys / s

#   # Render the camera view and overlay the projected corner coordinates.
#   pixels = renderer.render()
#   fig, ax = plt.subplots(1, 1)
#   ax.imshow(pixels)
#   ax.plot(x, y, '+', c='w')
#   ax.set_axis_off()
#   plt.show()
### end 1
### camera matrix

### start 2
### modifying the scene
def get_geom_speed(model, data, geom_name):
  """Returns the speed of a geom."""
  geom_vel = np.zeros(6)
  geom_type = mujoco.mjtObj.mjOBJ_GEOM
  geom_id = data.geom(geom_name).id
  mujoco.mj_objectVelocity(model, data, geom_type, geom_id, geom_vel, 0)
  return np.linalg.norm(geom_vel)

def add_visual_capsule(scene, point1, point2, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    return
  scene.ngeom += 1  # increment ngeom
  # initialise a new capsule, add it to the scene using mjv_connector
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                       mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                       point1, point2)

 # traces of time, position and speed
times = []
positions = []
speeds = []
offset = model.jnt_axis[0]/16  # offset along the joint axis

def modify_scene(scn):
  """Draw position trace, speed modifies width and colors."""
  if len(positions) > 1:
    for i in range(len(positions)-1):
      rgba=np.array((np.clip(speeds[i]/10, 0, 1),
                     np.clip(1-speeds[i]/10, 0, 1),
                     .5, 1.))
      radius=.003*(1+speeds[i])
      point1 = positions[i] + offset*times[i]
      point2 = positions[i+1] + offset*times[i+1]
      add_visual_capsule(scn, point1, point2, radius, rgba)

duration = 6    # (seconds)
framerate = 30  # (Hz)

# Simulate and display video.
frames = []

# Reset state and time.
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

with mujoco.Renderer(model) as renderer:
  while data.time < duration:
    # append data to the traces
    positions.append(data.geom_xpos[data.geom("green_sphere").id].copy())
    times.append(data.time)
    speeds.append(get_geom_speed(model, data, "green_sphere"))
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data)
      modify_scene(renderer.scene)
      pixels = renderer.render()
      frames.append(pixels)

video_filename = "camera_matrix_modifying_video.mp4"
imageio.mimwrite(video_filename, frames, fps=framerate)
cap = cv2.VideoCapture(video_filename) # read the video
if not cap.isOpened(): # check the video file whether be opend right or not
    print("Error")
    exit()
while True:                 # ret is the boolean value
    ret, frame = cap.read() # .read: Read video files frame by frame
    if not ret:             # ret = False, break
        break
    cv2.imshow('MuJoCo Simulation', frame)
    if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'): # each frame keeps the 1000 / framerate ms
      break
# media.show_video(frames, fps=framerate)