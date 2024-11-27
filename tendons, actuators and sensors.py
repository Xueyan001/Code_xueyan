import mujoco
import mujoco.viewer
import numpy as np
from IPython.display import clear_output
import mediapy as media
import matplotlib.pyplot as plt
import cv2
import imageio
import time

MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid"/>
    <site name="anchor" pos="0 0 .3" size=".01"/>
    <camera name="fixed" pos="0 -1.3 .5" xyaxes="1 0 0 0 1 2"/>

    <geom name="pole" type="cylinder" fromto=".3 0 -.5 .3 0 -.1" size=".04"/>
    <body name="bat" pos=".3 0 -.1">
      <joint name="swing" type="hinge" damping="1" axis="0 0 1"/>
      <geom name="bat" type="capsule" fromto="0 0 .04 0 -.3 .04"
       size=".04" rgba="0 0 1 1"/>
    </body>

    <body name="box_and_sphere" pos="0 0 0">
      <joint name="free" type="free"/>
      <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
      <geom name="green_sphere"  size=".06" pos=".1 .1 .1" rgba="0 1 0 1"/>
      <site name="hook" pos="-.1 -.1 -.1" size=".01"/>
      <site name="IMU"/>
    </body>
  </worldbody>

  <tendon>
    <spatial name="wire" limited="true" range="0 0.35" width="0.003">
      <site site="anchor"/>
      <site site="hook"/>
    </spatial>
  </tendon>

  <actuator>
    <motor name="my_motor" joint="swing" gear="1"/>
  </actuator>

  <sensor>
    <accelerometer name="accelerometer" site="IMU"/>
  </sensor>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)

# ### original
# height = 480
# width = 480

# with mujoco.Renderer(model, height, width) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data, "fixed")
#   img =renderer.render()
#   plt.imshow(img)
#   plt.show()
# ### original

### start 1
### actuated bat and passive "piñata":
n_frames = 180
height = 240
width = 320
frames = []
fps = 60.0
times = []
sensordata = []

# constant actuator signal
mujoco.mj_resetData(model, data)
data.ctrl = 20

# Simulate and display video.
with mujoco.Renderer(model, height, width) as renderer:
  for i in range(n_frames):
    while data.time < i/fps:
      mujoco.mj_step(model, data)
      times.append(data.time)
      sensordata.append(data.sensor('accelerometer').data.copy())
    renderer.update_scene(data, "fixed")
    frame = renderer.render()
    frames.append(frame)

# media.show_video(frames, fps=fps)
video_filename = "bat_and_passive_pinata_video.mp4"
imageio.mimwrite(video_filename, frames, fps=n_frames)
cap = cv2.VideoCapture(video_filename) # read the video
if not cap.isOpened(): # check the video file whether be opend right or not
    print("Error")
    exit()
while True:                 # ret is the boolean value
    ret, frame = cap.read() # .read: Read video files frame by frame
    if not ret:             # ret = False, break
        break
    cv2.imshow('MuJoCo Simulation', frame)
    if cv2.waitKey(int(1000 / n_frames)) & 0xFF == ord('q'): # each frame keeps the 1000 / framerate ms
      break

### end 1
### actuated bat and passive "piñata":