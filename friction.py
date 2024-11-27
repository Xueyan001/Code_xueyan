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
    <material name="grid" texture="grid" texrepeat="6 6"
     texuniform="true" reflectance=".2"/>
     <material name="wall" rgba='.5 .5 .5 1'/>
  </asset>

  <default>
    <geom type="box" size=".05 .05 .05" />
    <joint type="free"/>
  </default>

  <worldbody>
    <light name="light" pos="-.2 0 1"/>
    <geom name="ground" type="plane" size=".5 .5 10" material="grid"
     zaxis="-.3 0 1" friction=".1"/>
    <camera name="y" pos="-.1 -.6 .3" xyaxes="1 0 0 0 1 2"/>
    <body pos="0 0 .1">
      <joint/>
      <geom/>
    </body>
    <body pos="0 .2 .1">
      <joint/>
      <geom friction=".33"/>
    </body>
  </worldbody>

</mujoco>
"""
n_frames = 60
height = 300
width = 300
frames = []

# load
model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)

# Simulate and display video.
with mujoco.Renderer(model, height, width) as renderer:
  mujoco.mj_resetData(model, data)
  for i in range(n_frames):
    while data.time < i/30.0:
      mujoco.mj_step(model, data)
    renderer.update_scene(data, "y")
    frame = renderer.render()
    frames.append(frame)

# media.show_video(frames, fps=30)
# Simulate and display video.
video_filename = "friction_video.mp4"
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