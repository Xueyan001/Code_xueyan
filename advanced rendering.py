import mujoco
import mujoco.viewer
import numpy as np
from IPython.display import clear_output
import mediapy as media
import matplotlib.pyplot as plt
import cv2
import imageio
import time
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

# ### original
# with mujoco.Renderer(model) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data)
#   plt.imshow(renderer.render())
#   plt.show()
# ### original

# ### Enable transparency and frame visualization
# scene_option = mujoco.MjvOption()
# scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# with mujoco.Renderer(model) as renderer:
#   renderer.update_scene(data, scene_option=scene_option)
#   frame = renderer.render()
#   plt.imshow(frame)
#   plt.axis('off')
#   plt.show()
# ### Enable transparency and frame visualization

# ### Depth rendering
# with mujoco.Renderer(model) as renderer:
#   # update renderer to render depth
#   renderer.enable_depth_rendering()

#   # reset the scene
#   renderer.update_scene(data)

#   # depth is a float array, in meters.
#   depth = renderer.render()

#   # Shift nearest values to the origin.
#   depth -= depth.min()
#   # Scale by 2 mean distances of near rays.
#   depth /= 2*depth[depth <= 1].mean()
#   # Scale to [0, 255]
#   pixels = 255*np.clip(depth, 0, 1)

#   pixels_uint8 = (pixels.astype(np.uint8))
#   plt.imshow(pixels_uint8, cmap='jet')
#   plt.axis('off')
#   plt.show()
# ### Depth rendering

### Segmentation rendering
with mujoco.Renderer(model) as renderer:
  renderer.disable_depth_rendering()

  # update renderer to render segmentation
  renderer.enable_segmentation_rendering()

  # reset the scene
  renderer.update_scene(data)

  seg = renderer.render()

  # Display the contents of the first channel, which contains object
  # IDs. The second channel, seg[:, :, 1], contains object types.
  geom_ids = seg[:, :, 0]
  # Infinity is mapped to -1
  geom_ids = geom_ids.astype(np.float64) + 1
  # Scale to [0, 1]
  geom_ids = geom_ids / geom_ids.max()
  pixels = 255*geom_ids
  plt.imshow(pixels.astype(np.uint8))
  plt.show()
### Segmentation rendering

