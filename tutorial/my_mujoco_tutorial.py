import mujoco
import mujoco.viewer
import numpy as np
from IPython.display import clear_output
import mediapy as media
import matplotlib.pyplot as plt
import cv2
import imageio
import time

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

clear_output()

# We begin by defining and loading a simple model:
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
        <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1" friction="0.8 0.8 0.8"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1" friction="0.8 0.8 0.8"/>
    </body>
  </worldbody>
</mujoco>
"""
# geom: define the geometry, the position of red_box is default(0,0,0), green_sphere use the default type'sphere'
# rgba: a=1 full transparency, a=0 full non-transparency
# <light> :make the image be lighter
#initial rotate around z-axis with 30-degree 
# The from_xml_string() method invokes the model compiler, which creates a binary mjModel instance.
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

n = model.ngeom # .ngeom: the quantities of geometries
#print(n)

resp_color = model.geom_rgba # .geom_rgba: the color array of geometries
#print(resp_color)

# #Named Access
# try:
#   model.geom('red_box')
# except KeyError as e:
#   print(e)
# named_access = model.geom('green_sphere').rgba
# #print(named_access)

# mj_name2id function
id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'red_box') # red_box_id=0, green_sphere_id=1
id_rgba = model.geom_rgba[id, :] # Access the RGBA value of the specified geometry
print('id of "green_sphere": ', model.geom('green_sphere').id)
print('name of geom 1: ', model.geom(1).name)
print('name of body 0: ', model.body(0).name) # defined in xml

geom_names = [model.geom(i).name for i in range(model.ngeom)]
print(geom_names)

# .mjData
data = mujoco.MjData(model)
print(data.geom_xpos[0]) # print the position of id[0] geom

# .mj_kinematics, without this function, the position will not be explicitly propagated
mujoco.mj_kinematics(model, data)
print('raw access:\n', data.geom_xpos) # output[0.2 0.2 0.2]
# MjData also supports named access:
print('\nnamed access:\n', data.geom('green_sphere').xpos)

# model.geom('red_box').rgba[:3] = np.random.rand(3)
# model.geom('green_sphere').rgba[:3] = np.random.rand(3)

# enable joint visualization option:
# True means visualization.
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True # visualizing joint
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True # visualizing force
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True # visualizing force point

# viewer = mujoco.viewer.launch_passive(model, data)
# joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'swing')
# data.qvel[joint_id] = 1.0

duration = 3.8 # (seconds)
framerate = 60  # (Hz)
time_step = 1 / framerate

# flip gravity and re-render, or xml:<option gravity="0 0 10"/>
print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, 10) # default gravity is -9.8m/s**2
print('flipped gravity', model.opt.gravity)

# Make renderer, render and show the pixels
# Simulation mj_step, simulate and display video
frames = []                       # Initialize an empty list frames to store the rendered image of each frame
mujoco.mj_resetData(model, data)  # Reset state and time.

# with mujoco.Renderer(model) as renderer:
#     mujoco.mj_forward(model, data)
#     renderer.update_scene(data)
#     pixels = renderer.render() 
#     image_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
#     cv2.imshow('Rendered Image', image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

mujoco.mj_forward(model, data)
with mujoco.Renderer(model) as renderer:
  while data.time < duration:   # duration=3.8s, framerate=60Hz
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, scene_option=scene_option)
      pixels = renderer.render()
      frames.append(pixels)
      print('Total number of DoFs in the model:', model.nv)
      print('Generalized positions:', data.qpos) # the rotation angle og hinge joint, unit: rad
      print('Generalized velocities:', data.qvel)
      # frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
      # frames.append(frame_bgr)
      # cv2.imshow('Rendered Image', frame_bgr)
      # cv2.waitKey(0) # 0 means the function will wait indefinitely for a key press event from the user, number is the waiting time
      # cv2.destroyAllWindows() # close the cv2 windows
video_filename = "mujoco_simulation_video.mp4"
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
    



# # use the mujoco.viewer
# while data.time < duration:
#   mujoco.mj_step(model, data)
#   angle = data.qpos[joint_id]
#   print(f"current time:{data.time:.2f}s, 'swing' joint rotation angle(rad): {angle:.2f}")
#   viewer.sync()
#   time.sleep(1 / framerate) # Simulate real time



# # 使用 MuJoCo 的交互式查看器来显示模型
# try:
#     print("Starting MuJoCo interactive viewer...")
#     mujoco.viewer.launch(model, data)  # 使用交互式查看器来查看模型
# except Exception as e:
#     print(f"An error occurred while launching the viewer: {e}")


# other tries
# import mediapy as media #it's for Jupyter Notebook
# # show image
# media.show_image(r'D:\Model\mujoco_menagerie-main\franka_fr3\fr3.png')


# # Pillow
# try:
#     img = Image.open(r'D:\Model\mujoco_menagerie-main\franka_fr3\fr3.png')
#     img.show()  
# except Exception as e:
#     print("Error loading image with Pillow:", e)

# Set up GPU rendering.
# 安装 MuJoCo
# 在终端中执行此命令以安装 MuJoCo（如果尚未安装）
# pip install mujoco

# import os
# import subprocess
# import numpy as np
# import mujoco
# import matplotlib.pyplot as plt
# import mediapy as media
# from IPython.display import clear_output

# # 设置 GPU 渲染（可选，取决于你是否有 GPU 及其相关配置）
# try:
#     # 检查是否有可用 GPU
#     if subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL).returncode != 0:
#         print("No GPU detected. Running in CPU mode.")
#     else:
#         print("GPU detected. Setting up GPU rendering.")
#         os.environ['MUJOCO_GL'] = 'egl'  # 配置 MuJoCo 使用 GPU 渲染
# except FileNotFoundError:
#     print("nvidia-smi not found. Assuming no GPU is available.")

# # 检查 MuJoCo 安装是否成功
# try:
#     print('Checking that the installation succeeded:')
#     mujoco.MjModel.from_xml_string('<mujoco/>')
#     print('Installation successful.')
# except Exception as e:
#     raise RuntimeError(
#         'Something went wrong during installation. Check the shell output above '
#         'for more information.'
#     ) from e

# # 设置 NumPy 的打印选项以使输出更易于阅读
# np.set_printoptions(precision=3, suppress=True, linewidth=100)

# # 清理终端输出（可选）
# clear_output()

# # 示例代码：创建一个简单的 MuJoCo 模型并渲染
# xml = """"
# <mujoco>
#   <worldbody>
#     <light name="main_light" pos="0 0 3" diffuse="1 1 1"/>
#     <geom name="red_box" type="box" size=".5 .5 .5" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".5 .5 .5" size=".3" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>


# """

# # 创建模型和数据
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)

# # 使用渲染器进行渲染并显示
# with mujoco.Renderer(model) as renderer:
#     img = renderer.render()  # 获取渲染的图像数据

#     # 使用 matplotlib 显示图像
#     plt.imshow(img)
#     plt.axis('off')  # 隐藏坐标轴
#     plt.show()

# 安装 MuJoCo
# 在终端中执行此命令以安装 MuJoCo（如果尚未安装）
# pip install mujoco

