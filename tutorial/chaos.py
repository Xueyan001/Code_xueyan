import mujoco
import mujoco.viewer
import numpy as np
from IPython.display import clear_output
import mediapy as media
import matplotlib.pyplot as plt
import cv2
import imageio
import time

chaotic_pendulum = """
<mujoco>
  <option timestep=".001">
    <flag energy="enable" contact="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 -1 0"/>
    <geom type="capsule" size=".02"/>
  </default>

  <worldbody>
    <light pos="0 -.4 1"/>
    <camera name="fixed" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <body name="0" pos="0 0 .2">
      <joint name="root"/>
      <geom fromto="-.2 0 0 .2 0 0" rgba="1 1 0 1"/>
      <geom fromto="0 0 0 0 0 -.25" rgba="1 1 0 1"/>
      <body name="1" pos="-.2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="1 0 0 1"/>
      </body>
      <body name="2" pos=".2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 1 0 1"/>
      </body>
      <body name="3" pos="0 0 -.25">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(chaotic_pendulum)
data = mujoco.MjData(model)

PERTURBATION = 1e-7
SIM_DURATION = 10 # seconds
NUM_REPEATS = 8   # each subplot will show 8 curves

# # preallocate
# n_steps = int(SIM_DURATION / model.opt.timestep)
# sim_time = np.zeros(n_steps)
# angle = np.zeros(n_steps)
# energy = np.zeros(n_steps)

# # prepare plotting axes
# _, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True) # 2 raws, 1 column, share x-axis

# # simulate NUM_REPEATS times with slightly different initial conditions
# for _ in range(NUM_REPEATS):
#   # initialize
#   mujoco.mj_resetData(model, data)
#   data.qvel[0] = 10 # root joint velocity
#   # perturb initial velocities
#   data.qvel[:] += PERTURBATION * np.random.randn(model.nv)

#   # simulate
#   for i in range(n_steps):
#     mujoco.mj_step(model, data)
#     sim_time[i] = data.time
#     angle[i] = data.joint('root').qpos[0]
#     energy[i] = data.energy[0] + data.energy[1]

#   # plot
#   ax[0].plot(sim_time, angle)
#   ax[1].plot(sim_time, energy)

# # finalize plot
# ax[0].set_title('root angle')
# ax[0].set_ylabel('radian')
# ax[1].set_title('total energy')
# ax[1].set_ylabel('Joule')
# ax[1].set_xlabel('second')
# plt.tight_layout()
# plt.show()

# ### start 1
# ### Timestep and accurancy
# SIM_DURATION = 10 # (seconds)
# TIMESTEPS = np.power(10, np.linspace(-2, -4, 5))
# # np.linspace(start, stop, num): produce num equally spaced values [-2.0, -2.5, -3.0, -3.5, -4.0]
# # np.power(base, exponent)

# # prepare plotting axes
# _, ax = plt.subplots(1, 1)

# for dt in TIMESTEPS:
#    # set timestep, print
#   model.opt.timestep = dt

#   # allocate
#   n_steps = int(SIM_DURATION / model.opt.timestep)
#   sim_time = np.zeros(n_steps)
#   energy = np.zeros(n_steps)

#   # initialize
#   mujoco.mj_resetData(model, data)
#   data.qvel[0] = 9 # root joint velocity

#   # simulate
#   print('{} steps at dt = {:2.2g}ms'.format(n_steps, 1000*dt))
#   for i in range(n_steps):
#     mujoco.mj_step(model, data)
#     sim_time[i] = data.time
#     energy[i] = data.energy[0] + data.energy[1]

#   # plot
#   ax.plot(sim_time, energy, label='timestep = {:2.2g}ms'.format(1000*dt))

# # finalize plot
# ax.set_title('energy')
# ax.set_ylabel('Joule')
# ax.set_xlabel('second')
# ax.legend(frameon=True);
# plt.tight_layout()
# plt.show()
# ### Timestep and accurancy
# ### end 1

### start 2
### timestep and divergence
SIM_DURATION = 10 # (seconds)
TIMESTEPS = np.power(10, np.linspace(-2, -1.5, 7))

# get plotting axes
ax = plt.gca()

for dt in TIMESTEPS:
  # set timestep
  model.opt.timestep = dt

  # allocate
  n_steps = int(SIM_DURATION / model.opt.timestep)
  sim_time = np.zeros(n_steps)
  energy = np.zeros(n_steps) * np.nan
  speed = np.zeros(n_steps) * np.nan

  # initialize
  mujoco.mj_resetData(model, data)
  data.qvel[0] = 11 # set root joint velocity

  # simulate
  print('simulating {} steps at dt = {:2.2g}ms'.format(n_steps, 1000*dt))
  for i in range(n_steps):
    mujoco.mj_step(model, data)
    if data.warning.number.any():
      warning_index = np.nonzero(data.warning.number)[0][0]
      warning = mujoco.mjtWarning(warning_index).name
      print(f'stopped due to divergence ({warning}) at timestep {i}.\n')
      break
    sim_time[i] = data.time
    energy[i] = sum(abs(data.qvel))
    speed[i] = np.linalg.norm(data.qvel)

  # plot
  ax.plot(sim_time, energy, label='timestep = {:2.2g}ms'.format(1000*dt))
  ax.set_yscale('log')

# finalize plot
ax.set_ybound(1, 1e3)
ax.set_title('energy')
ax.set_ylabel('Joule')
ax.set_xlabel('second')
ax.legend(frameon=True, loc='lower right');
plt.tight_layout()
plt.show()
### end 2
### timestep and divergence