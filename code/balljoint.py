import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 3"/>

    <!-- 悬挂点，固定在空间中 -->
    <body name="arm" pos="0 0 1">
      <!-- ball关节：可以绕这个点任意方向转动 -->
      <joint name="shoulder" type="ball" pos="0 0 0" damping = "0.1"/>

      <!-- 一根杆子，垂下来 -->
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0 0 1 1" />
      <geom type="sphere" pos="0 0 -0.5" size="0.08" rgba="1 0 0 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 给一个初始的旋转扰动，让它偏离竖直方向
# qpos对于ball关节是四元数 [w, x, y, z]
# 这里让它绕x轴偏转一点角度
import numpy as np
angle = 0.5  # 弧度
data.qpos[0] = np.cos(angle / 2)   # w
data.qpos[1] = np.sin(angle / 2)   # x
data.qpos[2] = 0                    # y
data.qpos[3] = 0                    # z

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)