import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 3"/>

    <!-- 摆锤的悬挂点,从(0,0,1)这个高度开始 -->
    <body name="pendulum" pos="0 0 1">
      <!-- hinge关节：绕y轴转动，pos="0 0 0"表示旋转轴在这个body的局部原点 -->
      <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>

      <!-- 摆杆：一根细长的胶囊体，从(0,0,0)连到(0,0,-0.5) -->
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0 0 1 1"/>

      <!-- 摆锤末端的球 -->
      <geom type="sphere" pos="0 0 -0.5" size="0.08" rgba="1 0 0 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 给一个初始角度偏移，让它偏离竖直方向，这样才会开始摆动
data.qpos[0] = 1.0  # 弧度制，约57度

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)