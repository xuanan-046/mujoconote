import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="piston" pos="0 0 0.5">
      <joint name="slide_joint" type="slide" axis="0 0 1" range = " 0 1"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1" mass="1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="slide_joint" gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        data.ctrl[0] = 15.0  # 给一个恒定的推力，让方块沿z轴向上滑动

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)