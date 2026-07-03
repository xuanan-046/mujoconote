import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 3"/>

    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0 0 1 1"/>
      <geom type="sphere" pos="0 0 -0.5" size="0.08" rgba="1 0 0 1" mass="1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# ---- PD控制器参数 ----
target_angle = 3.14   # 目标角度，弧度制，约90度（水平位置）
Kp = 2              # 比例增益：偏差越大，纠正力越大
Kd = 2               # 微分增益：转动越快，阻尼力越大（防止震荡超调）

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 当前角度和角速度
        current_angle = data.qpos[0]
        current_velocity = data.qvel[0]

        # PD控制公式：力矩 = Kp * 位置误差 - Kd * 当前速度
        error = target_angle - current_angle
        torque = Kp * error - Kd * current_velocity

        data.ctrl[0] = torque

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)