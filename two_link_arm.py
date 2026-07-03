import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 3"/>

    <!-- 第一节：肩关节 + 上臂 -->
    <body name="upper_arm" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" rgba="0 0 1 1"/>

      <!-- 第二节：肘关节 + 前臂，嵌套在upper_arm里面 -->
      <body name="forearm" pos="0 0 -0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.025" rgba="0 1 0 1"/>
        <geom type="sphere" pos="0 0 -0.4" size="0.06" rgba="1 0 0 1" mass="0.5"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_shoulder" joint="shoulder" gear="1"/>
    <motor name="motor_elbow" joint="elbow" gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# PD控制器参数（两个关节各自独立控制）
target_shoulder = 1.0   # 肩关节目标角度
target_elbow = -1.0     # 肘关节目标角度
Kp = 20.0
Kd = 2.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 肩关节PD控制（qpos[0], qvel[0]对应shoulder）
        error_shoulder = target_shoulder - data.qpos[0]
        torque_shoulder = Kp * error_shoulder - Kd * data.qvel[0]

        # 肘关节PD控制（qpos[1], qvel[1]对应elbow）
        error_elbow = target_elbow - data.qpos[1]
        torque_elbow = Kp * error_elbow - Kd * data.qvel[1]

        data.ctrl[0] = torque_shoulder
        data.ctrl[1] = torque_elbow

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)