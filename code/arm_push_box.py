import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 机械臂：固定在空中的两节臂，和之前一样 -->
    <body name="upper_arm" pos="0 -0.5 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" rgba="0 0 1 1"/>

      <body name="forearm" pos="0 0 -0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.025" rgba="0 1 0 1"/>
        <!-- 末端加一个小球当"手指"，用来接触箱子 -->
        <geom name="end_effector" type="sphere" pos="0 0 -0.4" size="0.06" rgba="1 0 0 1" mass="0.5"/>
      </body>
    </body>

    <!-- 一个可以被推动的箱子，自由物体 -->
    <body name="box" pos="0 -0.35 0.15">
      <joint type="free"/>
      <geom name="box_geom" type="box" size="0.15 0.15 0.15" rgba="1 0.5 0 1" mass="1" friction="0.5 0.005 0.0001"/>
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

# PD控制：让手臂主动摆向箱子方向
target_shoulder = 0.8
target_elbow = -0.5
Kp = 30.0
Kd = 3.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        error_shoulder = target_shoulder - data.qpos[0]
        torque_shoulder = Kp * error_shoulder - Kd * data.qvel[0]

        error_elbow = target_elbow - data.qpos[1]
        torque_elbow = Kp * error_elbow - Kd * data.qvel[1]

        data.ctrl[0] = torque_shoulder
        data.ctrl[1] = torque_elbow   

        mujoco.mj_step(model, data)

        box_pos = data.geom("box_geom").xpos
        end_pos = data.geom("end_effector").xpos
        distance = ((box_pos - end_pos)**2).sum()**0.5

        print(f"箱子位置: {box_pos}, 末端球位置: {end_pos}, 距离: {distance:.3f}")

        # 检测并打印接触信息
        if data.ncon > 0:
            print(f"检测到 {data.ncon} 个接触点")

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step*5)