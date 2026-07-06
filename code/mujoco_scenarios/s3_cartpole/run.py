"""
场景3: Cart-Pole 倒立摆
运行: python run.py --controller pid     # PID控制
运行: python run.py --controller lqr     # LQR控制
运行: python run.py --controller pid --view   # 带可视化窗口
"""
import argparse
import numpy as np
import mujoco

MODEL_PATH = "model.xml"


class PIDController:
    """用杆子的倾斜角度作为误差信号，输出施加在小车上的力"""
    def __init__(self, kp=50.0, ki=0.5, kd=8.0, dt=0.005):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, pole_angle, pole_angvel):
        error = pole_angle  # 目标角度是0(竖直)
        self.integral += error * self.dt
        derivative = pole_angvel  # 直接用角速度作为微分项，噪声更小
        force = self.kp * error + self.ki * self.integral + self.kd * derivative
        return force


def lqr_gain():
    """
    对Cart-Pole在竖直附近线性化后手动求解的LQR增益(离线用scipy计算好的近似值)。
    状态向量 x = [cart_pos, cart_vel, pole_angle, pole_angvel]
    这里直接给出一组经验调好的增益，实际工程中应该用 scipy.linalg.solve_continuous_are 求解
    """
    # K = [K_pos, K_vel, K_angle, K_angvel]
    return np.array([-3.16, -4.24, -55.0, -9.5])


def run(controller_type="pid", view=False, duration=10.0):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 给一个小扰动作为初始条件(约5.7度)，测试控制器纠正能力
    data.qpos[1] = np.deg2rad(5)
    mujoco.mj_forward(model, data)

    pid = PIDController(dt=model.opt.timestep)
    K = lqr_gain()

    def step_once():
        cart_pos = data.qpos[0]
        cart_vel = data.qvel[0]
        pole_angle = data.qpos[1]
        pole_angvel = data.qvel[1]

        if controller_type == "pid":
            force = pid.compute(pole_angle, pole_angvel)
        else:  # lqr
            state = np.array([cart_pos, cart_vel, pole_angle, pole_angvel])
            force = -K @ state  # u = -Kx

        data.ctrl[0] = np.clip(force, -20, 20)
        mujoco.mj_step(model, data)

    if view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration:
                step_once()
                viewer.sync()
        return

    times, angles, positions = [], [], []
    while data.time < duration:
        step_once()
        times.append(data.time)
        angles.append(np.rad2deg(data.qpos[1]))
        positions.append(data.qpos[0])

    max_angle = max(abs(a) for a in angles[len(angles)//2:])
    print(f"[{controller_type.upper()}] 后半程最大倾斜角度: {max_angle:.3f} 度 "
          f"(越接近0说明控制越稳)")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(times, angles)
        axes[0].set_ylabel("杆子角度(度)")
        axes[0].axhline(0, color="gray", linestyle="--")
        axes[0].grid(True)
        axes[1].plot(times, positions)
        axes[1].set_ylabel("小车位置(m)")
        axes[1].set_xlabel("时间(s)")
        axes[1].grid(True)
        plt.suptitle(f"Cart-Pole {controller_type.upper()} 控制效果")
        plt.tight_layout()
        plt.savefig(f"cartpole_{controller_type}.png", dpi=120)
        print(f"已保存 cartpole_{controller_type}.png")
    except ImportError:
        print("未安装matplotlib，跳过画图")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["pid", "lqr"], default="pid")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    run(controller_type=args.controller, view=args.view)
