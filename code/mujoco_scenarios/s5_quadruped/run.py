"""
场景5: 简化四足机器人站立平衡
运行: python run.py            # 无窗口，输出躯干姿态稳定性数据
运行: python run.py --view     # 可视化窗口，中途会施加一次外部扰动
"""
import sys
import numpy as np
import mujoco

MODEL_PATH = "model.xml"

# 站立姿态下每条腿的目标关节角度(髋, 膝)，通过网格搜索找到的稳定站姿
STAND_HIP = 0.2
STAND_KNEE = -0.4


def set_stand_pose(data):
    data.ctrl[:] = [STAND_HIP, STAND_KNEE] * 4  # 4条腿，每条(hip, knee)


def run(view=False, duration=6.0, disturb_time=3.0):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始略微抬高躯干让腿有个"坐下站起"的过程
    data.qpos[2] = 0.45
    mujoco.mj_forward(model, data)

    def step_once():
        set_stand_pose(data)
        # 在指定时刻施加一段时间的侧向外力扰动，模拟"推一下"测试平衡恢复能力
        if disturb_time <= data.time < disturb_time + 0.1:
            data.xfrc_applied[model.body("torso").id][1] = 10.0  # y方向持续冲击力
        else:
            data.xfrc_applied[model.body("torso").id][1] = 0.0
        mujoco.mj_step(model, data)

    if view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration:
                step_once()
                viewer.sync()
        return

    times, heights, roll_pitch = [], [], []
    while data.time < duration:
        step_once()
        times.append(data.time)
        heights.append(data.qpos[2])

        # 从四元数计算roll/pitch，模拟IMU读出的姿态角，用于评估平衡质量
        quat = data.qpos[3:7]
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        roll_pitch.append((np.rad2deg(roll), np.rad2deg(pitch)))

    roll_pitch = np.array(roll_pitch)
    max_roll_after_disturb = np.max(np.abs(roll_pitch[int(disturb_time/model.opt.timestep):, 0]))
    print(f"扰动后最大roll角度: {max_roll_after_disturb:.2f} 度")
    print(f"最终躯干高度: {heights[-1]:.4f} m (站立目标约0.3-0.35m)")
    print(f"{'平衡恢复良好' if max_roll_after_disturb < 20 and heights[-1] > 0.2 else '摔倒或未能恢复平衡，可尝试调整kp或STAND_HIP/KNEE角度'}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(times, heights)
        axes[0].axvline(disturb_time, color="red", linestyle="--", label="施加扰动")
        axes[0].set_ylabel("躯干高度(m)")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(times, roll_pitch[:, 0], label="roll")
        axes[1].plot(times, roll_pitch[:, 1], label="pitch")
        axes[1].axvline(disturb_time, color="red", linestyle="--")
        axes[1].set_ylabel("角度(度)")
        axes[1].set_xlabel("时间(s)")
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig("quadruped_balance.png", dpi=120)
        print("已保存 quadruped_balance.png")
    except ImportError:
        print("未安装matplotlib，跳过画图")


if __name__ == "__main__":
    run(view="--view" in sys.argv)
