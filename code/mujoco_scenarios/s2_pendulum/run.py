"""
场景2: 单摆(能量守恒验证) + 双摆(混沌现象)
运行: python run.py            # 输出能量曲线与轨迹发散图
运行: python run.py --view     # 可视化窗口
"""
import sys
import numpy as np
import mujoco

MODEL_PATH = "model.xml"


def compute_single_pendulum_energy(model, data):
    """单摆总能量 = 动能 + 势能，用来检验数值积分是否守恒"""
    qvel = data.qvel[0]
    qpos = data.qpos[0]
    L, m, g = 1.0, 1.0, 9.81
    I = m * L ** 2  # 简化为质点摆
    KE = 0.5 * I * qvel ** 2
    PE = -m * g * L * np.cos(qpos)  # 以最低点为势能零点
    return KE + PE


def run_headless():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 设置初始角度: 单摆60度, 双摆A/B分别60度与60.0001度
    theta0 = np.deg2rad(60)
    theta0_b = np.deg2rad(60.0001)
    data.qpos[0] = theta0        # single pendulum
    data.qpos[1] = theta0        # double A joint1
    data.qpos[2] = 0.0           # double A joint2
    data.qpos[3] = theta0_b      # double B joint1
    data.qpos[4] = 0.0           # double B joint2
    mujoco.mj_forward(model, data)

    duration = 6.0
    times, energies, tip_diff = [], [], []

    while data.time < duration:
        mujoco.mj_step(model, data)
        times.append(data.time)
        energies.append(compute_single_pendulum_energy(model, data))

        # 双摆A和B末端摆锤在世界坐标系下的位置差 -> 体现混沌发散
        pos_A = data.xpos[model.body("dA_link2").id]
        pos_B = data.xpos[model.body("dB_link2").id]
        tip_diff.append(np.linalg.norm(pos_A - pos_B))

    e0 = energies[0]
    drift = max(abs(e - e0) for e in energies)
    print(f"单摆初始能量: {e0:.6f} J")
    print(f"仿真过程中最大能量漂移: {drift:.6f} J  (RK4积分器下应该很小)")
    print(f"双摆A/B初始角度差仅0.0001度，{duration}s后末端位置差: {tip_diff[-1]:.4f} m")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(times, energies)
        axes[0].set_title("单摆总能量随时间变化(应接近水平线)")
        axes[0].set_xlabel("时间(s)")
        axes[0].set_ylabel("能量(J)")
        axes[0].grid(True)

        axes[1].plot(times, tip_diff)
        axes[1].set_title("双摆A/B末端位置差(混沌发散)")
        axes[1].set_xlabel("时间(s)")
        axes[1].set_ylabel("位置差(m)")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("pendulum_analysis.png", dpi=120)
        print("已保存 pendulum_analysis.png")
    except ImportError:
        print("未安装matplotlib，跳过画图")


def run_viewer():
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    data.qpos[0] = np.deg2rad(60)
    data.qpos[1] = np.deg2rad(60)
    data.qpos[3] = np.deg2rad(60.0001)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    if "--view" in sys.argv:
        run_viewer()
    else:
        run_headless()
