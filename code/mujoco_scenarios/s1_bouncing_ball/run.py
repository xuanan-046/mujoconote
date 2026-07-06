"""
场景1: 自由落体 + 弹跳球
运行: python run.py            # 无窗口，输出高度曲线图 height.png
运行: python run.py --view     # 打开可视化窗口观察弹跳过程
"""
import sys
import mujoco
import numpy as np

MODEL_PATH = "model.xml"


def run_headless():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    duration = 4.0
    times, h_rubber, h_stone = [], [], []

    while data.time < duration:
        mujoco.mj_step(model, data)
        times.append(data.time)
        # freejoint 的前3个qpos是xyz位置
        h_rubber.append(data.qpos[2])   # ball_rubber
        h_stone.append(data.qpos[9])    # ball_stone (7个qpos一组: 3位置+4四元数)

    # 解析解对比: 自由落体理论最高点应等于初始高度(无阻力情况下能量守恒)
    print(f"橡胶球最终反弹的最大高度(仿真): {max(h_rubber[len(h_rubber)//2:]):.4f} m")
    print(f"石头球最终反弹的最大高度(仿真): {max(h_stone[len(h_stone)//2:]):.4f} m")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(times, h_rubber, label="橡胶地面球")
        plt.plot(times, h_stone, label="石头地面球")
        plt.xlabel("时间 (s)")
        plt.ylabel("高度 (m)")
        plt.title("不同恢复系数下的弹跳高度对比")
        plt.legend()
        plt.grid(True)
        plt.savefig("height.png", dpi=120)
        print("已保存 height.png")
    except ImportError:
        print("未安装matplotlib，跳过画图 (pip install matplotlib)")


def run_viewer():
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    if "--view" in sys.argv:
        run_viewer()
    else:
        run_headless()
