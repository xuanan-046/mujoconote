"""
场景6: 多物体堆叠 - 求解器收敛性压力测试
运行: python run.py                 # 对比不同迭代次数下的穿模程度
运行: python run.py --view --iterations 50   # 可视化某个具体配置

核心思路: 堆叠N个方块，求解器迭代次数不够时，方块之间会互相"穿模"(penetration)，
表现为方块间距小于方块尺寸之和，或者堆叠高度低于理论高度。
"""
import sys
import argparse
import numpy as np
import mujoco

N_BOXES = 10
BOX_SIZE = 0.03  # 半边长
GAP = 0.002      # 初始下落时留的小间隙，避免初始穿透


def build_model_xml(iterations, solver="Newton"):
    """动态生成堆叠N个box的MJCF，可指定求解器类型和迭代次数"""
    with open("model_template.xml", "r", encoding="utf-8") as f:
        template = f.read()

    solver_opt = f'solver="{solver}" iterations="{iterations}"'
    template = template.replace("SOLVER_OPTION_PLACEHOLDER", solver_opt)

    boxes_xml = []
    z = BOX_SIZE
    for i in range(N_BOXES):
        boxes_xml.append(f'''
    <body name="box{i}" pos="0 0 {z:.4f}">
      <freejoint/>
      <geom name="geom{i}" type="box" size="{BOX_SIZE} {BOX_SIZE} {BOX_SIZE}"
            mass="0.1" rgba="{0.2+i*0.07:.2f} {0.9-i*0.05:.2f} 0.3 1" friction="0.8 0.02 0.001"/>
    </body>''')
        z += 2 * BOX_SIZE + GAP

    template = template.replace("BOX_BODIES_PLACEHOLDER", "\n".join(boxes_xml))
    return template


def measure_penetration(model, data):
    """
    测量堆叠稳定后的总高度误差:
    理论高度 = N * 2 * BOX_SIZE (方块紧密堆叠且不穿模的情况下)
    实际高度用最顶部方块的z坐标 + BOX_SIZE 估算
    差值越大说明穿模/塌陷越严重
    """
    top_box_z = max(data.qpos[7 * i + 2] for i in range(N_BOXES))
    actual_top = top_box_z + BOX_SIZE
    theoretical_top = N_BOXES * 2 * BOX_SIZE
    return theoretical_top - actual_top  # 正值表示比理论矮(有塌陷/穿模)


def run_case(iterations, solver="Newton", duration=3.0):
    xml = build_model_xml(iterations, solver)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    while data.time < duration:
        mujoco.mj_step(model, data)

    return measure_penetration(model, data)


def run_headless():
    print(f"堆叠{N_BOXES}个方块，测试不同求解器迭代次数下的稳定性\n")
    print(f"{'迭代次数':<10}{'求解器':<10}{'高度误差(m)':<15}{'说明'}")
    # PGS(Projected Gauss-Seidel)属于逐步松弛类算法，收敛速度对迭代次数很敏感，
    # 比Newton法更适合用来演示"迭代次数不够->穿模"的现象
    for iterations in [1, 2, 3, 5, 10, 30, 100]:
        err = run_case(iterations, solver="PGS")
        note = "明显塌陷/穿模" if err > 0.02 else ("轻微误差" if err > 0.005 else "基本稳定")
        print(f"{iterations:<10}{'PGS':<10}{err:<15.4f}{note}")

    print("\n结论: 迭代次数太少时，接触约束求解不充分，方块会互相嵌入，")
    print("导致整体高度比理论值低很多。这也是为什么实时仿真(如游戏、RL训练)")
    print("要在'物理精度'和'计算速度'之间做权衡的原因。")


def run_viewer(iterations, solver="Newton"):
    import mujoco.viewer
    xml = build_model_xml(iterations, solver)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--solver", default="Newton", choices=["PGS", "CG", "Newton"])
    args = parser.parse_args()

    if args.view:
        run_viewer(args.iterations, args.solver)
    else:
        run_headless()
