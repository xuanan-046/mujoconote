"""
场景4: 3-DOF机械臂抓取方块
运行: python run.py            # 无窗口，打印抓取结果
运行: python run.py --view     # 可视化窗口观察抓取全过程
"""
import sys
import numpy as np
import mujoco

MODEL_PATH = "model.xml"

def solve_ik_numerically(model, data, target_x, target_z, n_grid=120):
    """
    数值逆运动学: 在shoulder/elbow的可行角度范围内网格搜索，
    用mj_forward(不做动力学积分，只算正运动学)找到让gripper_center
    最接近目标点(target_x, target_z)的一组关节角。
    比手写解析IK更稳妥，因为直接用真实模型的几何尺寸做正向验证，
    不用手动核对每段连杆的坐标系和偏移量。
    """
    shoulder_range = np.linspace(-1.55, 1.55, n_grid)
    elbow_range = np.linspace(-2.45, -0.05, n_grid)

    best_err = np.inf
    best_angles = (0.0, -0.5)
    site_id = model.site("gripper_center").id

    saved_qpos = data.qpos.copy()
    for s in shoulder_range:
        for e in elbow_range:
            data.qpos[7] = 0.0   # base_yaw
            data.qpos[8] = s     # shoulder
            data.qpos[9] = e     # elbow
            data.qpos[10] = 0.04  # fingers open
            data.qpos[11] = 0.04
            mujoco.mj_forward(model, data)
            gx, gz = data.site_xpos[site_id][0], data.site_xpos[site_id][2]
            err = (gx - target_x) ** 2 + (gz - target_z) ** 2
            if err < best_err:
                best_err = err
                best_angles = (s, e)

    data.qpos[:] = saved_qpos
    mujoco.mj_forward(model, data)
    return best_angles


def run(view=False):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cube_target_x = 0.35  # 方块在x轴上的位置(见model.xml)

    # 用数值IK分别求"方块上方悬停点"和"下降抓取点"对应的关节角度
    shoulder_pre, elbow_pre = solve_ik_numerically(model, data, cube_target_x, 0.20)
    shoulder_grasp, elbow_grasp = solve_ik_numerically(model, data, cube_target_x, 0.03)
    print(f"IK求解结果: 预抓取角度=({np.rad2deg(shoulder_pre):.1f}, {np.rad2deg(elbow_pre):.1f})度, "
          f"抓取角度=({np.rad2deg(shoulder_grasp):.1f}, {np.rad2deg(elbow_grasp):.1f})度")

    # 注意: finger关节qpos=0对应张开(离中心0.04m)，qpos越大越靠近中心(闭合)
    # 方块半边长0.03m，闭合目标设为0.015让手指略微压入方块表面产生夹持力
    FINGER_OPEN, FINGER_CLOSED = 0.0, 0.032

    # 定义控制阶段: (持续时间, base_yaw, shoulder, elbow, finger_l, finger_r)
    stages = [
        (1.0, 0.0, shoulder_pre,   elbow_pre,   FINGER_OPEN,   FINGER_OPEN),   # 移动到方块正上方，张开夹爪
        (1.0, 0.0, shoulder_grasp, elbow_grasp, FINGER_OPEN,   FINGER_OPEN),   # 下降到抓取高度
        (0.8, 0.0, shoulder_grasp, elbow_grasp, FINGER_CLOSED, FINGER_CLOSED), # 闭合夹爪
        (1.5, 0.0, shoulder_pre,   elbow_pre,   FINGER_CLOSED, FINGER_CLOSED), # 抬起
    ]

    def set_ctrl(base_yaw, shoulder, elbow, fl, fr):
        data.ctrl[0] = base_yaw
        data.ctrl[1] = shoulder
        data.ctrl[2] = elbow
        data.ctrl[3] = fl
        data.ctrl[4] = fr

    if view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            for dur, *ctrl in stages:
                t_end = data.time + dur
                while data.time < t_end and viewer.is_running():
                    set_ctrl(*ctrl)
                    mujoco.mj_step(model, data)
                    viewer.sync()
        return

    cube_heights = []
    for dur, *ctrl in stages:
        t_end = data.time + dur
        while data.time < t_end:
            set_ctrl(*ctrl)
            mujoco.mj_step(model, data)
            cube_heights.append(data.qpos[2])  # cube是第一个freejoint body，qpos[0:3]是xyz位置

    final_height = data.qpos[2]  # cube z位置
    success = final_height > 0.08  # 抓起并抬升超过初始高度即视为成功
    print(f"方块最终高度: {final_height:.4f} m")
    print(f"抓取{'成功' if success else '失败'} (可以尝试调整model.xml中的friction系数来测试不同抓取成功率)")


if __name__ == "__main__":
    run(view="--view" in sys.argv)
