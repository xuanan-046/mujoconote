# MuJoCo 物理仿真 6 大场景

配合"物理引擎与仿真模拟"学习路线使用的完整代码，每个场景一个独立文件夹，
包含 `model.xml`(MJCF模型) 和 `run.py`(控制/分析脚本)。

## 环境准备

```bash
pip install mujoco matplotlib --break-system-packages   # Linux/服务器环境去掉--break-system-packages也可
```

Windows/Mac 上直接 `pip install mujoco matplotlib` 即可，不需要额外配置。

## 各场景说明与运行方式

| 文件夹 | 场景 | 核心概念 | 运行命令 |
|---|---|---|---|
| `s1_bouncing_ball` | 自由落体+弹跳球 | 重力积分、恢复系数、接触求解器 | `python run.py` / `--view` |
| `s2_pendulum` | 单摆+双摆 | hinge关节、能量守恒、混沌现象 | `python run.py` / `--view` |
| `s3_cartpole` | 倒立摆 | 欠驱动系统、PID、LQR | `python run.py --controller pid\|lqr` / `--view` |
| `s4_arm_grasp` | 机械臂抓取 | 数值逆运动学、抓取稳定性 | `python run.py` / `--view` |
| `s5_quadruped` | 四足站立平衡 | 多点接触、PD控制、姿态反馈 | `python run.py` / `--view` |
| `s6_stacking` | 多物体堆叠 | 求解器收敛性(PGS vs 迭代次数) | `python run.py` / `--view --iterations N` |

无 `--view` 参数时脚本在后台运行，打印分析数据并保存对应的png分析图；
加上 `--view` 会打开 MuJoCo 自带的交互式可视化窗口(需要图形界面，服务器/容器环境不可用)。

## 踩坑记录(写代码时真实遇到的两个bug，值得记住)

1. **角度单位陷阱**：MJCF 默认把 `range`、`ref` 等角度类属性当作**度数**解析，
   而不是弧度。如果你写 `range="-1.57 1.57"` 想表达 ±90°，MuJoCo会把它理解成
   `-1.57°~1.57°`(几乎不能动)。解决办法是在 `<mujoco>` 标签内加一行：
   ```xml
   <compiler angle="radian"/>
   ```
   本项目所有模型都已经加上这个声明。这是新手写MJCF最容易踩的坑之一，
   排查方式是打印 `model.jnt_range` 看编译后的实际数值是否符合预期。

2. **执行器局部import踩坑**：在同一个Python函数里，如果只在 `if view:` 分支里
   写 `import mujoco.viewer`，Python会因为函数内出现了对 `mujoco` 的(局部)赋值，
   把整个函数体内的 `mujoco` 都当成局部变量，导致分支之前对全局 `mujoco` 的
   使用报 `UnboundLocalError`。解决办法是用 `from mujoco import viewer as mj_viewer`
   避免遮蔽外层的模块名。

3. **手指关节方向反了**：机械臂夹爪的 `slide` 关节，qpos=0 对应张开还是闭合
   取决于 `axis` 的方向定义，容易凭直觉写反导致夹爪在该开的时候闭合、
   该合的时候张开。建议用 `mj_forward` 单独验证一次关节值对应的实际世界坐标，
   而不是凭XML直觉猜。

4. **数值IK优于手写解析IK**：一开始用余弦定理手写了平面二连杆解析逆运动学，
   算出来的角度看似合理，实际因为连杆坐标系没完全对齐导致抓不到物体。
   后来改成"网格搜索+mj_forward验证真实末端位置"的数值IK，反而更快调对——
   这也是实际工程里常用的思路：与其苦想解析解，不如直接用仿真器做正向验证。

## 建议学习顺序

s1 → s2 → s3 → s6 → s4 → s5
(先打基础，再理解引擎局限性，最后做贴近机器人应用的综合场景)
