"""泳道(lane)抽象。每条泳道用同一段 MJCF 建模型, 按阶段跑 pipeline, 产出快照。

Lane 协议:
  name: str
  available() -> bool
  load(xml) -> LaneHandle
LaneHandle:
  reset(qpos, qvel, act, ctrl)                     设初值
  run_full(nstep, snap_stages) -> list[list[Snapshot]]   模式 A: 自由跑 nstep, 每步在 snap_stages 边界快照
  collision_only(qpos, qvel) -> np.ndarray         只跑到 S3, 返回 contact 数组(供 hybrid 用)
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .stages import STAGE_IDS, Snapshot, run_stage, snapshot

mujoco.set_mjcb_time(time.perf_counter)  # 让 d.timer 生效


# ============================================================ CPU (fp64 / fp32)
class MujocoHandle:
    def __init__(self, xml, dtype_tag):
        self.m = mujoco.MjModel.from_xml_string(xml)
        # 受控变量: 统一用 dense Jacobian, 便于逐行比 efc_J
        self.m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.d = mujoco.MjData(self.m)
        self.dtype_tag = dtype_tag  # "fp64" / "fp32"

    def reset(self, qpos=None, qvel=None, act=None, ctrl=None):
        mujoco.mj_resetData(self.m, self.d)
        if qpos is not None: self.d.qpos[:] = qpos
        if qvel is not None: self.d.qvel[:] = qvel
        if act is not None and self.d.act.size: self.d.act[:] = act
        if ctrl is not None and self.d.ctrl.size: self.d.ctrl[:] = ctrl
        mujoco.mj_forward(self.m, self.d)

    def run_full(self, nstep, snap_stages):
        out = []
        for step in range(nstep):
            snaps = []
            for sid in STAGE_IDS:
                secs = run_stage(self.m, self.d, sid)
                if sid in snap_stages:
                    snaps.append(snapshot(self.m, self.d, sid, step, secs))
            out.append(snaps)
        return out

    def collision_only(self, qpos, qvel):
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        for sid in ("S1", "S3"):
            run_stage(self.m, self.d, sid)
        from .stages import _contacts
        return _contacts(self.d)


class MujocoLane:
    def __init__(self, dtype_tag="fp64"):
        self.name = "cpu_" + dtype_tag
        self.dtype_tag = dtype_tag

    def available(self):
        if self.dtype_tag == "fp64":
            return not _is_single_build()
        return _is_single_build()

    def load(self, xml):
        return MujocoHandle(xml, self.dtype_tag)


def _is_single_build():
    """当前 mujoco wheel 是不是 mjtNum=float 构建。"""
    m = mujoco.MjModel.from_xml_string("<mujoco/>")
    return m.qpos0.dtype == np.float32


# ============================================================ MuJoCo Warp
class MjWarpHandle:
    # mujoco_warp 的硬限制:dense 雅可比只支持 nv<=60(put_model 里会抛 ValueError)。
    # 其他泳道(cpu_fp64/cpu_fp32/npu)统一强制 dense 是为了 efc_J 能逐元素直接比对；
    # 这里超过阈值就退化成 sparse——代价是这类大模型(目前只有 spheres_20, nv=120)
    # 的 S4 阶段 efc_J/efc_pos 对比会因为 dense/sparse 数据布局不同而报 shape_mismatch,
    # 是已知的、可接受的降级(不影响 S3/S9/S10,那些是逐自由度/逐接触的量,跟雅可比
    # 存储方式无关)。
    _DENSE_NV_MAX = 60

    def __init__(self, xml):
        import mujoco_warp as mjw
        self.mjw = mjw
        self.mjm = mujoco.MjModel.from_xml_string(xml)
        if self.mjm.nv <= self._DENSE_NV_MAX:
            self.mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        else:
            self.mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        self.mjd = mujoco.MjData(self.mjm)
        self.m = mjw.put_model(self.mjm)
        self.d = None

    def reset(self, qpos=None, qvel=None, act=None, ctrl=None):
        mujoco.mj_resetData(self.mjm, self.mjd)
        if qpos is not None: self.mjd.qpos[:] = qpos
        if qvel is not None: self.mjd.qvel[:] = qvel
        mujoco.mj_forward(self.mjm, self.mjd)
        self.d = self.mjw.put_data(self.mjm, self.mjd)

    def run_full(self, nstep, snap_stages):
        """MJWarp 不暴露与 C 完全一致的阶段切分; 这里退化为整步 + 末态快照。
        逐阶段对照请用 mujoco_warp 自带的 forward_test / constraint_test / solver_test。"""
        import numpy as _np
        mjw = self.mjw
        out = []
        for step in range(nstep):
            t0 = time.perf_counter()
            mjw.step(self.m, self.d)
            secs = time.perf_counter() - t0
            # 把 device 数据搬回 host mjData 以复用 snapshot()
            self.mjd.qpos[:] = self.d.qpos.numpy()[0]
            self.mjd.qvel[:] = self.d.qvel.numpy()[0]
            self.mjd.qacc[:] = self.d.qacc.numpy()[0]
            nacon = int(self.d.nacon.numpy()[0])
            self.mjd.ncon = nacon  # 近似
            snaps = [snapshot(self.mjm, self.mjd, "S10", step, secs)]
            out.append(snaps)
        return out

    def collision_only(self, qpos, qvel):
        mjw = self.mjw
        self.mjd.qpos[:] = qpos
        self.mjd.qvel[:] = qvel
        mujoco.mj_forward(self.mjm, self.mjd)
        d = mjw.put_data(self.mjm, self.mjd)
        mjw.collision(self.m, d)
        n = int(d.nacon.numpy()[0])
        if n == 0:
            return np.zeros((0, 21))
        dist = d.contact.dist.numpy()[:n]
        pos = d.contact.pos.numpy()[:n]
        frame = d.contact.frame.numpy()[:n].reshape(n, 9)
        dim = d.contact.dim.numpy()[:n] if hasattr(d.contact, "dim") else np.full(n, 3)
        geom = d.contact.geom.numpy()[:n] if hasattr(d.contact, "geom") else np.zeros((n, 2))
        fri = d.contact.friction.numpy()[:n][:, :5] if hasattr(d.contact, "friction") else np.zeros((n, 5))
        return np.column_stack([dist, pos, frame, dim, geom, fri])


class MjWarpLane:
    name = "mjwarp"

    def available(self):
        import os
        try:
            import mujoco_warp  # noqa: F401
            import warp as wp
            if os.environ.get("MJWARP_EVAL_FORCE", "0") == "1":
                # 强制启用，即使没有 CUDA —— Warp 有 CPU device 后备，可以在没有 GPU 的机器上
                # 把整条 mjwarp 泳道(put_model/put_data/collision/step)先跑通做联调，
                # 排除 harness/wiring 问题；跑出来的数字不代表真实 GPU 性能，只用于验证正确性。
                return True
            return wp.get_cuda_device_count() > 0
        except Exception:  # noqa: BLE001
            return False

    def load(self, xml):
        return MjWarpHandle(xml)


# ============================================================ NPU (被测对象)
# 真正要改的只有 npu_eval/npu_backend.py 里的三个函数。下面这层是通用胶水，不用动。
from . import npu_backend


class NpuHandle:
    """NPU 碰撞检测泳道。

    - collision_only(qpos, qvel) -> [K,21]：主机算运动学 → 调 NPU 碰撞 → 规范排序返回。
      hybrid 模式用这个（NPU 碰撞 + CPU 下游求解），回答"碰撞差异是否影响仿真"。
    - run_full(nstep, snap_stages)：整步在主机 CPU 上跑，只把 S3 碰撞换成 NPU 结果，
      每步在阶段边界快照。free 模式用这个（逐阶段 + 轨迹发散）。
    """

    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.d = mujoco.MjData(self.m)          # 主机驱动用（下游阶段、运动学）
        self.static = npu_backend.build_static(self.m)
        self.handle = npu_backend.npu_open(self.static)

    def close(self):
        npu_backend.npu_close(self.handle)

    def reset(self, qpos=None, qvel=None, act=None, ctrl=None):
        mujoco.mj_resetData(self.m, self.d)
        if qpos is not None: self.d.qpos[:] = qpos
        if qvel is not None: self.d.qvel[:] = qvel
        if act is not None and self.d.act.size: self.d.act[:] = act
        if ctrl is not None and self.d.ctrl.size: self.d.ctrl[:] = ctrl
        mujoco.mj_forward(self.m, self.d)

    def _npu_contacts(self, qpos, qvel):
        xpos, xmat = npu_backend.host_kinematics(self.m, self.d, qpos, qvel)
        c = np.asarray(npu_backend.npu_collide(self.handle, xpos, xmat), dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != npu_backend.NCOL:
            raise ValueError(f"npu_collide 返回 shape={c.shape}, 期望 [K,{npu_backend.NCOL}]")
        return npu_backend.canonical_sort(c)

    def collision_only(self, qpos, qvel):
        return self._npu_contacts(qpos, qvel)

    def _inject_and_step(self, c_npu):
        """在 self.d 上：先 CPU 跑 S1+S3（拿到接触参数 solref/solimp/... 和确定性行顺序），
        再把每个 CPU 接触槽用"同 geom 对、位置最近"的 NPU 接触覆盖其 dist/pos/frame，
        保持 efc_J 行顺序与参考一致（否则行错位会造成假的巨大相对差）。
        NPU 比 CPU 多出的接触，追加到末尾（参数从同 geom 对的槽复制，标记 mismatch）。
        然后跑 S4..S10。返回 (注入数, 是否 ncon 不一致)。"""
        d, m = self.d, self.m
        run_stage(m, d, "S1")
        run_stage(m, d, "S3")
        ncpu = int(d.ncon)
        npu = np.atleast_2d(c_npu) if c_npu.size else np.zeros((0, npu_backend.NCOL))
        used = set()
        for i in range(ncpu):
            cc = d.contact[i]
            g = (int(cc.geom[0]), int(cc.geom[1]))
            best, bd = -1, 1e18
            for j in range(len(npu)):
                if j in used:
                    continue
                if (int(npu[j, 14]), int(npu[j, 15])) != g:
                    continue
                dd = float(np.linalg.norm(npu[j, 1:4] - cc.pos))
                if dd < bd:
                    best, bd = j, dd
            if best >= 0:
                used.add(best)
                r = npu[best]
                cc.dist = float(r[0]); cc.pos[:] = r[1:4]; cc.frame[:] = r[4:13]
        extra = [j for j in range(len(npu)) if j not in used]
        cap = len(d.contact)
        for k, j in enumerate(extra):
            slot = ncpu + k
            if slot >= cap:
                break
            r = npu[j]
            src = next((i for i in range(ncpu)
                        if (int(d.contact[i].geom[0]), int(d.contact[i].geom[1]))
                        == (int(r[14]), int(r[15]))), 0 if ncpu else None)
            if src is not None:
                s = d.contact[src]
                d.contact[slot].includemargin = s.includemargin
                d.contact[slot].friction[:] = s.friction
                d.contact[slot].solref[:] = s.solref
                d.contact[slot].solimp[:] = s.solimp
                d.contact[slot].dim = s.dim
            d.contact[slot].dist = float(r[0])
            d.contact[slot].pos[:] = r[1:4]
            d.contact[slot].frame[:] = r[4:13]
            d.contact[slot].geom[:] = r[14:16].astype(int)
        d.ncon = min(ncpu + len(extra), cap)
        for sid in ("S4", "S5", "S6", "S7", "S9", "S10"):
            run_stage(m, d, sid)
        return d.ncon, (d.ncon != ncpu)

    def run_full(self, nstep, snap_stages):
        import time as _t
        out = []
        for step in range(nstep):
            snaps = []
            qpos = np.array(self.d.qpos); qvel = np.array(self.d.qvel)
            c_npu = self._npu_contacts(qpos, qvel)
            # S1 边界快照（若需要）
            if "S1" in snap_stages:
                run_stage(self.m, self.d, "S1")
                snaps.append(snapshot(self.m, self.d, "S1", step))
                self.d.qpos[:] = qpos; self.d.qvel[:] = qvel
            t0 = _t.perf_counter()
            self._inject_and_step(c_npu)
            secs = _t.perf_counter() - t0
            for sid in snap_stages:
                if sid == "S1":
                    continue
                s = snapshot(self.m, self.d, sid, step, secs if sid == "S3" else 0.0)
                if sid == "S3" and c_npu.size:
                    s.fields["contact"] = c_npu
                    s.fields["ncon"] = np.array([len(c_npu)])
                snaps.append(s)
            out.append(snaps)
        return out


class NpuLane:
    name = "npu"

    def available(self):
        import os
        return os.environ.get("NPU_EVAL_ENABLE", "0") == "1"

    def load(self, xml):
        return NpuHandle(xml)


# ============================================================ Mock NPU
def _mock_perturb(c):
    """对 [N,21] 接触数组按 NPU_MOCK_PROFILE 施加扰动，返回 fp64。
    profile:
      fp32      仅把接触字段做一次 fp32 往返（默认）—— 纯精度损失下界的代理
      fp32+jit  fp32 + 一点确定性扰动（穿透深度 ~1e-4）—— 叠加移植/重排小偏差
      none      不动（用于自检：npu 泳道应与 fp64 逐位一致）
    """
    import os
    prof = os.environ.get("NPU_MOCK_PROFILE", "fp32")
    if c.size == 0 or prof == "none":
        return c
    c = c.astype(np.float32).astype(np.float64)
    if prof == "fp32+jit":
        c[:, 0] += 1e-4 * np.sin(c[:, 1] * 97.0)  # 扰动穿透深度
    return c


class MockNpuHandle(NpuHandle):
    """无 NPU 时的占位。见 _mock_perturb 的 profile 说明。

    继承 NpuHandle（而不是 MujocoHandle）是有意为之：只重写 _npu_contacts，让它用
    CPU 参考碰撞(S1+S3) + mock 扰动 生成"NPU 结果"；run_full/collision_only 都继承
    NpuHandle 原样的 _inject_and_step 传播逻辑，扰动会像真 NPU 一样真的被注入、
    重新跑 S4~S10。

    [已修复的 bug] 旧版本继承 MujocoHandle，run_full 只是跑一遍纯 fp64 参考轨迹，
    然后在事后把 S3 快照里的 contact 字段覆盖成扰动值展示——扰动从未真正参与 S4~S10
    的计算，所以 free 模式下 S4/S6/S9/S10 恒为 0，制造"下游没问题"的假象。
    hybrid 模式不受影响（它是在 runner.py 里独立注入的），但 free 模式此前一直是
    摆设。现在两个模式口径一致，都走真实注入。见 `首轮完整数据汇总.md` §0.3。
    """

    def __init__(self, xml, dtype_tag="fp32"):
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.d = mujoco.MjData(self.m)      # 主机驱动用（下游阶段、注入目标）
        self._ref = mujoco.MjData(self.m)   # 独立参考态，仅供 _npu_contacts 内部计算碰撞
        self.dtype_tag = dtype_tag

    def close(self):
        pass  # 没有真实 NPU 资源需要释放

    def _npu_contacts(self, qpos, qvel):
        d = self._ref
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        run_stage(self.m, d, "S1")
        run_stage(self.m, d, "S3")
        from .stages import _contacts
        c = _mock_perturb(_contacts(d))
        return npu_backend.canonical_sort(c)


class MockNpuLane:
    name = "npu"  # 占位, 报告里标注为 mock

    def available(self):
        return True

    def load(self, xml):
        return MockNpuHandle(xml, "fp32")


# ============================================================ registry
def get_lanes(names, allow_mock_npu=True):
    reg = {
        "cpu_fp64": MujocoLane("fp64"),
        "cpu_fp32": MujocoLane("fp32"),
        "mjwarp": MjWarpLane(),
        "npu": NpuLane(),
    }
    lanes = []
    for n in names:
        lane = reg[n]
        if n == "npu" and not lane.available() and allow_mock_npu:
            lane = MockNpuLane()
        lanes.append(lane)
    return lanes
