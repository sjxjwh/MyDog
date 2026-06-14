# 待办：五次多项式 + 静摩擦约束控制方案

## 已完成

- [x] `QuinticTrajectory1D` / `QuinticTrajectory3D` — C² 连续五次多项式轨迹 (`trajectory.py`)
- [x] `FrictionForceDistributor` — 2 腿零空间解析力分配 (`friction_force.py`)
- [x] `QuinticFootTrajectoryPlanner` — 五次摆动 + 正弦 Z 抬升 (`gait.py`)
- [x] `QuinticFrictionController` — 分层控制器 (Tier 1/2/3) (`force_controller.py`)
- [x] CLI 集成 — `--quintic`, `--mu-max`, `--adapt-params` (`main.py`)
- [x] Stance 脚跟随身体旋转 — `com + R·offset`（替代世界锁死）
- [x] 纯旋转 CoM 锚点 PD (Kp=20) — 仅在 vx=vy=0 模式
- [x] Kp_vy 50 → 200 — 增强侧向阻尼

## 当前效果

| 指标 | 数值 | 说明 |
|------|------|------|
| vx 跟踪 | 64% (0.19/0.30) | 与 MPC (69%) 持平 |
| vyaw 跟踪（纯旋转） | 峰值 50% / 均值 38% | MPC 仅 1% |
| 纯旋转 CoM 漂移 | ~4.3cm / 3s | Trot 对角几何限制 |
| 摆动平滑性 | C² 连续 | MPC/Body PD 仅 C0 |

## 高优先级

- [ ] **纯旋转 CoM 漂移彻底消除（目标 < 1cm/3s）**
  - 分析：Trot 对角支撑腿不共线 → Mz 差分力带出残余线力
  - 方案 A：增加旋转轨道偏移 — stance 脚在世界系画完整的圆
  - 方案 B：换 Pace 步态做纯偏航（同侧腿同时着地 → 力对称）
  - 方案 C：零空间解中显式加入 Σf=0 约束的积分修正

- [ ] **Tier 1 参数自适应 — 从规则式升级为梯度优化**
  - 当前：Bang-bang（μ<0.85 ↑L，μ>0.95 ↓L）
  - 目标：`min_{T,L} w_v·v_err² + w_μ·μ²`
  - 方向：滑动窗口平均 + 有限差分，或模型辅助梯度

- [ ] **五次多项式 Z 分量修复**
  - 当前：X/Y 是五次，Z 仍是正弦
  - 原因：Z 边界 x0=xT 且零导数 → 五次退化
  - 方案：双段五次 — 中点分两段拼接

## 中优先级

- [ ] **扩展 FrictionForceDistributor 到 Pace 步态**
  - Pace 左右腿同时支撑 → 力对称 → 零空间可能变 2D

- [ ] **μ_utilized 作为 RL feedback signal**
  - 输出给 UniLab：`obs["friction_utilization"]`

- [ ] **场景摩擦自适应**
  - 在线估计地面实际 μ，动态调整 `mu_max`

## 低优先级

- [ ] **步长 L 自动跟随 v_target**
  - `L = v_target * duty * T_cycle` 作为默认值

- [ ] **位控模式 stance 锚定同步更新**
  - `BodyController` 也需 stance 跟随旋转

- [ ] **单元测试**
  - `test_quintic_trajectory.py` / `test_friction_force.py`

## 文件清单

```
src/
  trajectory.py         — QuinticTrajectory1D, QuinticTrajectory3D
  gait.py               — QuinticFootTrajectoryPlanner
  friction_force.py     — FrictionForceDistributor [新]
  force_controller.py   — QuinticFrictionController + stance 旋转跟踪
  main.py               — --quintic, --mu-max, --adapt-params
CLAUDE.md               — 文档已更新
```
