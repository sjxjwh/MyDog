# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

分层控制架构，在 MuJoCo 中对宇树 Go1 四足机器人实现 locomotion 控制与智能决策：

```
┌──────────────────────────────────────────────────┐
│     上层：RL 决策（UniLab）                        │
│     速度选择、方向规划、步态切换、                   │
│     地面摩擦估计 (μ_max)、地形适应、行为策略         │
├──────────────────────────────────────────────────┤
│     下层：传统控制（MyDog 本体）                    │
│     Tier 1: 步态参数自适应 (~10 Hz, 可选)          │
│     Tier 2: 五次多项式摆动轨迹 (C² 连续)           │
│     Tier 3: 静摩擦约束力分配 (500 Hz)              │
│     ＋ IK/位控/MPC 等其他模式                      │
├──────────────────────────────────────────────────┤
│     物理引擎：MuJoCo（Go1 模型）                    │
└──────────────────────────────────────────────────┘
```

- **下层（MyDog 自身）**：用传统控制方法做整机的 locomotion 原语
  - 已完成：单腿足尖轨迹跟踪 ✅，四足步态规划与控制（trot/walk/pace/bound）✅
  - 已完成：浮基动力学仿真（MIT 力控 + 位控 stance 锚定）✅
  - 已完成：SRB 凸 MPC + MIT 阻抗控制 ✅
  - 已完成：三次多项式 + 静摩擦约束力分配（Quintic+Friction 三层架构）✅
  - 已完成：偏航控制全链路修复 — Izz 矫正 + 混合偏航架构 + 偏心诊断 ✅
  - 进行中：RL 集成 — 让 RL 决策步态类型 + 估计地面 μ_max
  - 进行中：sim2real 迁移
- **上层（UniLab）**：用 RL 做智能决策
  - 部署于本地 `/home/scj/UniLab`
  - 输出高层动作：**目标速度、航向角、步态类型、地面摩擦系数 μ_max**
  - 下发到下层传统控制器执行：μ_max 直接传入 Tier 3 摩擦锥约束，步态类型选择 Tier 2 轨迹+Tier 3 力分配策略

## 技术栈

- Python 3.10
- **物理引擎**: MuJoCo 3.9.0（MyDog 仿真）, MuJoCoUni 3.8.0（UniLab 依赖）
- **数值计算**: NumPy, SciPy, matplotlib
- **模型格式**: MJCF（原生） + URDF（自动转换）
- **底层控制**: 解析 FK 的 Gauss-Newton IK + 相位调度步态规划 + Body PD 力控 / SRB 凸 MPC + MIT 阻抗控制（torque via JᵀF）
- **优化求解**: OSQP — 运筹分裂二次规划求解器（MPC QP）
- **上层决策**: UniLab — 异构 CPU/GPU RL 运行时，PPO/SAC/TD3/APPO
- **包管理**: pip（MyDog 项目）+ uv（UniLab 项目）

## 环境

### MyDog 项目（传统控制层）
```bash
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### UniLab 框架（RL 决策层，`/home/scj/UniLab`）
```bash
cd ~/UniLab
uv sync --extra motrix
```

两个项目的 Python 环境独立：MyDog 用标准 MuJoCo 3.9.0，UniLab 用 MuJoCoUni 3.8.0。

## 项目架构

```
~/                           # 用户 home 目录
├── MyDog/                   # 本项目（传统控制 + RL 集成层）
│   ├── model/
│   │   ├── go1.xml          #   原始 MJCF（浮基，带 freejoint）
│   │   ├── go1_fixed.xml    #   固定基座变体（自动生成）
│   │   ├── scene.xml        #   浮基场景 = go1 + 地面 + 天空 + 灯光
│   │   ├── scene_fixed.xml  #   固定基座场景
│   │   ├── scene_hifric.xml #   高精度接触力场景（力控用）
│   │   └── unitree/         #   URDF 源文件
│   ├── src/
│   │   ├── simulator.py     #   MuJoCo 仿真封装（加载/step/name index）
│   │   ├── kinematics.py    #   解析 FK / Jacobian（LegKinematics）
│   │   ├── trajectory.py    #   笛卡尔轨迹生成（circle/line/sine/lissajous）
│   │   ├── controller.py    #   IKFootController — 单腿 IK 位置控制
│   │   ├── gait.py          #   步态调度 + 足尖轨迹规划
│   │   ├── body_controller.py   # BodyController — 四腿协调（位控）
│   │   ├── force_controller.py  # MITBodyController + MPCMITBodyController + QuinticFrictionController
│   │   ├── friction_force.py    #   FrictionForceDistributor — 静摩擦约束力分配
│   │   ├── mpc_controller.py    #   SRB 凸 MPC 求解器 (OSQP, N=10, dt=30ms)
│   │   ├── urdf_loader.py   #   URDF → MJCF 转换
│   │   ├── rl/              #   ⭐ RL 集成层（待实现）
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py   #   UniLab 接口适配
│   │   │   └── env.py       #   基于 MyDog 仿真器的 Gym 环境
│   │   └── main.py          #   入口：单腿/步态/力控模式
│   └── output/              #   分析图表输出
│
└── UniLab/                  # 外部项目（不放入 MyDog，独立 git 管理）
    ├── src/unilab/
    │   ├── algos/           #   RL 算法
    │   ├── envs/            #   Gym 环境
    │   ├── training/        #   训练管线
    │   └── ...
    └── conf/                #   Hydra 配置
```

## 控制架构

### 两层控制器

```
┌─────────────────────────────────────────────────┐
│  MPC Mode (--force --mpc --float)               │
│  MPCMITBodyController ← MITBodyController       │
│  ├─ MPC (31 Hz): SRB 凸 QP → 优化 GRF           │
│  │    N=10, dt=30ms, OSQP warm start            │
│  │    12D 状态 (rpy, p, ω, v) + 12D 控制 (4×3D F)│
│  │    约束: 摩擦锥 (μ=0.6) + fz∈[1,150]           │
│  ├─ Stance: τ = Jᵀ·(F_mpc + Kp·Δx - Kd·v)       │
│  ├─ Swing:  τ = Jᵀ·(Kp·Δx - Kd·v)  (纯阻抗)     │
│  └─ QP 失败 → 自动 fallback 到 Body PD           │
├─────────────────────────────────────────────────┤
│  Quintic+Friction Mode (--force --quintic --float) │
│  QuinticFrictionController ← MITBodyController  │
│  ├─ Tier 1 (~10Hz, 可选): 步态参数自适应             │
│  │    RL 决策步态类型 → Tier 2 轨迹 + Tier 3 力分配  │
│  │    RL 估计 μ_max → 传入 Tier 3 摩擦锥约束         │
│  │    --adapt-params: 规则化调整 step_length         │
│  ├─ Tier 2 (按需): 五次多项式摆动轨迹 (C² 连续)       │
│  │    QuinticFootTrajectoryPlanner               │
│  ├─ Tier 3 (500Hz): 静摩擦约束力分配              │
│  │    FrictionForceDistributor                   │
│  │    2 腿零空间解析解 + 线性化摩擦锥约束           │
│  │    f = f_particular + α·f_null, α 由约束确定   │
│  │    μ_max 来源: RL 估计 > 手动指定 (--mu-max)     │
│  ├─ Stance: τ = Jᵀ·(F_friction + Kp·Δx - Kd·v)   │
│  ├─ Swing:  τ = Jᵀ·(Kp·Δx - Kd·v) (五次轨迹跟踪)  │
│  └─ 摩擦利用率 μ_utilized 实时监控                 │
├─────────────────────────────────────────────────┤
│  Force Mode (--force --float)                   │
│  MITBodyController                              │
│  ├─ Body PD: 高度/姿态/速度 → 身体 wrench        │
│  ├─ Force distribution: wrench → stance 腿 GRF  │
│  ├─ Stance: 阻抗控制 + 前馈力 (Kp·Δx - Kd·v + ff)│
│  ├─ Swing:  阻抗控制跟踪轨迹 (Kp·Δx - Kd·v)       │
│  └─ τ = Jᵀ · F  (qfrc_applied, 不走 position actuator) │
├─────────────────────────────────────────────────┤
│  Position Mode (--gait [--float])               │
│  BodyController                                 │
│  ├─ GaitScheduler: 相位调度 stance/swing         │
│  ├─ FootTrajectoryPlanner: hip 系轨迹            │
│  ├─ Fixed base: target_hip → target_world       │
│  ├─ Floating: stance 锚定 + swing hip 轨迹       │
│  └─ IKFootController.solve_ik → position ctrl   │
├─────────────────────────────────────────────────┤
│  Single Leg Mode (--leg FR --traj circle)       │
│  IKFootController 直接控制单腿                    │
└─────────────────────────────────────────────────┘
```

### IK 控制器 (`controller.py`)

- **`solve_ik`**：Gauss-Newton 阻尼最小二乘，使用**解析 FK + MuJoCo 偏移修正**（`_moco_offset`）
  - 解析 FK (`LegKinematics`) 忽略 MJCF 的大腿侧向偏移 `[0, ±0.08, 0]`
  - `_moco_offset` 必须在初始化后设置，否则 IK 最终 MuJoCo FK 验证失败（误差 > 5cm）→ 返回 None
  - `BodyController.__init__` 和 `MITBodyController.__init__` 自动设置偏移
  - 单腿模式 (`run_simulation`/`run_simulation_viewer`) 需要在设 home 角度后手动设置
  - 仅最终验证调一次 `_forward_safe`（避免 WSL2 上 MuJoCo SIGFPE）
  - 阻尼 0.01，无 line search，100 次迭代上限
  - 关节限位带 1% margin（`_clamp_q`）

### 步态系统 (`gait.py`)

- **`GaitScheduler`**：相位 = `(offset[leg] + t/T_cycle) % 1.0`，φ < duty_factor → stance
- **`GaitType`**：TROT (对角同步)、WALK (波形)、PACE (同侧同步)、BOUND (前后同步)
- **`FootTrajectoryPlanner`**：
  - Stance: 脚在 hip 系中后移（x = neutral_x + L/2 - L·s）
  - Swing: 脚前移 + 正弦抬升（z = neutral_z + H·sin(π·s)）
  - 支持 warm_up 渐进斜坡
  - 浮基模式：XY 用 hip 系（`get_target_hip_xy`），Z 用世界系（`get_target_world_z`：stance=0，swing=H·sin）

### 浮基 stance 锚定 (`body_controller.py`)

- Stance 进入时记录当前足端世界 XY 作为锚点 → stance 期间脚固定不动 → 防止滑动
- Swing 按规划轨迹运动（hip 系 XY → 世界 XY + 世界 Z）
- `_last_gait_state` 跟踪状态转换，`_stance_anchor` 存储锚点

### MIT 力控 + MPC (`force_controller.py` + `mpc_controller.py`)

- **`LegTorqueController`**：力矩控制原语
  - `apply_force(F_world)`: τ = Jᵀ·F，限幅 ±23 Nm
  - `apply_impedance(target, Kp, Kd, ff)`: F = ff + Kp·(xᵈ-x) - Kd·(v-vᵈ)，限力 150N
  - `jacobian_world(q)`: hip_rot @ J_hip(q) → 3×3 世界 Jacobian

- **`MITBodyController`**：Body PD 力控
  - Body PD: Fz = -mg + Kp_z·(z - h_target) - Kd_z·vz, Fx = -Kp_vx·(vx_target - vx)
  - 力分配：Fz/n 均分到 stance 腿，roll/pitch moment 差动分配
  - Stance 脚跟随身体旋转：捕获身体系偏移 `Rᵀ·(p_foot - p_com)`，目标 = `com + R·offset`
    （而非锁死世界坐标 → 避免纯旋转时身体被迫平移）
  - 纯旋转模式 (vx=vy=0)：记录 settle 后锚点，Kp_px=20 温和抑制 CoM 漂移（~4.3cm/3s）
  - Kp_vy=200（侧向速度 PD 增强），Kp_vx=500
  - 重构为 `_compute_body_pd_wrench()` → `_apply_leg_impedance()` 两步
  - Stance: 阻抗控制追踪计算的目标 (target.z=0) + 前馈 GRF
  - Swing: 阻抗控制追踪规划的 hip 系轨迹
  - 启动时先 position settle (0.5s)，再切力矩控制

- **`SrbMpcSolver`** (`mpc_controller.py`)：SRB 凸 MPC 求解器
  - 纯数值求解，不依赖 MuJoCo
  - 单刚体 (SRB) 动力学，小角度线性化（忽略陀螺项）
  - 状态 x ∈ R¹²: [roll, pitch, yaw, px, py, pz, ωx, ωy, ωz, vx, vy, vz]
  - 控制 u ∈ R¹²: 4 腿 × 3D 世界系 GRF（ground→foot 约定）
  - 紧凑 QP 形式：消去状态，仅优化控制序列 U ∈ R¹²⁰（N=10）
  - 约束：线性化摩擦锥 (μ=0.6, 4 边金字塔) + fz∈[1,150] + 摆动腿零力
  - A_c² = 0（幂零）→ 离散化用闭式解，无需 expm
  - OSQP warm start + update (q/l/u)，典型求解 4-6ms
  - 权重设计：MPC 侧重高度/速度跟踪，阻抗层负责姿态稳定

- **`MPCMITBodyController(MITBodyController)`**：MPC + 阻抗控制
  - 每 16 步 (31 Hz) 运行一次 MPC 求解 → 缓存 GRF
  - 符号取反：MPC 输出 GRF (ground→foot) → 阻抗约定的 foot→ground
  - 阻抗层每步 (500 Hz) 执行：τ = Jᵀ·(F_mpc + Kp·Δx - Kd·v)
  - Stance 增益比纯 Body PD 模式更软 (Kp=[80,80,200], Kd=[5,5,10])
  - QP 失败 → 自动 fallback 到 `_compute_body_pd_wrench()`
  - 启动延迟 8 步后才首次 MPC 求解（前几步用 Body PD 稳过渡）
  - `mpc_stats` 属性返回求解时间/fallback 统计

**力控约定**：Body PD / 阻抗层使用「足端→地面」约定（Fz<0=下压），MPC 使用「地面→足端」约定（fz>0=支撑）。MPC 输出在传入阻抗层前取反。

## 模型说明

| 文件 | 类型 | 用途 |
|------|------|------|
| `go1.xml` | 浮基 MJCF | 原始模型，带 freejoint |
| `go1_fixed.xml` | 固定基座 | 自动生成，trunk welded |
| `scene.xml` | 浮基场景 | go1 + 地面 + 天空，`--gait --float` |
| `scene_fixed.xml` | 固定场景 | `--viewer` 单腿模式 |
| `scene_hifric.xml` | 高精度场景 | 增强接触力参数，力控用 |

关节命名规则：执行器 `FR_hip`，关节 `FR_hip_joint`。控制器 `_find_names()` 自动适配两种命名。

足端位置优先级：site (`FR`) → body (`FR_foot`) → 解析 FK 近似。

**MJCF 大腿侧向偏移**：hip abduction 关节和 thigh pitch 关节之间有一个 `[0, ±0.08, 0]` 的 body position 偏移（FR/RL 为负，FL/RR 为正）。解析 FK 忽略此偏移，需通过 `_moco_offset` 修正。修正量随 abduction 角度变化（R_abd @ [0, ±0.08, 0]），但正常行走时 abduction ≈ 0，常数修正可接受。

## 偏航控制架构（2026-06-15 更新）

详见 `docs/yaw_control_architecture.md`。关键发现：

- **Izz**: 躯干单独 0.017 → 全系统 M[wz,wz]=0.423（25×），`experiments/golden_izz_test.py` 验证
- **偏航方案**: 运动学落脚点偏移（物理可实现，73%/vyaw=0.2）+ 直连偏航力矩（仿真辅助，90%）。
  - 混合模式 `_direct_yaw_scale=0.3`（当前默认）：81% vyaw=0.2, 54% vx
  - 纯运动学 `_direct_yaw_scale=0.0`：73% vyaw=0.2, 50% vx
- **drift 分析**: 侧向漂移来自前进力与偏航力在摩擦锥内竞争，非 Trot 几何固有。
  原地自旋时 drift 从 0.028→0.009 m/s（3× 缩减）。`experiments/trace_mz_sign.py` Mz 符号链追踪。
- **步态影响**: Walk vyaw=0.2 达 95%，Pace 前进+偏航耦合最优
- **调试报告**: 仿真后自动生成 11 节诊断（5 层 Mz 链 + 偏心偏航检测），输出到 `output/`

## 常用命令

所有命令需先 `source venv/bin/activate`。

### 单腿轨迹跟踪

```bash
# 图形化模式（3D 交互窗口）
python3 -m src.main --viewer --leg FR --traj circle
python3 -m src.main --viewer --leg FR --traj lissajous

# 无头模式（批量仿真 + 输出分析图表）
python3 -m src.main --leg FR --traj circle
python3 -m src.main --urdf --leg FL --traj circle
```

### 步态仿真（位控）

```bash
# 固定基座步态（脚在空中画轨迹）
python3 -m src.main --gait --viewer --gait-type trot
python3 -m src.main --gait --gait-type walk --gait-cycles 3

# 浮基步态（身体自由 + stance 锚定）
python3 -m src.main --gait --float --viewer --gait-type trot \
    --step-length 0.08 --step-height 0.05
```

### 力控仿真（MIT 模式 — 真正的动力学）

```bash
# 浮基力控 — Body PD（推荐用于动力学调试）
python3 -m src.main --gait --float --force --viewer --gait-type trot

# 调整目标速度和步态参数
python3 -m src.main --gait --float --force --viewer --gait-type trot \
    --target-vx 0.5 --step-length 0.08 --step-height 0.05

# 无头力控
python3 -m src.main --gait --float --force --gait-type trot \
    --gait-cycles 10 --target-vx 0.3
```

### MPC + MIT 阻抗控制

```bash
# MPC 模式（图形化，推荐参数 — T=0.25s, L=0.22m）
python3 -m src.main --gait --float --force --mpc --viewer --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3

# MPC 模式（无头，前进+侧向）
python3 -m src.main --gait --float --force --mpc --gait-type trot \
    --gait-T 0.25 --step-length 0.22 \
    --target-vx 0.3 --target-vy 0.2 \
    --gait-cycles 8

# 对比 Body PD vs MPC（各自推荐参数）
python3 -m src.main --gait --float --force --gait-type trot \
    --gait-T 0.5 --step-length 0.10 --gait-cycles 5  # Body PD (需较长周期)
python3 -m src.main --gait --float --force --mpc --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --gait-cycles 8  # MPC (短周期大步长)
```

#### MPC 输出示例（优化参数 `--gait-T 0.25 --step-length 0.22`）

```
============================================================
  MyDog — MPC + MIT Impedance Gait [headless]
  Gait: trot, T_cycle=0.25s, duty=0.60
  Target vx=0.30, vy=0.00, vyaw=0.00
  Model: /home/scj/MyDog/model/scene.xml
============================================================

Model loaded: 19 qpos, 18 qvel, 12 actuators
Body mass: 12.9 kg, weight: 126 N
Settling body on ground (0.5s)...

Simulating 8.0 cycles (2.0s) at dt=0.002s → 1000 steps
Running simulation...
  Step 0/1000:    pos=[-0.010, 0.000, 0.265]  vel=[ 0.017, 0.000, 0.022]
  Step 500/1000:  pos=[ 0.098, 0.001, 0.279]  vel=[ 0.205, 0.010, 0.001]  MPC: 5.2ms
  Step 1000/1000: pos=[ 0.200, 0.003, 0.279]  vel=[ 0.207, 0.005,-0.005]  MPC: 5.0ms

MPC stats: avg solve 5.1ms, fallbacks 0/62 (0.0%)
Simulation complete.
```

**关键观察**：
- `pos[2]`（高度）稳定在 **0.279m**，接近目标 0.28m
- `vel[0]` ≈ 0.21 m/s — 达到目标 0.3 的 70%，比旧参数 (0.04) 提升 5x
- T_cycle 缩短到 0.25s（原 0.5s）是速度提升的关键因素
- Body PD 在 T=0.25 时不稳定，仅 MPC 可用

### Quintic + 静摩擦约束控制（当前推荐）

```bash
# 混合偏航（运动学落脚点 + 30% 直连，vyaw=0.2 达 81%）
python3 -m src.main --gait --float --force --quintic --viewer --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3 --target-vyaw 0.2

# 纯运动学偏航（物理可实现基准，vyaw=0.2 达 73%）
# 设置 _direct_yaw_scale=0.0 后运行同上

# 原地自旋（drift 诊断）
python3 -m src.main --gait --float --force --quintic --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --target-vx 0 --target-vyaw 0.2

# Walk 步态（偏航最强，vyaw=0.2 达 95%）
python3 -m src.main --gait --float --force --quintic --viewer --gait-type walk \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3 --target-vyaw 0.2

# 启用 Tier 1 参数自适应
python3 -m src.main --gait --float --force --quintic --adapt-params --viewer \
    --gait-type trot --gait-T 0.25 --step-length 0.08 --target-vx 0.3

# 自定义摩擦系数
python3 -m src.main --gait --float --force --quintic --viewer --gait-type trot \
    --mu-max 0.8 --target-vx 0.3
```

#### 输出示例

```
============================================================
  MyDog — Quintic + Friction Force Gait [headless]
  Gait: trot, T_cycle=0.25s, duty=0.60
  Target vx=0.30, vy=0.00, vyaw=0.00
  Friction μ_max=0.6, adapt_params=False
============================================================

  Step 500/1250: pos=[ 0.175, -0.008, 0.280] vel=[ 0.194, 0.008,-0.007] μ=0.72 α=0.0
  Step 1000/1250:pos=[ 0.366, -0.011, 0.280] vel=[ 0.194, 0.009,-0.007] μ=0.72 α=0.0

Final state: pos=[ 0.460, -0.012, 0.280] vel=[ 0.193, 0.010,-0.005]
Friction stats: μ_utilized=0.713, feasible=True, α=0.000
```

**关键指标**：
- `μ_utilized`: 摩擦利用率（0=无摩擦, 1=达 μ_max 极限）；实时反映系统离摩擦极限的距离
- `α`: 零空间参数 — 0 表示最小范数解在摩擦锥内（系统有余量）；非零表示需偏移才能满足约束
- `feasible`: 期望 wrench 是否在摩擦锥内可行

#### 三层架构

```
Tier 1 (~10 Hz, 可选): 步态参数自适应
  └─ 根据 μ_utilized 和速度误差调整 step_length

Tier 2 (按需): 五次多项式摆动轨迹 (QuinticFootTrajectoryPlanner)
  └─ C² 连续足端轨迹：零速度/零加速度 at touchdown & liftoff

Tier 3 (500 Hz): 静摩擦约束力分配 (FrictionForceDistributor)
  └─ 2 腿 trot: min ||f||² s.t. A·f = [Fx,Fy,Mz], |fxᵢ|,|fyᵢ| ≤ μ·fzᵢ
  └─ 1D 零空间解析解: f = f_p + α·f_null, α 由线性化摩擦锥边界确定
```

#### 对比 Body PD / MPC / Quintic+Friction / Momentum

| 指标 | Body PD | MPC | Quintic+Friction | Momentum |
|------|---------|-----|------------------|----------|
| vx 跟踪 (target=0.3) | ~0.06 (20%) | **0.20 (67%)** | 0.19 (63%) | ~0.17 (57%) |
| vy 跟踪 (target=0.2) | 不稳定 | ~0.09 (45%) | **0.11 (57%)** | ~0.10 (50%) |
| vyaw 跟踪 (target=0.5) | ~1% | ~1% | **~45% 均值（运动学）** | **~40% 均值（运动学）** |
| 身体稳定 (roll/pitch) | ±5° 振荡 | 0.6°/1.0° RMS | **0.2°/0.3° RMS** | 0.5°/1.0° RMS |
| 摆动平滑性 | C0 | C0 | **C² 连续** | **C² 连续** |
| 摩擦安全性 | 无保证 | QP 约束 | **显式约束 + μ_util** | **6D 求解 + 二分缩放** |
| 力分配方式 | 均分 + 启发式 | QP 优化 | **3D 零空间解** | **6D Newton-Euler** |
| Roll/Pitch 主动控制 | ✅ (新增) | ✅ (继承) | ✅ (新增) | ✅ (原有) |
| 偏航角度控制 | P (基础) | P (基础) | **P+I+直连** | **P+I** |
| 计算开销 | O(1) | 4-6ms QP | **O(1)** | **6×6 solve + 二分搜索** |
| 可解释性 | 低 | 低（13 权重） | **高（μ_util, α）** | **高（μ_max, cond）** |

**身体姿态 + 偏航控制** (`force_controller.py`):
- **Roll/Pitch**: 主动 PD 力矩通过差动 Fz 实现身体水平控制（之前显式设 Mx=My=0）
  - 增益: Kp_roll=30, Kd_roll=10, Kp_pitch=30, Kd_pitch=10 Nm/rad
  - 效果: roll/pitch 从 ±5° 振荡降至 0.2-1.0° RMS
- **Yaw**: P + I 角度控制器 + 角速度阻尼，增益按运动方向自适应
  - 前进时：小 P (5 Nm/rad) + 慢 I (2 Nm/rad·s)，维持航向
  - 侧向时：大 P (25 Nm/rad) + 快 I (40 Nm/rad·s)，对抗 COM 偏移力矩
  - 侧向辅助直连力矩 (`qfrc_applied`，绕过腿力传递链损耗)
- Stance 脚跟随身体旋转：捕获身体系偏移 `Rᵀ·(p_foot - p_com)`，目标 = `com + R·offset`
  （而非锁死世界坐标 → 纯旋转时脚绕 CoM 画弧，身体不被迫平移）
- CoM 位置锚点 PD (Kp=20 N/m)：仅在纯旋转模式 (vx=vy=0) 启用，抑制残余漂移至 ~4.3cm/3s
- **落脚点偏移钳制**：`step_delta` ±15cm, `lat_offset` ±5cm, `yaw_offset` ±45°，防止脚超出工作空间
- **COM 修正**：摩擦力分配器使用整机 COM 估计（前方 ~1cm），而非 trunk body 原点

#### 算法改进 (2026-06)

**1. Roll/Pitch 主动力矩控制** — `MITBodyController` 和 `QuinticFrictionController` 之前显式设 Mx=My=0，仅靠 stance 阻抗被动稳姿。现在通过差动 Fz 实现主动身体水平控制：
- 身体 roll/pitch RMS: **0.2-1.0°**（改进前 ±5° 振荡）
- 前进时 vy 漂移仅 ~2mm/s

**2. 五次 Z 轴足端轨迹** — `QuinticFootTrajectoryPlanner` 用两段五次多项式替代 `z = H·sin(πs)`，实现 C² 连续触地和离地（零速度/零加速度），触地冲击速度从 ~0.63 m/s 降至 0。

**3. 偏航角度 P + I 控制** — 原只有角速度 PD（`Kp_yaw * dwz`），无法消除稳态偏航误差。新增：
- 四元数微分的世界系偏航角计算（`qvel[5]` 不是世界系 wz）
- 偏航角度 P + 积分 I 控制器，增益按侧向/前进比例自适应（前进时小增益保持航向，侧向时大增益对抗 COM 偏移力矩）
- 落脚点偏移钳制（`lat_offset` ±5cm, `step_delta` ±15cm, `yaw_offset` ±45°）
- 纯侧向运动时辅助直连力矩通道（`qfrc_applied`，绕过腿力传递链损耗）
- 偏航误差 >9° 时自动降低侧向力释放摩擦预算

**4. MPC 控制代价正则化** — `R_diag` 从 `[1e-6, 1e-6, 1e-8]` 提升至 `[1e-3, 1e-3, 1e-4]`，消除力指令跳变，改善 QP Hessian 条件数（2e12 → 1e6）。

**5. OSQP Warm-Start** — 复用 OSQP 求解器对象而非每次新建，保持内部状态用于热启动。

**6. 速度跟踪指标采集** — `VelocityMetrics` 类记录每步 vx/vy/wz 误差、高度、姿态角，仿真结束时输出稳态汇总（均值/RMSE/跟踪百分比）。所有力控模式均支持。

**7. 力控参数重调** — Quintic+Friction 模式的增益和阻抗经系统调优：

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `K_kin_vx` | 2.0 | 1.0 | 落脚点偏移减半，避免腿过度前伸 |
| `K_kin_vy` | 0.6 | 0.4 | 侧向落脚点更温和 |
| `Kp_vx` | 300 | 500 | 更大的前向力（摩擦约束会自动限幅） |
| `Kp_vy` | 150 | 200 | 更大的侧向力 |
| `Kp_yaw` | 10 | 15 | 更强的偏航角速度阻尼 |
| `stance_Kp[xy]` | 200 | 100 | 更软的水平阻抗，让前馈力有效推脚 |
| `stance_Kd[xy]` | 10 | 5 | 减少阻尼对抗前馈力 |

**8. 摩擦力分配器 COM 修正** — `com_position` 改用整机 COM 估计（前方 ~1cm，下方 ~15cm），而非 trunk body 原点。原代码用 trunk body 位置当 COM，在侧向运动时产生 ~0.5 Nm 持续偏航力矩。

#### 速度跟踪最终结果

**Quintic+Friction 模式 (T=0.25, L=0.22, μ=0.6)**：

| 目标 | 实际 vx | 实际 vy | roll/pitch | 备注 |
|------|---------|---------|------------|------|
| vx=0.3, vy=0 | **0.190 (63%)** | −0.002 | 0.2°/0.3° RMS | 前进稳定，几乎无侧漂 |
| vx=0.5, vy=0 | 0.175 (35%) | +0.022 | 0.4°/0.5° RMS | 超最大速度 (~0.2)，摩擦饱和 |
| vx=0, vy=0.3 | 0.014 | **0.135 (45%)** | 0.8°/0.6° RMS | 纯侧向，~7° 偏航残余 |
| vx=0.3, vy=0.2 | 0.142 (47%) | **0.114 (57%)** | 0.7°/0.3° RMS | vy 较原始提升 68% |

**MPC 模式 (T=0.25, L=0.22)**：

| 目标 | 实际 vx | MPC solve | fallback | roll/pitch |
|------|---------|-----------|----------|------------|
| vx=0.3, vy=0 | **0.201 (67%)** | 4.4ms | 0% | 0.6°/1.0° |

**关键发现**：
- **vy 跟踪大幅改善** — 侧向运动从 43%→45%（纯侧向）和 34%→57%（组合），提升 25-68%
- **身体姿态极稳** — roll/pitch 从 ±5° 振荡降至 <1.0° RMS
- **MPC 求解稳定** — 0% fallback，正则化 + warm-start 有效
- **Quintic 前向速度 63% (0.19 m/s)** — 接近 MPC 的 67%，远优于 Body PD 的 20%
- **Trot 纯侧向有 ~7° 偏航残余** — COM 偏移 + 对角支撑力矩抵消的固有限制。推荐 Pace/Walk 步态
- **最大速度约 0.2 m/s** — 受关节阻尼（2 N·m·s/rad × 12）+ 接触阻尼限制。`vx=0.5` 时摩擦锥饱和，偏航修正被挤掉

#### 步态对比 — 不同运动方向需要不同步态！

**核心发现**：Trot 偏航从力控基本无效（1%）→ 运动学落脚点偏移提升至 **45-57%**。侧向运动仍建议换步态：

| 目标 | 指标 | Trot | Pace | Walk | 最佳 |
|------|------|------|------|------|------|
| vx=0.3 | vx | 0.207 (69%) | 0.216 (72%) | 0.190 (63%) | Pace |
| vy=0.2 | vy | 0.087 (43%) | 0.105 (52%) | **0.113 (56%)** | Walk |
| vyaw=0.5 | wz | 0.005 (1%) | 0.003 (1%) | **0.053 (11%)** | Walk |
| **vx+vy** | vy | 0.068 (34%) | **0.122 (61%)** | 0.110 (55%) | **Pace** |

**分析**：
- **Pace（同侧同步）**：FR+FL 同时着地、RR+RL 同时着地。侧向运动时同侧腿朝同一方向推，无力矩抵消。组合 vx+vy 时 vy 效率从 34%→61%（翻倍），vx 不受影响。
- **Walk（波形，3 足着地）**：纯 vy 效率最高（56%），vyaw 有 10x 提升（1%→11%）。但因着地腿多，前向推进力分散，vx 略低。
- **Trot（对角同步）**：前向运动最优（仅略逊 Pace），侧向有限制但偏航已通过运动学落脚点偏移解决（~50% 效率）。
- **Vyaw 运动学方案**：力控偏航在 trot 下有几何限制（对角线力矩相消），运动学落脚点偏移（K_kin_wz=0.20）是偏航主力通道。

**推荐策略**：
- 前进为主 → Trot 或 Pace
- 前进+侧向 → **Pace**（vy 效率翻倍）
- 纯侧向 → **Walk**
- 偏航 → 力控方案无效，运动学落脚点偏移已实现（K_kin_wz=0.20, ~50% trot 效率）

**速度瓶颈分析**：
- **接触阻尼**：MuJoCo 默认接触产生 ~142 N·s/m 等效阻尼，需要 ~30N 净推力维持 0.2 m/s
- **Jᵀ 传输效率**：MPC 力通过 Jacobian 传到身体，效率约 71%
- **T_cycle (步态周期) 是第一优先调优参数**：T=0.25s → vx=0.207 vs T=0.5s → vx=0.112
- **Gait type 是第二优先**：侧向运动换 Pace/Walk 可获 1.3-1.8x 提升
- **step_length 有辅助效果**：L=0.22m → vx=0.207 vs L=0.06m → vx=0.171 (在 T=0.25)
- **MPC Q 权重 (Q_vx, Q_vy) 影响很小**：瓶颈在接触物理和步态几何，不在优化
- **stance 阻抗 Kp/Kd 对速度无影响**：MPC 前馈力主导，阻抗仅维持接触

**偏航（vyaw）运动学方案**：
- 力控偏航在 trot 下有根本性几何限制（对角力矩相消）→ 1% 效率
- 运动学落脚点偏移：swing 时按 wz 误差偏移落地 x 位置，左脚落后 / 右脚超前 → CCW
- K_kin_wz=0.20 时 trot 偏航跟踪 ~45-57%（相比力控 1%，提升 50x）
- 仍有步态周期性振荡（离散落脚点调整的固有延迟），std≈0.24 rad/s
- Momentum 控制器的 6D Mz 通道可与运动学偏航互补，提供微调

#### Body PD vs MPC 对比

| 指标 | Body PD (`--force`) | MPC (`--force --mpc`) |
|------|---------------------|------------------------|
| 推荐 T_cycle | 0.5s（短周期不稳定） | 0.25s（稳定） |
| 身体高度 | 0.280m | 0.279m |
| 高度波动 | ±0.02m | ±0.002m（更平稳） |
| 前向速度 (vx=0.3) | ~0.06 m/s | ~0.21 m/s |
| 侧向速度 (vy=0.2) | 不稳定 | ~0.09 m/s |
| 力分配 | QP-free 均分 | QP 优化 |
| 求解开销 | 无 | 4-6ms OSQP |
| 预测能力 | 无（纯反馈） | 0.3s 时域滚动优化 |

### 常用参数

| 参数 | 默认值 | 推荐 (MPC) | 推荐 (Body PD) | 说明 |
|------|--------|-----------|---------------|------|
| `--gait-type` | trot | 见下方 | trot | trot(前向) / **pace(前向+侧向)** / walk(纯侧向/偏航) |
| `--gait-T` | 0.5 | **0.25** | 0.5 | 步态周期 (s)，MPC 建议短周期 |
| `--gait-duty` | 0.6 | 0.6 | 0.6 | stance 占空比 (0~1) |
| `--step-length` | 0.06 | **0.22** | 0.10 | 步长 (m)，MPC 建议大步长 |
| `--step-height` | 0.04 | 0.04 | 0.04 | 抬脚高度 (m) |
| `--target-vx` | 0.3 | 0.3 | 0.3 | 目标前进速度 (m/s) |
| `--target-vy` | 0.0 | 0.0~0.2 | 0.0 | 目标侧向速度 (m/s) |
| `--target-vyaw` | 0.0 | 0.0 | 0.0 | 目标偏航角速度 (rad/s) |
| `--mpc` | - | ✓ | - | 启用 SRB 凸 MPC（需 --force --float） |
| `--quintic` | - | ✓ | - | 启用五次轨迹+摩擦约束力控 |
| `--momentum` | - | - | - | 启用 6×6 Newton-Euler 力分配（需 --quintic） |
| `--mu-max` | 0.6 | 0.6 | 0.6 | 最大静摩擦系数（高目标速度时建议 0.9） |
| `--adapt-params` | - | - | - | 启用 Tier 1 步态参数自适应（需 --quintic） |
| `--dt` | 0.002 | 0.002 | 0.002 | 仿真步长 (s) |
| `--viewer` | - | - | - | 开启 3D 交互窗口 |
| `--no-plot` | - | - | - | 跳过图表输出 |

**步态选择指南**：
- `--target-vx 0.3`（纯前进）→ `--gait-type trot` 或 `pace`
- `--target-vx 0.3 --target-vy 0.2`（前进+侧向）→ `--gait-type pace`（vy 效率翻倍）
- `--target-vx 0 --target-vy 0.3`（纯侧向）→ `--gait-type pace` 或 `walk`（trot 有 ~7° 偏航残余）
- `--target-vyaw 0.5`（偏航）→ 运动学落脚点偏移 (K_kin_wz=2.0, ~50% trot 效率)
- `--target-vx 0.5`（高速）→ 需配合 `--mu-max 0.9`，否则摩擦锥饱和

### Viewer 操作

Space 暂停/恢复 | 滚轮缩放 | 右键拖拽旋转 | Ctrl+右键拖拽平移

WSL 中需要 Windows 端 X Server（VcXsrv），设置 `export DISPLAY=:0`。

### RL 决策层（UniLab，在 `~/UniLab` 目录执行）

```bash
# 预训练模型演示
uv run demo dance                 # G1 动作跟踪
uv run demo wallflip              # G1 翻墙

# 训练 locomotion 策略
uv run train --algo ppo --task go1_joystick_flat --sim mujoco

# 评估/回放
uv run eval --algo ppo --task go1_joystick_flat --sim mujoco --load-run -1
```

## 调试注意事项

1. **IK 失败排查**：检查 `_moco_offset` 是否已设置。单腿模式需在 `sim.forward()` 后手动计算偏移。偏移 = MuJoCo FK 足端(hip 系) - 解析 FK 足端(hip 系)，在 home 角度下计算。

2. **浮基滑动**：位控浮基模式 stance 脚滑动是预期行为（无推进力）。stance 锚定缓解此问题，但身体不会前进。需要力控模式获得真正的推进力。

3. **SIGFPE**：WSL2 上 MuJoCo 的 `mj_forward` / `mj_step` 偶发浮点异常。`controller.py` 的 `_forward_safe` 和 `force_controller.py` 的 `_step_safe` 通过屏蔽 FP 异常 + signal handler 来安全处理（`force_controller.py` 顶部有独立副本）。解析 IK 减少了 `mj_forward` 调用次数（仅最终验证一次）。

4. **力控/MPC 稳定性**：
   - **Body PD**：如果狗抖动或飞起，降低 `force_controller.py` 中的增益（`Kp_z`, `Kp_vx`, `_stance_Kp`, `_swing_Kp`）。力矩限幅 ±23 Nm。
   - **MPC**：如果狗飞起或 QP 失败率高，检查 `mpc_controller.py` 中的权重（`Q_diag`, `R_diag`）和 `MPCMITBodyController._run_mpc()` 中的 `max_ff` 钳位（当前 80N/分量）。
   - MPC 前 8 步用 Body PD 过渡，避免冷启动大信号冲击。

5. **MPC 力约定**：MPC 求解器使用「地面→足端」约定（fz>0=支撑力），阻抗层使用「足端→地面」约定（Fz<0=下压力）。`MPCMITBodyController._run_mpc()` 在缓存前自动取反。如果速度跟踪方向反了，检查此处。

6. **场景文件选择**：
   - 单腿 + viewer → `scene_fixed.xml`（固定基座 + 地面）
   - 步态 + 浮基 → `scene.xml`（自由关节 + 地面）
   - 力控 / MPC → 同上，也可用 `scene_hifric.xml`（更好的接触参数）

7. **MPC 速度跟踪偏低**：如果前向速度远低于目标（如 0.04 vs 0.3 m/s）：
   - **第一优先**：缩短 `--gait-T`（T_cycle），这是影响最大的参数（T=0.25→vx=0.21 vs T=0.5→vx=0.11）
   - **第二优先**：增大 `--step-length`（L=0.22→vx=0.21 vs L=0.06→vx=0.17 at T=0.25）
   - MPC Q 权重 (Q_vx, Q_vy)、stance Kp/Kd、R_diag 对稳态速度影响极小（瓶颈在接触物理）
   - `scene_hifric.xml` 摩擦更高 (1.5 vs 0.8) 反而更慢（vx=0.009），不要用于提速
   - Body PD 在 T<0.5 时不稳定，短周期只能用 MPC
