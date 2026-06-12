# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

分层控制架构，在 MuJoCo 中对宇树 Go1 四足机器人实现 locomotion 控制与智能决策：

```
┌─────────────────────────────────────────┐
│     上层：RL 决策（UniLab）               │
│     速度选择、方向规划、步态切换、         │
│     地形适应、行为策略                    │
├─────────────────────────────────────────┤
│     下层：传统控制（MyDog 本体）           │
│     四足 IK、步态规划、足尖轨迹生成、      │
│     力控/位控执行、姿态稳定                │
├─────────────────────────────────────────┤
│     物理引擎：MuJoCo（Go1 模型）           │
└─────────────────────────────────────────┘
```

- **下层（MyDog 自身）**：用传统控制方法做整机的 locomotion 原语
  - 已完成：单腿足尖轨迹跟踪 ✅，四足步态规划与控制（trot/walk/pace/bound）✅
  - 已完成：浮基动力学仿真（MIT 力控 + 位控 stance 锚定）✅
  - 已完成：SRB 凸 MPC + MIT 阻抗控制 ✅
  - 进行中：MPC 速度跟踪调优、RL 集成
- **上层（UniLab）**：用 RL 做智能决策
  - 部署于本地 `/home/scj/UniLab`
  - 输出高层动作（目标速度、航向角、步态类型）
  - 下发到下层传统控制器执行

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
│   │   ├── force_controller.py  # MITBodyController + MPCMITBodyController
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
  - 重构为 `_compute_body_pd_wrench()` → `_apply_leg_impedance()` 两步
  - Stance: 阻抗控制追踪当前脚位 (target.z=0) + 前馈 GRF
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

#### 速度跟踪调优结果

**T_cycle + step_length 优化后，Trot 步态 (T=0.25, L=0.22)**：

| 目标 | 实际 vx | 实际 vy | 实际 wz | 效率 |
|------|---------|---------|---------|------|
| vx=0.3, vy=0 | 0.207 | -0.000 | -0.000 | vx=69% |
| vx=0, vy=0.2 | 0.052 | 0.087 | +0.055 | vy=44% |
| vx=0, vyaw=0.5 | 0.047 | -0.004 | **0.005** | vyaw=**1%** |
| vx=0.3, vy=0.2 | 0.207 | 0.068 | +0.043 | vx=69%, vy=34% |

#### 步态对比 — 不同运动方向需要不同步态！

**核心发现**：Trot 的对角支撑腿在侧向/偏航时力矩相消，切换步态可大幅改善：

| 目标 | 指标 | Trot | Pace | Walk | 最佳 |
|------|------|------|------|------|------|
| vx=0.3 | vx | 0.207 (69%) | 0.216 (72%) | 0.190 (63%) | Pace |
| vy=0.2 | vy | 0.087 (43%) | 0.105 (52%) | **0.113 (56%)** | Walk |
| vyaw=0.5 | wz | 0.005 (1%) | 0.003 (1%) | **0.053 (11%)** | Walk |
| **vx+vy** | vy | 0.068 (34%) | **0.122 (61%)** | 0.110 (55%) | **Pace** |

**分析**：
- **Pace（同侧同步）**：FR+FL 同时着地、RR+RL 同时着地。侧向运动时同侧腿朝同一方向推，无力矩抵消。组合 vx+vy 时 vy 效率从 34%→61%（翻倍），vx 不受影响。
- **Walk（波形，3 足着地）**：纯 vy 效率最高（56%），vyaw 有 10x 提升（1%→11%）。但因着地腿多，前向推进力分散，vx 略低。
- **Trot（对角同步）**：前向运动最优（仅略逊 Pace），但侧向/偏航被对角力矩相消严重限制。
- **Vyaw 仍然偏低**：即使最佳步态（Walk）也仅有 11% 效率。纯力控偏航在 MuJoCo 接触模型下有根本性局限。

**推荐策略**：
- 前进为主 → Trot 或 Pace
- 前进+侧向 → **Pace**（vy 效率翻倍）
- 纯侧向 → **Walk**
- 偏航 → 需要运动学方案（落脚点偏移），力控无法解决

**速度瓶颈分析**：
- **接触阻尼**：MuJoCo 默认接触产生 ~142 N·s/m 等效阻尼，需要 ~30N 净推力维持 0.2 m/s
- **Jᵀ 传输效率**：MPC 力通过 Jacobian 传到身体，效率约 71%
- **T_cycle (步态周期) 是第一优先调优参数**：T=0.25s → vx=0.207 vs T=0.5s → vx=0.112
- **Gait type 是第二优先**：侧向运动换 Pace/Walk 可获 1.3-1.8x 提升
- **step_length 有辅助效果**：L=0.22m → vx=0.207 vs L=0.06m → vx=0.171 (在 T=0.25)
- **MPC Q 权重 (Q_vx, Q_vy) 影响很小**：瓶颈在接触物理和步态几何，不在优化
- **stance 阻抗 Kp/Kd 对速度无影响**：MPC 前馈力主导，阻抗仅维持接触

**偏航（vyaw）局限性**：
- SRB MPC 力差动偏航效率：Trot 1%, Pace 1%, Walk 11%
- 即使 Walk 步态 10x 提升，仍然太低（目标 0.5 → 实际 0.053 rad/s）
- 运动学偏航（落脚点偏移）机制已预留（`gait.py` target_vyaw，gain=0），待后续开发
- 可能的方向：步态相位偏移 + 落脚点旋转的组合方案

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
| `--target-vyaw` | 0.0 | 0.0 | 0.0 | 目标偏航角速度 (rad/s)，效果微弱 |

**步态选择指南**（MPC 模式）：
- `--target-vx 0.3`（纯前进）→ `--gait-type trot` 或 `pace`
- `--target-vx 0.3 --target-vy 0.2`（前进+侧向）→ `--gait-type pace`（vy 效率翻倍）
- `--target-vx 0 --target-vy 0.2`（纯侧向）→ `--gait-type walk`
- `--target-vyaw 0.5`（偏航）→ 力控无法解决，需运动学方案
| `--mpc` | - | ✓ | - | 启用 SRB 凸 MPC（需 --force --float） |
| `--dt` | 0.002 | 0.002 | 0.002 | 仿真步长 (s) |
| `--viewer` | - | - | - | 开启 3D 交互窗口 |
| `--no-plot` | - | - | - | 跳过图表输出 |

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
