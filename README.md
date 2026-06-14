# MyDog — 四足机器狗分层控制系统

基于 MuJoCo 物理引擎的宇树 Go1 四足机器人控制仿真平台。采用**分层控制架构**：
底层用传统方法做 locomotion 原语（IK + 步态规划 + 力控 + MPC），上层用强化学习做智能决策（UniLab）。

```
┌─────────────────────────────────────────┐
│     上层：RL 决策（UniLab）               │
│     速度选择 · 方向规划 · 步态切换        │
│     地形适应 · 行为策略                   │
├─────────────────────────────────────────┤
│     下层：传统控制（MyDog 本体）           │
│     四足 IK · 步态规划 · 足尖轨迹生成      │
│     力控/位控执行 · 姿态稳定 · SRB MPC    │
├─────────────────────────────────────────┤
│     物理引擎：MuJoCo（Go1 模型）           │
└─────────────────────────────────────────┘
```

## 项目状态

| 模块 | 状态 |
|------|------|
| 单腿足尖轨迹跟踪（circle/line/sine/lissajous） | ✅ 完成 |
| 四足步态规划与协调（trot/walk/pace/bound） | ✅ 完成 |
| 浮基位置控制（stance 锚定） | ✅ 完成 |
| MIT 风格力控 — Body PD + 阻抗（torque via JᵀF） | ✅ 完成 |
| SRB 凸 MPC + MIT 阻抗控制（OSQP, N=10） | ✅ 完成 |
| Quintic+Friction 力控（C² 轨迹 + 摩擦约束力分配） | ✅ 完成 |
| Momentum 6×6 Newton-Euler 力分配 | ✅ 完成 |
| vx/vy 速度跟踪（MPC 67%, Quintic 63%） | ✅ 完成 |
| Roll/Pitch 主动姿态控制（0.2° RMS） | ✅ 完成 |
| 偏航角度 P+I 控制（自适应增益） | ✅ 完成 |
| RL 决策层集成（UniLab adapter） | 🚧 待实现 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10 |
| 物理引擎 | MuJoCo 3.9.0 |
| 数值计算 | NumPy, SciPy, matplotlib |
| 优化求解 | OSQP — 运筹分裂二次规划求解器（MPC QP） |
| 模型格式 | MJCF（原生）+ URDF（自动转换） |
| 控制方法 | 解析 FK Gauss-Newton IK / 相位步态调度 / MIT 力控（JᵀF）/ SRB 凸 MPC |
| 上层决策 | [UniLab](https://github.com/unilabsim/UniLab) — PPO/SAC/TD3/APPO |
| 包管理 | pip（MyDog）+ uv（UniLab） |

## 项目结构

```
MyDog/
├── model/
│   ├── go1.xml                 # Go1 MJCF 模型（浮基，带 freejoint）
│   ├── go1_fixed.xml           # 固定基座变体（自动生成）
│   ├── scene.xml               # 浮基场景 = go1 + 地面 + 天空 + 灯光
│   ├── scene_fixed.xml         # 固定基座场景
│   ├── scene_hifric.xml        # 高精度接触力场景（力控用）
│   ├── assets/                 # STL 网格文件
│   └── unitree/                # Unitree 官方 URDF + DAE 网格
├── src/
│   ├── simulator.py            # MuJoCoSim 仿真封装
│   ├── kinematics.py           # LegKinematics：解析 FK / Jacobian
│   ├── trajectory.py           # 笛卡尔轨迹生成器（circle/line/sine/lissajous）
│   ├── controller.py           # IKFootController：单腿 IK 位置控制
│   ├── gait.py                 # 步态调度 + 足尖轨迹规划
│   ├── body_controller.py      # BodyController：四腿协调（位控）
│   ├── force_controller.py     # MITBodyController + MPCMITBodyController + QuinticFrictionController + MomentumController
│   ├── friction_force.py       # FrictionForceDistributor — 静摩擦约束力分配
│   ├── mpc_controller.py       # SrbMpcSolver：SRB 凸 MPC 求解器 (OSQP, N=10)
│   ├── urdf_loader.py          # URDF → MJCF 转换器
│   ├── main.py                 # 主入口（单腿/步态/力控/MPC 模式）
│   └── rl/                     # RL 集成层（待实现）
│       ├── adapter.py          # UniLab 接口适配
│       └── env.py              # 基于 MyDog 仿真器的 Gym 环境
├── output/                     # 仿真结果图表
├── CLAUDE.md                   # Claude Code 项目指南
└── requirements.txt
```

外部依赖（不放入本仓库）：

```
~/UniLab/                        # UniLab 框架（独立 git 管理，可跟踪上游更新）
```

## 快速开始

### 环境安装

```bash
# 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 四种仿真模式

#### 1. 单腿轨迹跟踪

控制一条腿的足尖跟踪笛卡尔轨迹，用于验证运动学和 IK。

```bash
source venv/bin/activate

# 图形化模式（3D 交互窗口）
python3 -m src.main --viewer --leg FR --traj circle
python3 -m src.main --viewer --leg FR --traj lissajous

# 无头模式（批量仿真 + 输出分析图表）
python3 -m src.main --leg FR --traj circle
```

#### 2. 步态仿真（位置控制）

四腿协调步态，支持固定基座和浮基。

```bash
# 固定基座 — 脚在空中画轨迹，身体不动
python3 -m src.main --gait --viewer --gait-type trot

# 浮基 — 身体自由 + stance 锚定防滑动
python3 -m src.main --gait --float --viewer --gait-type trot \
    --step-length 0.08 --step-height 0.05

# 无头步态仿真
python3 -m src.main --gait --gait-type walk --gait-cycles 5
```

#### 3. 力控仿真（Body PD — 浮基动力学）

MIT Cheetah 风格控制器：Body PD → 力分配 → 阻抗控制 → τ = JᵀF 力矩执行。

```bash
# 浮基力控（推荐参数：T=0.5s, L=0.10m — Body PD 需较长周期）
python3 -m src.main --gait --float --force --viewer --gait-type trot \
    --gait-T 0.5 --step-length 0.10 --target-vx 0.3

# 无头力控
python3 -m src.main --gait --float --force --gait-type trot \
    --gait-T 0.5 --step-length 0.10 --target-vx 0.3 --gait-cycles 8
```

#### 4. MPC 仿真（SRB 凸优化 + MIT 阻抗 — 推荐）

MPC 每 31Hz 求解 10 步时域 QP，输出最优 GRF 馈入阻抗控制器。
Body PD 作为 fallback（QP 失败时自动切换）。

```bash
# MPC 纯前进（Trot 或 Pace 最佳）
python3 -m src.main --gait --float --force --mpc --viewer --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3

# MPC 前进+侧向（Pace 最佳 — vy 效率翻倍！）
python3 -m src.main --gait --float --force --mpc --viewer --gait-type pace \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3 --target-vy 0.2

# MPC 纯侧向（Walk 最佳）
python3 -m src.main --gait --float --force --mpc --gait-type walk \
    --gait-T 0.25 --step-length 0.22 --target-vx 0 --target-vy 0.2 \
    --gait-cycles 8

# MPC 无头 — 对比三种步态
python3 -m src.main --gait --float --force --mpc --gait-type pace \
    --gait-T 0.25 --step-length 0.22 \
    --target-vx 0.3 --target-vy 0.2 --gait-cycles 8
```

#### 5. Quintic+Friction 仿真（C² 轨迹 + 摩擦约束力分配 — 轻量替代）

五次多项式摆动（零速度/零加速度触地）+ 静摩擦约束力分配（解析零空间解，O(1) 计算）。
支持三层架构：Tier 1 参数自适应 + Tier 2 五次轨迹 + Tier 3 摩擦约束。

```bash
# 纯前进
python3 -m src.main --gait --float --force --quintic --viewer --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3

# 纯侧向（Pace 推荐，Trot 有 ~7° 偏航残余）
python3 -m src.main --gait --float --force --quintic --viewer --gait-type pace \
    --gait-T 0.25 --step-length 0.22 --target-vy 0.3 --target-vx 0.0

# 前进+侧向（Pace 最佳 — vy 效率翻倍）
python3 -m src.main --gait --float --force --quintic --viewer --gait-type pace \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.3 --target-vy 0.2

# 高速模式（需提高摩擦系数）
python3 -m src.main --gait --float --force --quintic --viewer --gait-type trot \
    --gait-T 0.25 --step-length 0.22 --target-vx 0.5 --mu-max 0.9

# Momentum 模式（6×6 Newton-Euler，全 6D 力分配）
python3 -m src.main --gait --float --force --quintic --momentum --viewer \
    --gait-type trot --gait-T 0.25 --step-length 0.22 --target-vx 0.3
```

### 命令行参数

| 参数 | 默认值 | 推荐 (MPC) | 推荐 (Body PD) | 说明 |
|------|--------|-----------|---------------|------|
| `--leg` | FR | - | - | 控制腿（FR/FL/RR/RL，单腿模式） |
| `--traj` | circle | - | - | 轨迹类型（circle/line/sine/lissajous） |
| `--gait` | - | ✓ | ✓ | 启用步态模式 |
| `--gait-type` | trot | trot | trot | 步态类型（trot/walk/pace/bound） |
| `--float` | - | ✓ | ✓ | 浮基模型（身体可自由运动） |
| `--force` | - | ✓ | ✓ | MIT 力控模式（需配合 `--float`） |
| `--mpc` | - | ✓ | - | SRB 凸 MPC 模式（需配合 `--force --float`） |
| `--quintic` | - | ✓ | - | 五次轨迹 + 摩擦约束力控（需 `--force --float`） |
| `--momentum` | - | - | - | 6×6 Newton-Euler 力分配（需 `--quintic`） |
| `--mu-max` | 0.6 | 0.6 | 0.6 | 最大静摩擦系数（高速建议 0.9） |
| `--adapt-params` | - | - | - | 启用 Tier 1 步态参数自适应（需 `--quintic`） |
| `--gait-T` | 0.5 | **0.25** | 0.5 | 步态周期（秒），MPC 短周期更快 |
| `--gait-duty` | 0.6 | 0.6 | 0.6 | Stance 占空比（0~1） |
| `--step-length` | 0.06 | **0.22** | 0.10 | 步长（米），MPC 大步长更快 |
| `--step-height` | 0.04 | 0.04 | 0.04 | 抬脚高度（米） |
| `--target-vx` | 0.3 | 0.3 | 0.3 | 目标前进速度（m/s） |
| `--target-vy` | 0.0 | 0.0~0.2 | 0.0 | 目标侧向速度（m/s） |
| `--target-vyaw` | 0.0 | 0.0 | 0.0 | 目标偏航角速度（rad/s），效果微弱 |
| `--gait-cycles` | 5 | 8 | 5 | 无头模式仿真周期数 |
| `--dt` | 0.002 | 0.002 | 0.002 | 仿真步长（秒） |
| `--viewer` | - | - | - | 开启 3D 交互窗口 |
| `--no-plot` | - | - | - | 跳过图表输出 |
| `--urdf` | - | - | - | 使用 URDF 输入（自动转换 MJCF） |
| `--model` | - | - | - | 自定义 MJCF 模型路径 |

### Viewer 操作

| 操作 | 按键 |
|------|------|
| 暂停/恢复 | `Space` |
| 缩放 | `滚轮` |
| 旋转视角 | `右键拖拽` |
| 平移视角 | `Ctrl + 右键拖拽` |

> WSL 用户需先在 Windows 端启动 X Server（如 VcXsrv），并 `export DISPLAY=:0`。

## 控制架构

### 四种模式对比

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
│  Quintic+Friction (--force --quintic --float)   │
│  QuinticFrictionController ← MITBodyController  │
│  ├─ Tier 2: 五次多项式摆动 (C² 触地/离地)         │
│  ├─ Tier 3: 静摩擦约束力分配 (零空间解析解)       │
│  ├─ Roll/Pitch 主动 PD + 偏航 P+I 角度控制       │
│  ├─ Stance: τ = Jᵀ·(F_friction + Kp·Δx - Kd·v)  │
│  └─ Swing:  τ = Jᵀ·(Kp·Δx - Kd·v)               │
├─────────────────────────────────────────────────┤
│  Force Mode (--force --float)                   │
│  MITBodyController                              │
│  ├─ Body PD: 高度/姿态/速度 → 身体 wrench        │
│  ├─ Roll/Pitch 主动 PD + 力分配                  │
│  ├─ Stance: 阻抗控制 + 前馈力                    │
│  ├─ Swing:  阻抗控制跟踪轨迹                      │
│  └─ τ = Jᵀ·F  (qfrc_applied)                    │
├─────────────────────────────────────────────────┤
│  Position Mode (--gait [--float])               │
│  BodyController                                 │
│  ├─ GaitScheduler: 相位调度                      │
│  ├─ FootTrajectoryPlanner: hip 系轨迹            │
│  ├─ Fixed: target_hip → target_world            │
│  ├─ Float: stance 锚定 + swing hip 轨迹          │
│  └─ IKFootController.solve_ik → position ctrl   │
├─────────────────────────────────────────────────┤
│  Single Leg (--leg FR --traj circle)            │
│  IKFootController 直接控制单腿                    │
└─────────────────────────────────────────────────┘
```

### 力控详解 (`force_controller.py`)

四种控制器，共享 MIT 阻抗框架：

**MITBodyController** — Body PD + 力分配 + 阻抗：

1. **Body PD**：根据期望高度/速度与实际状态的误差计算身体 wrench
   - 垂直力：`Fz = -mg + Kp_z·(z - h_target) - Kd_z·vz`
   - 前进力：`Fx = -Kp_vx·(vx_target - vx)`
   - 侧向力：`Fy = -Kp_vy·(vy_target - vy)`
   - 姿态力矩：`Mx/My = -Kp·angle - Kd·angvel`（主动水平控制）
   - 偏航力矩：`Mz = Kp_yaw·(vyaw_target - ωz) + P·yaw_error + I·∫yaw_error`

2. **力分配**：将身体 wrench 分配到各 stance 腿
   - Fz/n 均分 + roll/pitch 差动分配
   - yaw moment 转为左右腿差动 Fx 或 fy 不平衡

3. **阻抗控制**：`F = ff + Kp·(xᵈ-x) - Kd·v`
   - Stance: target 随身体旋转（`com + R·offset`），前馈 GRF
   - Swing: target 跟踪规划的 hip 系轨迹

4. **力矩执行**：`τ = Jᵀ·F`（MuJoCo `mj_jacSite`），写入 `qfrc_applied`

**MPCMITBodyController** — SRB 凸 MPC + 阻抗（最高性能）：

- 每 16 步 (31 Hz) 运行一次 MPC，缓存 GRF
- 阻抗层每步 (500 Hz) 执行：`τ = Jᵀ·(F_mpc + Kp·Δx - Kd·v)`
- Stance 增益更软 (Kp=[80,80,200], Kd=[5,5,10]) — MPC 前馈承担主要负荷
- QP 失败 → 自动 fallback 到 Body PD
- OSQP 对象复用 + warm-start（减少求解时间和噪声）

**QuinticFrictionController** — C² 轨迹 + 摩擦约束力分配（轻量高性能）：

- Tier 2: 五次多项式摆动（X/Y/Z 全轴 C² 连续，零冲击触地）
- Tier 3: 静摩擦约束力分配（2 腿零空间解析解，O(1) 计算）
- 偏航 P+I 角度控制，增益按运动方向自适应
- 落脚点偏移钳制（防止超出腿部工作空间）
- 纯侧向时辅助直连力矩通道

| 增益 | Body PD 值 | MPC 值 | Quintic 值 | 作用 |
|------|-----------|--------|------------|------|
| `Kp_z` / `Kd_z` | 200 / 40 | 200 / 40 | 200 / 40 | 高度 PD |
| `Kp_vx` | 500 | 500 | **500** | 前进速度跟踪 |
| `Kp_vy` | 200 | 200 | **200** | 侧向速度跟踪 |
| `Kp_roll` / `Kp_pitch` | **30 / 30** | 30 / 30 | 30 / 30 | 主动姿态控制（新增） |
| `Kp_yaw` | 30 | 30 | **15** | 偏航角速度阻尼 |
| Stance Kp / Kd | [150,150,500] / [10,10,20] | [80,80,200] / [5,5,10] | **[100,100,500] / [5,5,20]** | 支撑腿阻抗 |
| Swing Kp / Kd | [400,400,400] / [15,15,15] | [400,400,400] / [15,15,15] | [400,400,400] / [15,15,15] | 摆动腿阻抗 |

### MPC 详解 (`mpc_controller.py`)

SRB（单刚体）凸 MPC，纯数值求解（不依赖 MuJoCo）：

- **状态** x ∈ R¹²: `[roll, pitch, yaw, px, py, pz, ωx, ωy, ωz, vx, vy, vz]`
- **控制** u ∈ R¹²: 4 腿 × 3D 世界系 GRF（ground→foot 约定）
- **时域**：N=10 步, dt=30ms → 0.3s 预测时域
- **约束**：线性化摩擦锥 (μ=0.6, 4 边金字塔) + fz∈[1,150] + 摆动腿零力
- **QP 规模**：120 变量, 紧凑形式（消去状态，仅优化控制序列）
- **求解器**：OSQP warm start + update (q/l/u)，典型 4-6ms

| MPC 权重 | 值 | 说明 |
|----------|-----|------|
| Q[9] vx | 20000 | 前向速度跟踪（主导） |
| Q[10] vy | 8000 | 侧向速度跟踪 |
| Q[5] pz | 500 | 高度跟踪 |
| Q[8] ωz | 50 | 偏航速率阻尼 |
| R[0:2] fx,fy | **1e-3** | 水平力惩罚（正则化，原 1e-6） |
| R[2] fz | **1e-4** | 垂直力惩罚（正则化，原 1e-8） |

**力约定**：Body PD / 阻抗层使用「足端→地面」约定（Fz<0=下压），MPC 使用「地面→足端」约定（fz>0=支撑）。MPC 输出在传入阻抗层前自动取反。

### 速度跟踪性能

**Trot 步态 (T=0.25, L=0.22)**：

| 模式 | 目标 | 实际 vx | 实际 vy | roll/pitch | 备注 |
|------|------|---------|---------|------------|------|
| MPC | vx=0.3 | **0.201 (67%)** | +0.002 | 0.6°/1.0° | 4.4ms, 0% fallback |
| Quintic | vx=0.3 | 0.190 (63%) | −0.002 | 0.2°/0.3° | 前进稳定，几乎无侧漂 |
| Quintic | vx=0.5 | 0.175 (35%) | +0.024 | 0.4°/0.5° | 超最大速度 (~0.2)，摩擦饱和 |
| Quintic | vy=0.3 | 0.014 | **0.135 (45%)** | 0.8°/0.6° | 纯侧向，~7° 偏航残余 |
| Quintic | vx+vy=0.3+0.2 | 0.142 (47%) | **0.114 (57%)** | 0.7°/0.3° | vy 较原始 Trot 提升 68% |

**步态对比 — 不同方向需要不同步态！**

| 目标 | 指标 | Trot | Pace | Walk | 最佳 |
|------|------|------|------|------|------|
| vx=0.3 | vx | 0.190 (63%) | 0.216 (72%) | 0.190 (63%) | Pace |
| vy=0.2 | vy | 0.087 (43%) | 0.105 (52%) | **0.113 (56%)** | Walk |
| vyaw=0.5 | wz | 0.005 (1%) | 0.003 (1%) | **0.053 (11%)** | Walk |
| vx+vy | vy | 0.068 (34%) | **0.122 (61%)** | 0.110 (55%) | **Pace** |

**分析**：
- **Pace（同侧同步）**：组合 vx+vy 时 vy 效率翻倍（34%→61%）。同侧腿朝同一方向推，无力矩抵消。
- **Walk（波形，3 足着地）**：纯侧向最优（56%），偏航有提升但有限（11%）。
- **Trot（对角同步）**：前进最优，纯侧向有 ~7° 偏航残余（COM 偏移 + 对角力矩相消的固有限制）。
- **偏航（vyaw）**：力控方案有根本性几何限制。运动学落脚点偏移 (K_kin_wz=2.0) 是主力通道 (~50% 效率)。

**推荐策略**：
- 前进为主 → `--gait-type trot` 或 `pace`
- 前进+侧向 → `--gait-type pace`（vy 翻倍）
- 纯侧向 → `--gait-type pace` 或 `walk`（trot 偏航不可接受）
- 高速 (>0.3) → 配合 `--mu-max 0.9`

**调优优先级**：T_cycle >> Gait type > step_length >> 力控增益 >> MPC Q 权重（几乎无影响）

### 步态系统 (`gait.py`)

- **GaitScheduler**：相位 = `(offset[leg] + t/T) % 1.0`，φ < duty_factor → stance
- **GaitType**：TROT（对角同步）、WALK（波形）、PACE（同侧同步）、BOUND（前后同步）
- **FootTrajectoryPlanner**：
  - Stance：脚在 hip 系中后移 `x = nx + L/2 - L·s`
  - Swing：脚前移 + 正弦抬升 `z = nz + H·sin(π·s)`
  - 浮基：XY 用 hip 系轨迹，Z 用世界系（stance=0, swing=H·sin）
  - 预留运动学偏航接口（`target_vyaw` 属性，gain=0 禁用）

### IK 控制器 (`controller.py`)

- **解析 FK + MuJoCo 偏移修正**：LegKinematics 解析 FK 忽略 MJCF 大腿侧向偏移 `[0, ±0.08, 0]`，通过 `_moco_offset` 修正
- **Gauss-Newton 阻尼最小二乘**：`Δq = Jᵀ(JJᵀ + λ²I)⁻¹e`，阻尼 λ=0.01
- **SIGFPE 保护**：WSL2 上 MuJoCo `mj_forward` 偶发浮点异常，`_forward_safe` 安全包装
- **关节限位**：1% margin 的 `_clamp_q` 防止越界

## 模型说明

| 文件 | 类型 | 用途 |
|------|------|------|
| `go1.xml` | 浮基 MJCF | 原始模型，带 freejoint |
| `go1_fixed.xml` | 固定基座 | 自动生成，trunk welded |
| `scene.xml` | 浮基场景 | go1 + 地面 + 天空，`--gait --float` |
| `scene_fixed.xml` | 固定场景 | `--viewer` 单腿模式 |
| `scene_hifric.xml` | 高精度场景 | 增强接触力参数（摩擦 1.5），力控用 |

> `go1_fixed.xml` 和 `go1_from_urdf.xml` 由程序自动生成，首次运行时创建。

## RL 集成（基于 UniLab）

### 设计思路

RL 策略不直接输出关节角，而是输出**高层动作命令**，由传统控制器执行：

```
RL 策略输出: [vx, vy, ω_z, height, gait_mode]
       │
       ▼
adapter.py  ──→  步态规划  ──→  四足足尖轨迹  ──→  IK/力控/MPC  ──→  MuJoCo 仿真
```

RL 只需学习"往哪走、走多快、用什么步态"，底层执行的稳定性和可解释性由传统控制保证。

### UniLab 可用命令

```bash
cd ~/UniLab

# 预训练模型演示
uv run demo dance          # G1 动作跟踪
uv run demo wallflip       # G1 翻墙

# 训练
uv run train --algo ppo --task go1_joystick_flat --sim mujoco

# 评估
uv run eval --algo ppo --task go1_joystick_flat --sim mujoco --load-run -1
```

### 后续融合路线

1. 在 UniLab 中基于 Go1 训练 locomotion 策略（已有 `go1_joystick_flat` 任务）
2. 在 MyDog 的 `src/rl/adapter.py` 中加载 checkpoint，提取策略网络
3. 策略输出映射为传统控制器的速度/步态命令
4. MyDog 仿真器执行，采集数据回流训练（闭环）

## 核心设计决策

- **分层架构**：RL 做决策，传统控制做执行，各司其职。不是 RL 端到端替代传统控制。
- **解析 FK + MuJoCo 偏移**：避免 MuJoCo FK 的 SIGFPE 问题，同时保持与仿真器运动学一致。
- **Stance 锚定**：浮基位控模式 stance 脚锚定世界坐标，防止滑动。
- **力控优先**：浮基动力学仿真的正确做法是力控（JᵀF），而非位置控制。
- **MPC + 阻抗**：MPC 优化 GRF 前馈，阻抗层维持接触稳定，分工明确。OSQP warm-start 复用降低求解噪声。
- **C² 足端轨迹**：五次多项式摆动消除触地冲击（~0.63 m/s → 0），减少身体振荡。
- **主动姿态控制**：Roll/Pitch PD 将身体振荡从 ±5° 降至 <1.0° RMS。偏航 P+I 控制消除稳态角度误差。
- **增益自适应**：偏航修正强度按侧向/前进比例自动调整（侧向时高增益对抗 COM 偏移，前进时低增益维持航向）。
- **UniLab 外部依赖**：不 fork 不复制，通过 adapter 层调用公开 API，跟踪上游更新。
- **双模型输入**：同时支持 MJCF 和 URDF，自动转换命名约定。

## 调试注意事项

1. **IK 失败**：检查 `_moco_offset` 是否已设置（单腿模式需在 `sim.forward()` 后手动计算）
2. **浮基滑动**：位控模式 stance 滑动是预期行为。stance 锚定缓解，力控模式真正解决
3. **SIGFPE 崩溃**：解析 IK 已大幅减少 `mj_forward` 调用，仍有问题见 `_forward_safe`
4. **力控不稳**：降低增益 `Kp_z`/`Kp_vx`/`_stance_Kp`，力矩限幅 ±35 Nm
5. **MPC 速度偏低**：**第一优先缩短 `--gait-T`**（T_cycle），**第二优先增大 `--step-length`**。Q 权重和 stance Kp/Kd 影响极小，瓶颈在接触物理
6. **场景选择**：`scene_hifric.xml` 摩擦更高 (1.5) 反而导致速度更低 (vx=0.009)，不推荐用于提速
7. **Body PD 限速**：T<0.5 时 Body PD 不稳定，短周期必须用 MPC

## 依赖

```
mujoco>=3.9.0
numpy>=1.21.0
scipy>=1.8.0
matplotlib>=3.5.0
osqp>=0.6.0
```

## 许可证

本项目仅用于教育和研究目的。Go1 机器人模型版权归宇树科技（Unitree Robotics）所有。
