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
  - 进行中：力控参数调优、步态稳定性
- **上层（UniLab）**：用 RL 做智能决策
  - 部署于本地 `/home/scj/UniLab`
  - 输出高层动作（目标速度、航向角、步态类型）
  - 下发到下层传统控制器执行

## 技术栈

- Python 3.10
- **物理引擎**: MuJoCo 3.9.0（MyDog 仿真）, MuJoCoUni 3.8.0（UniLab 依赖）
- **数值计算**: NumPy, SciPy, matplotlib
- **模型格式**: MJCF（原生） + URDF（自动转换）
- **底层控制**: 解析 FK 的 Gauss-Newton IK + 相位调度步态规划 + MIT 力控（torque via JᵀF）
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
│   │   ├── force_controller.py  # MITBodyController — MIT 力控（力矩）
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

### MIT 力控 (`force_controller.py`)

- **`LegTorqueController`**：力矩控制原语
  - `apply_force(F_world)`: τ = Jᵀ·F，限幅 ±20 Nm
  - `apply_impedance(target, Kp, Kd, ff)`: F = ff + Kp·(xᵈ-x) - Kd·(v-vᵈ)，限力 80N
  - `jacobian_world(q)`: hip_rot @ J_hip(q) → 3×3 世界 Jacobian
  
- **`MITBodyController`**：整机力控
  - Body PD: Fz = -mg + Kp_z·(z - h_target) - Kd_z·vz, Fx = -Kp_vx·(vx_target - vx)
  - 力分配：Fz/n 均分到 stance 腿，roll/pitch moment 差动分配
  - Stance: 阻抗控制追踪当前脚位 (target.z=0) + 前馈 GRF
  - Swing: 阻抗控制追踪规划的 hip 系轨迹
  - 启动时先 position settle (0.5s)，再切力矩控制

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
# 浮基力控（推荐用于动力学调试）
python3 -m src.main --gait --float --force --viewer --gait-type trot

# 调整目标速度和步态参数
python3 -m src.main --gait --float --force --viewer --gait-type trot \
    --target-vx 0.5 --step-length 0.08 --step-height 0.05

# 无头力控
python3 -m src.main --gait --float --force --gait-type trot \
    --gait-cycles 10 --target-vx 0.3
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gait-type` | trot | trot / walk / pace / bound |
| `--gait-T` | 0.5 | 步态周期 (s) |
| `--gait-duty` | 0.6 | stance 占空比 (0~1) |
| `--step-length` | 0.06 | 步长 (m) |
| `--step-height` | 0.04 | 抬脚高度 (m) |
| `--target-vx` | 0.3 | 力控目标前进速度 (m/s) |
| `--dt` | 0.002 | 仿真步长 (s) |
| `--viewer` | - | 开启 3D 交互窗口 |
| `--no-plot` | - | 跳过图表输出 |

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

3. **SIGFPE**：WSL2 上 MuJoCo 的 `mj_forward` 偶发浮点异常。`controller.py` 的 `_forward_safe` 通过屏蔽 FP 异常 + signal handler 来安全处理。解析 IK 减少了 `mj_forward` 调用次数（仅最终验证一次）。

4. **力控稳定性**：如果狗抖动或飞起，降低 `force_controller.py` 中的增益（`Kp_z`, `Kp_vx`, `_stance_Kp`, `_swing_Kp`）。力矩限幅 ±20 Nm，接触力限幅 80N。

5. **场景文件选择**：
   - 单腿 + viewer → `scene_fixed.xml`（固定基座 + 地面）
   - 步态 + 浮基 → `scene.xml`（自由关节 + 地面）
   - 力控 → 同上，也可用 `scene_hifric.xml`（更好的接触参数）
