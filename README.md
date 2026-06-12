# MyDog — 四足机器狗分层控制系统

基于 MuJoCo 物理引擎的宇树 Go1 四足机器人控制仿真平台。采用**分层控制架构**：
底层用传统方法做 locomotion 原语（IK + 步态规划 + 力控），上层用强化学习做智能决策（UniLab）。

```
┌─────────────────────────────────────────┐
│     上层：RL 决策（UniLab）               │
│     速度选择 · 方向规划 · 步态切换        │
│     地形适应 · 行为策略                   │
├─────────────────────────────────────────┤
│     下层：传统控制（MyDog 本体）           │
│     四足 IK · 步态规划 · 足尖轨迹生成      │
│     力控/位控执行 · 姿态稳定               │
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
| MIT 风格力控（torque via JᵀF） | ✅ 完成 |
| 偏航/侧向阻尼 | ✅ 完成 |
| MPC 模型预测控制 | 🚧 计划中 |
| RL 决策层集成（UniLab adapter） | 🚧 待实现 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10 |
| 物理引擎 | MuJoCo 3.9.0 |
| 数值计算 | NumPy, SciPy, matplotlib |
| 模型格式 | MJCF（原生）+ URDF（自动转换） |
| 控制方法 | 解析 FK Gauss-Newton IK / 相位步态调度 / MIT 力控（JᵀF） |
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
│   ├── force_controller.py     # MITBodyController：MIT 力控（力矩）
│   ├── urdf_loader.py          # URDF → MJCF 转换器
│   ├── main.py                 # 主入口（单腿/步态/力控模式）
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

### 三种仿真模式

#### 1. 单腿轨迹跟踪

控制一条腿的足尖跟踪笛卡尔轨迹，用于验证运动学和 IK。

```bash
source venv/bin/activate

# 图形化模式（3D 交互窗口）
python3 -m src.main --viewer --leg FR --traj circle
python3 -m src.main --viewer --leg FR --traj lissajous

# 无头模式（批量仿真 + 输出分析图表）
python3 -m src.main --leg FR --traj circle
python3 -m src.main --leg FL --traj sine
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

#### 3. 力控仿真（MIT 模式 — 真正的动力学）

MIT Cheetah 风格控制器：Body PD → 力分配 → 阻抗控制 →  τ = JᵀF 力矩执行。

```bash
# 浮基力控（推荐用于动力学调试）
python3 -m src.main --gait --float --force --viewer --gait-type trot

# 调整目标速度和步态
python3 -m src.main --gait --float --force --viewer \
    --gait-type trot --target-vx 0.5 --step-length 0.08
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--leg` | FR | 控制腿（FR/FL/RR/RL，单腿模式） |
| `--traj` | circle | 轨迹类型（circle/line/sine/lissajous） |
| `--gait` | - | 启用步态模式 |
| `--gait-type` | trot | 步态类型（trot/walk/pace/bound） |
| `--float` | - | 浮基模型（身体可自由运动） |
| `--force` | - | MIT 力控模式（需配合 `--float`） |
| `--gait-T` | 0.5 | 步态周期（秒） |
| `--gait-duty` | 0.6 | Stance 占空比（0~1） |
| `--step-length` | 0.06 | 步长（米） |
| `--step-height` | 0.04 | 抬脚高度（米） |
| `--target-vx` | 0.3 | 力控目标前进速度（m/s） |
| `--gait-cycles` | 5 | 无头模式仿真周期数 |
| `--dt` | 0.002 | 仿真步长（秒） |
| `--viewer` | - | 开启 3D 交互窗口 |
| `--no-plot` | - | 跳过图表输出 |
| `--urdf` | - | 使用 URDF 输入（自动转换 MJCF） |
| `--model` | - | 自定义 MJCF 模型路径 |

### Viewer 操作

| 操作 | 按键 |
|------|------|
| 暂停/恢复 | `Space` |
| 缩放 | `滚轮` |
| 旋转视角 | `右键拖拽` |
| 平移视角 | `Ctrl + 右键拖拽` |

> WSL 用户需先在 Windows 端启动 X Server（如 VcXsrv），并 `export DISPLAY=:0`。

## 控制架构

### 三种模式对比

```
┌─────────────────────────────────────────────────┐
│  Force Mode (--force --float)                   │
│  MITBodyController                              │
│  ├─ Body PD: 高度/姿态/速度 → 身体 wrench        │
│  ├─ 力分配: wrench → stance 腿 GRF               │
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

MIT Cheetah 风格控制器，是浮基动力学仿真的核心：

1. **Body PD**：根据期望高度/速度与实际状态的误差计算身体 wrench
   - 垂直力：`Fz = -mg + Kp_z·(z - h_target) - Kd_z·vz`
   - 前进力：`Fx = -Kp_vx·(vx_target - vx)`
   - 偏航力矩：`Mz = -Kd_yaw·ωz`（抑制偏航漂移）
   - 侧向阻尼：`Fy = -Kd_vy·vy`

2. **力分配**：将身体 wrench 分配到各 stance 腿
   - Fz/n 均分承担体重，roll/pitch moment 差动分配，yaw moment 转为左右腿差动前向力

3. **阻抗控制**：`F = ff + Kp·(xᵈ-x) - Kd·v`
   - Stance: target.z=0（锚定在地面），前馈 GRF 支撑体重
   - Swing: target 跟踪规划的 hip 系轨迹

4. **力矩执行**：`τ = Jᵀ·F` 写入 `qfrc_applied`，绕过 position actuator

| 增益 | 值 | 作用 |
|------|-----|------|
| `Kp_z` / `Kd_z` | 200 / 40 | 高度 PD |
| `Kp_vx` | 100 | 前进速度跟踪 |
| `Kd_yaw` | 30 | 偏航角速度阻尼 |
| `Kd_vy` | 50 | 侧向速度阻尼 |
| Stance Kp / Kd | [150,150,500] / [10,10,20] | 支撑腿阻抗 |
| Swing Kp / Kd | [400,400,400] / [15,15,15] | 摆动腿阻抗 |

### 步态系统 (`gait.py`)

- **GaitScheduler**：相位 = `(offset[leg] + t/T) % 1.0`，φ < duty_factor → stance
- **GaitType**：TROT（对角同步）、WALK（波形）、PACE（同侧同步）、BOUND（前后同步）
- **FootTrajectoryPlanner**：
  - Stance：脚在 hip 系中后移 `x = nx + L/2 - L·s`
  - Swing：脚前移 + 正弦抬升 `z = nz + H·sin(π·s)`
  - 浮基：XY 用 hip 系轨迹，Z 用世界系（stance=0, swing=H·sin）

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
| `scene_hifric.xml` | 高精度场景 | 增强接触力参数，力控用 |

> `go1_fixed.xml` 和 `go1_from_urdf.xml` 由程序自动生成，首次运行时创建。

## RL 集成（基于 UniLab）

### 设计思路

RL 策略不直接输出关节角，而是输出**高层动作命令**，由传统控制器执行：

```
RL 策略输出: [vx, vy, ω_z, height, gait_mode]
       │
       ▼
adapter.py  ──→  步态规划  ──→  四足足尖轨迹  ──→  IK/力控  ──→  MuJoCo 仿真
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
- **UniLab 外部依赖**：不 fork 不复制，通过 adapter 层调用公开 API，跟踪上游更新。
- **双模型输入**：同时支持 MJCF 和 URDF，自动转换命名约定。

## 调试注意事项

1. **IK 失败**：检查 `_moco_offset` 是否已设置（单腿模式需在 `sim.forward()` 后手动计算）
2. **浮基滑动**：位控模式 stance 滑动是预期行为。stance 锚定缓解，力控模式真正解决
3. **SIGFPE 崩溃**：解析 IK 已大幅减少 `mj_forward` 调用，仍有问题见 `_forward_safe`
4. **力控不稳定**：降低增益 `Kp_z`/`Kp_vx`/`_stance_Kp`，力矩限幅 ±20Nm，接触力限幅 80N
5. **偏航漂移**：力控模式已内置 `Kd_yaw` 阻尼，如仍漂移可适当增大

## 依赖

```
mujoco>=3.9.0
numpy>=1.21.0
scipy>=1.8.0
matplotlib>=3.5.0
```

## 许可证

本项目仅用于教育和研究目的。Go1 机器人模型版权归宇树科技（Unitree Robotics）所有。
