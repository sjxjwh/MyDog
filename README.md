# MyDog — 四足机器狗分层控制系统

基于 MuJoCo 物理引擎的宇树 Go1 四足机器人控制仿真平台。采用**分层控制架构**：
底层用传统方法做 locomotion 原语（IK + 步态规划），上层用强化学习做智能决策（UniLab）。

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

## 项目目标

- **下层（当前阶段 ✅）**：单腿足尖轨迹跟踪，验证 FK/IK 与运动学建模
- **下层（下一步）**：扩展为四条腿协同步态（trot/walk），机身姿态稳定
- **上层（UniLab 已部署）**：RL 训练后输出高层决策（目标速度/航向/步态），下发到传统控制器执行
- **最终目标**：传统控制的稳定性 + RL 的自适应性，形成可解释且鲁棒的层级控制

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10 |
| 物理引擎 | MuJoCo 3.9.0 |
| 数值计算 | NumPy, SciPy, matplotlib |
| 模型格式 | MJCF（原生）+ URDF（自动转换） |
| 底层控制 | Gauss-Newton IK（基于 MuJoCo FK）+ 步态规划 |
| 上层决策 | [UniLab](https://github.com/unilabsim/UniLab) — 异构 CPU/GPU RL 运行时，PPO/SAC/TD3/APPO |
| RL 后端 | MuJoCoUni / MotrixSim（通过 UniLab 适配器） |

## 项目结构

```
MyDog/
├── model/
│   ├── go1.xml                 # Go1 MJCF 模型（mujoco_menagerie）
│   ├── go1_fixed.xml           # 固定基座变体（自动生成）
│   ├── go1_from_urdf.xml       # URDF 转换的 MJCF（自动生成）
│   ├── scene.xml               # 场景定义
│   ├── scene_fixed.xml         # 固定基座场景（地面、天空盒、灯光）
│   ├── assets/                 # STL 网格文件
│   └── unitree/                # Unitree 官方 URDF + DAE 网格
├── src/
│   ├── simulator.py            # MuJoCoSim 仿真封装 + 固定基座生成
│   ├── kinematics.py           # LegKinematics：FK / Jacobian / IK
│   ├── trajectory.py           # 笛卡尔轨迹生成器
│   ├── controller.py           # IKFootController：数值 IK 位置控制
│   ├── urdf_loader.py          # URDF → MJCF 转换器
│   ├── main.py                 # 主入口
│   └── rl/                     # RL 集成层（待实现）
│       ├── adapter.py          # UniLab 接口适配：RL 决策 → 控制器命令
│       └── env.py              # 可选：基于 MyDog 仿真器的 Gym 环境
├── output/                     # 仿真结果图表
└── requirements.txt

外部依赖（不放入本仓库）：
~/UniLab/                        # UniLab 框架（独立 git 管理，可跟踪上游更新）
```

> UniLab 作为外部项目独立维护，MyDog 仅通过 `src/rl/adapter.py` 调用其公开 API。详见 [CLAUDE.md](CLAUDE.md)。

## 快速开始

### 环境

```bash
# MyDog 虚拟环境
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

UniLab 使用独立的 `uv` 环境：

```bash
cd ~/UniLab
uv sync --extra motrix
```

### 传统控制仿真

#### 图形化模式（3D 实时窗口）

```bash
python3 -m src.main --viewer
python3 -m src.main --viewer --leg FR --traj circle
python3 -m src.main --viewer --leg FR --traj lissajous
```

> WSL 用户需先在 Windows 端启动 X Server（如 VcXsrv），并 `export DISPLAY=:0`。

**窗口操作**：`Space` 暂停 · `滚轮` 缩放 · `右键拖拽` 旋转 · `Ctrl+右键` 平移

#### 无头模式（批量仿真 + 分析图表）

```bash
python3 -m src.main                          # 默认：FR 腿画圆
python3 -m src.main --leg FL --traj sine     # FL 腿正弦
python3 -m src.main --dt 0.001 --no-plot     # 1ms 步长，跳过绘图
```

输出 4 张 PNG（位于 `output/`）：3D 轨迹对比、位置-时间、误差-时间、关节角度。

## 传统控制模块

### 运动学 (`kinematics.py`)

3-DOF 腿部运动学，基于旋量理论：

- **FK**：关节角 → 足端位置（髋关节坐标系）
- **Jacobian**：有限差分 3×3 矩阵
- **IK**：Gauss-Newton 阻尼最小二乘 `Δq = Jᵀ(JJᵀ + λ²I)⁻¹e`

Go1 腿参数：大腿 0.213m，小腿 0.213m。关节轴：abduction(X), hip_pitch(Y), knee_pitch(Y)。

### 控制器 (`controller.py`)

`IKFootController` 使用 MuJoCo FK 做 IK（每步调用 `sim.set_qpos` + `sim.forward`，有限差分算 Jacobian），确保控制解与仿真器运动学一致。支持 MJCF 和 URDF 两种命名约定的自动适配。

### 轨迹生成 (`trajectory.py`)

| 类型 | 参数 |
|------|------|
| `LinearTrajectory` | 起点、终点、时长 |
| `CircleTrajectory` | 圆心、半径、旋转轴、时长 |
| `SinusoidalTrajectory` | 中心、振幅、频率 |
| `LissajousTrajectory` | 中心、振幅、频率比 |

统一接口：`evaluate(t)` 和 `sample(dt)`。

## RL 集成（基于 UniLab）

### 设计思路

RL 策略不直接输出关节角，而是输出**高层动作命令**，由传统控制器执行：

```
RL 策略输出: [vx, vy, ω_z, height, gait_mode]
       │
       ▼
adapter.py  ──→  步态规划  ──→  四足足尖轨迹  ──→  IK 解关节角  ──→  MuJoCo 仿真
```

这样 RL 只需学习"往哪走、走多快、用什么步态"，底层执行的稳定性和可解释性由传统控制保证。

### UniLab 可用命令

```bash
cd ~/UniLab

# 预训练模型演示
uv run demo dance          # G1 动作跟踪
uv run demo wallflip       # G1 翻墙
uv run demo locomani       # Go2 运动操作

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
- **MuJoCo FK 驱动 IK**：控制器用仿真器 FK 算 Jacobian，而非解析式，消除运动学偏差。
- **固定基座（当前）→ 浮动基座（未来）**：先验证腿部运动学，再放开机身自由度做全身控制。
- **UniLab 外部依赖**：不 fork 不复制，通过 adapter 层调用公开 API，跟踪上游更新。
- **双模型输入**：同时支持 MJCF 和 URDF，自动转换命名约定。

## 依赖

```
mujoco>=3.9.0
numpy>=1.21.0
scipy>=1.8.0
matplotlib>=3.5.0
```

## 许可证

本项目仅用于教育和研究目的。Go1 机器人模型版权归宇树科技（Unitree Robotics）所有。
