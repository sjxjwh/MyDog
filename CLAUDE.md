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
  - 当前阶段：单腿足尖轨迹跟踪（验证 FK/IK/运动学建模）✅
  - 目标：扩展为四条腿的步态规划与控制（trot/walk/gallop 步态，机身姿态稳定）
- **上层（UniLab）**：用 RL 做智能决策
  - 部署于本地 `/home/scj/UniLab`
  - 输出高层动作（目标速度、航向角、步态类型）
  - 下发到下层传统控制器执行

## 技术栈

- Python 3.10
- **物理引擎**: MuJoCo 3.9.0（MyDog 仿真）, MuJoCoUni 3.8.0（UniLab 依赖）
- **数值计算**: NumPy, SciPy, matplotlib
- **模型格式**: MJCF（原生） + URDF（自动转换）
- **底层控制**: 基于 MuJoCo FK 的数值 Gauss-Newton IK + 步态 CPG/轨迹规划
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

UniLab 是外部依赖，**不放入 MyDog 仓库**。MyDog 通过 `src/rl/` 集成层调用 UniLab。

```
~/                           # 用户 home 目录
├── MyDog/                   # 本项目（传统控制 + RL 集成层）
│   ├── model/               #   Go1 模型文件
│   ├── src/
│   │   ├── simulator.py     #   MuJoCo 仿真封装
│   │   ├── kinematics.py    #   运动学（FK / Jacobian）
│   │   ├── trajectory.py    #   笛卡尔轨迹生成
│   │   ├── controller.py    #   IK 控制器（位置/力控）
│   │   ├── urdf_loader.py   #   URDF → MJCF
│   │   ├── rl/              #   ⭐ RL 集成层（待实现）
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py   #   UniLab 接口适配：RL 决策 → 控制器命令
│   │   │   └── env.py       #   可选：基于 MyDog 仿真器的 Gym 环境
│   │   └── main.py          #   入口
│   └── output/
│
└── UniLab/                  # 外部项目（不放入 MyDog，独立 git 管理）
    ├── src/unilab/
    │   ├── algos/           #   RL 算法
    │   ├── envs/            #   Gym 环境（可迁移到 src/rl/env.py）
    │   ├── training/        #   训练管线
    │   └── ...
    └── conf/                #   Hydra 配置
```

## 关键设计

### 传统控制层
- **控制器使用 MuJoCo FK 做 IK**（`controller.py:solve_ik`）—— 每次迭代通过 `sim.set_qpos` + `sim.forward` 获取足端位置，用有限差分算 Jacobian，阻尼最小二乘求解。
- **当前支持单腿控制**：通过 IK 跟踪笛卡尔轨迹（circle/line/sine/lissajous），未来扩展为四条腿协同步态。
- **步态规划方向**：CPG（中枢模式发生器）或基于时间的相位调度，生成各腿的足尖轨迹，再由 IK 解算关节角。

### 两层接口设计（待实现）
- **RL 动作空间**：高层决策输出（如 body 目标线速度 vx/vy、航向角速率 ω、机身高度、步态模式），不直接输出关节角。
- **传统控制器接收**：解析 RL 输出的高层命令 → 生成四足步态 + 足尖轨迹 → IK 解算关节角 → 发送到 MuJoCo 仿真器。
- **RL 观测空间**：包含本体感知（IMU、关节状态、足端接触）+ 任务相关（目标速度、地形高度采样）。

### RL 集成层设计（`src/rl/`，待实现）

```
UniLab RL 策略 (PPO/SAC)
        │
        ▼  输出: [vx, vy, ω, height, gait_mode]
┌───────────────────┐
│  src/rl/adapter.py │  ← RL 集成层（MyDog 内唯一感知 UniLab 的模块）
│  UniLabAdapter     │
└───────────────────┘
        │  调用 MyDog 传统控制器
        ▼  输入: 高层命令 → 步态规划 → 足尖轨迹 → IK 关节角
┌───────────────────┐
│  src/controller.py │  ← 传统控制层（不依赖 UniLab）
│  src/kinematics.py │
│  src/trajectory.py │
└───────────────────┘
        │
        ▼  MuJoCo 仿真
```

**adapter.py 职责**：
1. 加载 UniLab 训练好的 RL checkpoint（通过 UniLab 的 eval API 或直接加载 Torch 模型）
2. 从仿真器收集观测（本体感知 + 任务上下文），封装为 RL 策略输入
3. 将 RL 策略输出的高层动作映射为传统控制器命令
4. 不修改 UniLab 源码，仅通过公开 API 调用

**为什么不迁移 UniLab**：
- UniLab 有独立的上游仓库（`github.com/unilabsim/UniLab`），需跟踪更新
- 框架本身 ~几百 MB，放入 MyDog 会让仓库臃肿
- 集成只需 `sys.path` 或在 `requirements.txt` 中做可编辑安装引用即可

### 模型输入
- **两种模型输入方式**：MJCF 直接加载（`--model model/go1.xml`）、URDF 自动转换（`--urdf`）
- **关节命名**：MJCF 中执行器名为 `FR_hip`，关节名为 `FR_hip_joint`；URDF 转换后均为 `FR_hip_joint`。控制器 `_find_names()` 自动适配。
- **足端位置**：优先用 site（`FR`），其次用 body（`FR_foot`），最后用解析 FK 近似。
- **渲染**：viewer 模式加载 `model/scene_fixed.xml`（地面、天空盒、灯光），材质从深灰提亮为 `(0.5,0.5,0.55)`。

## 两种运行模式

- **图形化模式**（`--viewer`）：打开 MuJoCo 原生 3D 交互窗口，实时观看机器狗足尖跟踪轨迹。
  - 操作：Space 暂停/恢复，滚轮缩放，右键拖拽旋转，Ctrl+右键拖拽平移。
  - 适用：调试轨迹、观察运动学、演示效果。
- **无头模式**（默认）：批量仿真 + 自动生成 `output/` 下的 4 张 PNG 分析图表（3D 轨迹对比、位置/时间、误差/时间、关节角度）。
  - 适用：定量分析、CI/CD、远程服务器。
  - 注意：WSL 中图形化需要 Windows 端有 X Server（如 VcXsrv），或使用 `export DISPLAY=:0` 指向运行的 X Server。

## 常用命令

### 传统控制（MyDog，在项目根目录执行）
```bash
# 图形化模式（3D 交互窗口）
python3 -m src.main --viewer
python3 -m src.main --viewer --leg FR --traj circle
python3 -m src.main --viewer --leg FR --traj lissajous

# 无头模式（批量仿真 + 输出分析图表）
python3 -m src.main
python3 -m src.main --leg FR --traj circle
python3 -m src.main --urdf --leg FL --traj circle
python3 -m src.main --dt 0.001 --no-plot
```

### RL 决策层（UniLab，在 `~/UniLab` 目录执行）
```bash
# 预训练模型演示
uv run demo dance                 # G1 动作跟踪
uv run demo wallflip              # G1 翻墙

# 训练 locomotion 策略
uv run train --algo ppo --task go1_joystick_flat --sim mujoco

# 评估/回放
uv run eval --algo ppo --task go1_joystick_flat --sim mujoco --load-run -1

# Notebook 交互
uv run jupyter notebook notebook/unilab_walkthrough_ppo_go1_joystick_mujoco.ipynb
```
