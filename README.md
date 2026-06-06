# MyDog — 四足机器狗控制系统

基于 MuJoCo 物理引擎的宇树 Go1 四足机器人控制仿真平台，融合**传统控制**与**强化学习**两条技术路线，实现从底层运动学到高层决策的完整控制栈。

## 项目目标

本项目旨在构建一个四足机器狗的完整控制框架，涵盖两个层次：

| 层次 | 方法 | 目标 |
|------|------|------|
| **底层运动控制** | 传统控制（IK + 轨迹规划） | 单足/多足足尖精确轨迹跟踪，实现稳定、可解释的运动基元 |
| **高层决策控制** | 强化学习（RL） | 在复杂地形中自主决策步态、落足点与运动策略，实现鲁棒导航 |

最终目标是将传统控制的稳定基元与强化学习的自适应决策相结合，形成一个既能精确执行动作、又能灵活应对环境的层级控制架构。

## 技术栈

- **语言**: Python 3.10+
- **物理引擎**: MuJoCo 3.9.0
- **数值计算**: NumPy, SciPy
- **可视化**: Matplotlib + MuJoCo 原生 Viewer
- **机器人模型**: 宇树 Unitree Go1（MJCF 原生 + URDF 自动转换）
- **IK 求解**: Gauss-Newton 阻尼最小二乘法（基于 MuJoCo FK 的有限差分 Jacobian）
- **RL 框架**（规划中）: 待定（候选：Isaac Gym / Brax / Gymnasium + MuJoCo）

## 项目架构

```
MyDog/
├── model/                        # 机器人模型
│   ├── go1.xml                   # Go1 MJCF 模型（mujoco_menagerie）
│   ├── go1_fixed.xml             # 固定基座变体（自动生成）
│   ├── go1_from_urdf.xml         # URDF 转换的 MJCF（自动生成）
│   ├── scene.xml                 # 场景定义
│   ├── scene_fixed.xml           # 固定基座场景（含地面、天空盒、灯光）
│   ├── assets/                   # STL 网格文件
│   └── unitree/                  # Unitree 官方 URDF + DAE 网格
│       └── meshes/
├── src/
│   ├── __init__.py               # 公开 API
│   ├── simulator.py              # MuJoCoSim 仿真封装 + 固定基座生成
│   ├── kinematics.py             # LegKinematics：解析 FK / Jacobian / IK
│   ├── trajectory.py             # 笛卡尔轨迹生成器
│   ├── controller.py             # IKFootController：基于 MuJoCo FK 的数值 IK 位置控制
│   ├── urdf_loader.py            # URDF → MJCF 转换器
│   └── main.py                   # 主入口
├── output/                       # 仿真结果图表（自动生成）
├── requirements.txt
├── CLAUDE.md                     # Claude Code 项目指引
└── README.md
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
python3 -m pip install -r requirements.txt
```

### 运行仿真

#### 图形化模式（实时 3D 窗口）

```bash
# FR 腿画圆，交互式 3D 窗口
python3 -m src.main --viewer

# 指定腿和轨迹类型
python3 -m src.main --viewer --leg FR --traj circle     # 圆形
python3 -m src.main --viewer --leg FR --traj line       # 直线
python3 -m src.main --viewer --leg FR --traj sine       # 正弦
python3 -m src.main --viewer --leg FR --traj lissajous  # 李萨如

# 使用 URDF 输入
python3 -m src.main --viewer --urdf --leg FL --traj circle
```

> **WSL 用户注意**: 图形化模式需要 Windows 端运行 X Server（如 VcXsrv），并设置 `export DISPLAY=:0`。

**窗口操作**: `Space` 暂停/恢复 · `滚轮` 缩放 · `右键拖拽` 旋转 · `Ctrl+右键拖拽` 平移

#### 无头模式（批量仿真 + 分析图表）

```bash
# FR 腿画圆，自动生成 4 张分析图表
python3 -m src.main

# 自定义参数
python3 -m src.main --leg FL --traj sine     # FL 腿正弦轨迹
python3 -m src.main --dt 0.001               # 仿真步长 1ms
python3 -m src.main --no-plot                # 跳过绘图
python3 -m src.main --output my_results/     # 自定义输出目录
python3 -m src.main --model path/to/robot.xml  # 自定义模型
```

输出图表（位于 `output/`）：
- `trajectory_3d_{leg}_{traj}.png` — 3D 轨迹对比（目标 vs 实际）
- `position_time_{leg}_{traj}.png` — 各轴位置-时间曲线
- `error_time_{leg}_{traj}.png` — 各轴跟踪误差（mm）
- `joints_{leg}_{traj}.png` — 关节角度-时间曲线

## 传统控制模块

### 运动学 (`kinematics.py`)

解析式 3-DOF 腿部运动学模型，基于旋量理论（Product of Exponentials）：

- **正向运动学 (FK)**: 给定关节角度 → 足端位置（髋关节坐标系）
- **雅可比矩阵**: 有限差分法计算 3×3 Jacobian
- **逆向运动学 (IK)**: Gauss-Newton 阻尼最小二乘法求解
  ```
  Δq = Jᵀ (JJᵀ + λ²I)⁻¹ e
  ```

```
Go1 腿部尺寸:
  - 大腿 (thigh): 0.213 m
  - 小腿 (calf):  0.213 m
  - 关节轴: abduction(X), hip_pitch(Y), knee_pitch(Y)
```

### 控制器 (`controller.py`)

`IKFootController` 是基于 MuJoCo FK 的闭环位置控制器：

1. 将世界坐标系目标变换到髋关节坐标系
2. 用 MuJoCo FK + 有限差分 Jacobian 求解 IK（确保运动学一致性）
3. 通过位置控制驱动关节执行器

关键设计：
- **MuJoCo FK 做 IK**：每次迭代调用 `sim.set_qpos` + `sim.forward` 获取足端位置，避免解析模型与仿真器之间的运动学偏差
- **自动适配命名约定**：MJCF 中执行器名为 `FR_hip`，URDF 转换后为 `FR_hip_joint`，控制器自动检测并适配
- **足端位置反馈**：优先使用 site (`FR`)，其次 body (`FR_foot`)，最后解析 FK 近似

### 轨迹生成 (`trajectory.py`)

支持的笛卡尔轨迹类型：

| 轨迹 | 类 | 参数 |
|------|-----|------|
| 直线 | `LinearTrajectory` | 起点、终点、时长 |
| 圆形 | `CircleTrajectory` | 圆心、半径、旋转轴、时长 |
| 正弦 | `SinusoidalTrajectory` | 中心、振幅、频率 |
| 李萨如 | `LissajousTrajectory` | 中心、振幅、频率比 |

所有轨迹继承自 `Trajectory` 抽象基类，提供统一的 `evaluate(t)` 和 `sample(dt)` 接口。

### 仿真器 (`simulator.py`)

`MuJoCoSim` 封装了 MuJoCo 底层 API：

- 模型加载与名称索引构建
- 身体/关节/执行器/Site 的位置与状态查询
- 固定基座 MJCF 生成（移除 freejoint，固定躯干高度）
- 材质提亮处理以改善渲染可见性

## 强化学习路线图（规划中）

传统控制提供了稳定、可解释的运动基元，但面对复杂地形时需要人工设计步态策略。强化学习模块将解决高层决策问题：

### Phase 1: 环境搭建

- [ ] 基于 MuJoCo 的 RL 训练环境（Gymnasium 接口）
- [ ] 状态空间：关节角度/角速度、IMU 数据、足端接触力、地形高度图
- [ ] 动作空间：12 维关节位置目标（四条腿 × 3 关节）
- [ ] 奖励函数：前进速度跟踪、能耗惩罚、姿态稳定、足端轨迹平滑

### Phase 2: 步态学习

- [ ] 平地行走步态（trot, walk, bound）
- [ ] 速度指令跟踪（前进/侧向/转向速度）
- [ ] 与传统 IK 控制器的对比基准

### Phase 3: 地形自适应

- [ ] 楼梯、斜坡、障碍物跨越
- [ ] 基于视觉/高度图的落足点选择
- [ ] Sim-to-Real 域随机化

### Phase 4: 层级控制融合

- [ ] 上层 RL 策略选择步态模式与落足点
- [ ] 下层传统控制器执行精确轨迹跟踪
- [ ] 策略切换与平滑过渡

### 候选 RL 算法

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| PPO | 稳定、易调参、广泛验证 | 步态学习 baseline |
| SAC | 样本效率高、连续动作空间 | 精细运动控制 |
| Dreamer | 基于世界模型、样本效率极高 | 复杂地形探索 |
| AMP | 对抗运动先验、风格迁移 | 自然步态生成 |

## 核心设计决策

- **固定基座仿真**: 第一阶段聚焦单腿运动学控制，移除躯干 freejoint，固定基座高度 0.445m。后续 RL 阶段将启动力学仿真。
- **MuJoCo FK 驱动 IK**: 控制器使用仿真器的前向运动学计算 Jacobian，而非解析式，确保 IK 解与仿真器状态完全一致。
- **双模型输入**: 同时支持原生 MJCF 和 URDF 两种格式，自动转换命名约定。
- **无头 + 图形化双模式**: 定量分析与交互演示共用同一套控制代码。

## 依赖

```
mujoco>=3.9.0
numpy>=1.21.0
scipy>=1.8.0
matplotlib>=3.5.0
```

## 许可证

本项目仅用于教育和研究目的。Go1 机器人模型版权归宇树科技（Unitree Robotics）所有。
