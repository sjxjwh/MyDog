# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

机器狗单个足尖的特定轨迹仿真 —— 在 MuJoCo 物理引擎中对四足机器人（宇树 Go1）的单条腿足尖进行轨迹规划与跟踪仿真。

## 技术栈

- Python 3.10 + MuJoCo 3.9.0, NumPy, SciPy, matplotlib
- 模型格式：MJCF（原生） + URDF（自动转换）
- IK 求解：基于 MuJoCo FK 的数值 Gauss-Newton 法（阻尼最小二乘）
- 固定基座仿真（freejoint 移除）

## 环境

```bash
source venv/bin/activate                    # 激活
python3 -m pip install <pkg>                # 安装包（使用 python3 -m pip）
python3 -m pip install -r requirements.txt  # 同步依赖
```

## 项目架构

```
MyDog/
├── model/
│   ├── go1.xml              # Go1 MJCF 模型（mujoco_menagerie）
│   ├── go1_fixed.xml        # 固定基座变体（自动生成）
│   ├── go1_from_urdf.xml    # URDF 转换的 MJCF（自动生成）
│   ├── assets/              # STL 网格文件（MuJoCo 兼容）
│   ├── scene.xml            # 场景定义
│   └── unitree/             # Unitree 官方 URDF + DAE 网格
│       └── meshes/
├── src/
│   ├── __init__.py          # 公开 API 导出
│   ├── simulator.py         # MuJoCoSim 封装（加载/步进/位姿查询）+ 固定基座 MJCF 生成
│   ├── kinematics.py        # LegKinematics：解析 FK, Jacobian, IK（独立于 MuJoCo 的模型）
│   ├── trajectory.py        # 笛卡尔轨迹：Linear, Circle, Sinusoidal, Lissajous
│   ├── controller.py        # IKFootController：基于 MuJoCo FK 的数值 IK 位置控制
│   ├── urdf_loader.py       # URDF → MJCF 转换器（处理 package:// 路径）
│   └── main.py              # 主入口：加载模型 → 生成轨迹 → 仿真 → 绘图
├── output/                  # 仿真结果图表
└── requirements.txt
```

## 关键设计

- **控制器使用 MuJoCo FK 做 IK**（`controller.py:solve_ik`）—— 每次迭代通过 `sim.set_qpos` + `sim.forward` 获取足端位置，用有限差分算 Jacobian，阻尼最小二乘求解。确保 IK 与仿真器运动学一致。
- **两种模型输入方式**：
  - MJCF 直接加载（`--model model/go1.xml`）
  - URDF 自动转换（`--urdf`，转换器处理 `package://` 路径，跳过 MuJoCo 不支持的 DAE 网格，使用原始碰撞几何体）
- **关节命名**：MJCF 中执行器名为 `FR_hip`，关节名为 `FR_hip_joint`；URDF 转换后执行器名和关节名均为 `FR_hip_joint`。控制器 `_find_names()` 自动适配两种约定。
- **足端位置**：优先用 site（`FR`），其次用 body（`FR_foot`），最后用解析 FK 近似。
- **图形化渲染**：viewer 模式加载 `model/scene_fixed.xml`（含地面、天空盒、头灯），而非裸模型文件。`create_fixed_base_mjcf()` 会自动将材质从深灰 `(0.2,0.2,0.2)` 提亮为 `(0.5,0.5,0.55)` 以改善渲染可见性。visual mesh 在 geom group 2，MuJoCo viewer 默认全部渲染。

## 两种运行模式

- **图形化模式**（`--viewer`）：打开 MuJoCo 原生 3D 交互窗口，实时观看机器狗足尖跟踪轨迹。
  - 操作：Space 暂停/恢复，滚轮缩放，右键拖拽旋转，Ctrl+右键拖拽平移。
  - 适用：调试轨迹、观察运动学、演示效果。
- **无头模式**（默认）：批量仿真 + 自动生成 `output/` 下的 4 张 PNG 分析图表（3D 轨迹对比、位置/时间、误差/时间、关节角度）。
  - 适用：定量分析、CI/CD、远程服务器。
  - 注意：WSL 中图形化需要 Windows 端有 X Server（如 VcXsrv），或使用 `export DISPLAY=:0` 指向运行的 X Server。

## 常用命令

```bash
# ===== 图形化模式 =====
# 默认：FR 腿画圆，3D 交互窗口
python3 -m src.main --viewer

# 指定腿和轨迹类型（窗口内实时观看）
python3 -m src.main --viewer --leg FR --traj circle     # 圆形
python3 -m src.main --viewer --leg FR --traj line       # 直线
python3 -m src.main --viewer --leg FR --traj sine       # 正弦
python3 -m src.main --viewer --leg FR --traj lissajous  # 李萨如

# URDF 输入 + 图形化
python3 -m src.main --viewer --urdf --leg FL --traj circle

# ===== 无头模式（批量仿真 + 生成分析图表） =====
# 默认：FR 腿画圆
python3 -m src.main

# 指定腿和轨迹类型
python3 -m src.main --leg FR --traj circle     # 圆形
python3 -m src.main --leg FR --traj line       # 直线
python3 -m src.main --leg FR --traj sine       # 正弦
python3 -m src.main --leg FR --traj lissajous  # 李萨如

# URDF 输入
python3 -m src.main --urdf --leg FL --traj circle

# 自定义 MJCF 模型
python3 -m src.main --model path/to/robot.xml --leg FR

# 仿真步长
python3 -m src.main --dt 0.001

# 跳过绘图（无头模式）
python3 -m src.main --no-plot

# 自定义输出目录
python3 -m src.main --output my_results/
```
