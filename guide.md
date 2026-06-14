四足机器人 Trot 步态问题排查指南（给智能体的排查文档）
本文用于指导智能体或工程调试流程，对四足机器人 Trot 步态中出现的：漂移、抖动、yaw不稳定、力分配失败等问题进行系统性排查。
1. 系统基本假设检查
- 是否明确使用浮动基动力学模型
- 是否存在接触状态（stance/swing）明确切换
- 是否假设足端无滑动（v_foot_world = 0）
- 是否忽略摩擦锥约束
- 是否忽略力矩饱和与关节力矩限制
2. 核心动力学一致性检查
检查是否满足：
M(q) a + h(q, qdot) = S^T f + J^T lambda

以及简化浮动基模型：
[a_x, a_y, alpha_z] 是否可由当前接触力集合生成
重点验证：
- A矩阵是否满秩（接触配置是否可控）
- 是否存在单支撑导致欠驱动
- yaw力矩是否可观
3. 接触力求解器检查（QP/WLS）
必须确认优化问题是否为带约束QP：
min ||A f - b||^2 + λ||f||^2
s.t. 摩擦锥约束 + 力界 + 接触模式
常见错误：
- 仅使用最小二乘（无约束）
- 未加入 friction cone
- 未限制法向力 fz > 0
- 未限制力变化率
4. Trot 步态特有问题
- 双支撑/单支撑切换导致解空间突变
- 支撑多边形面积过小导致 yaw 不稳定
- 相位切换未平滑（force discontinuity）
- swing leg 落脚误差导致 COM 偏移
5. 必须记录的调试数据
- 每条腿接触状态（stance/swing）
- foot position world/body frame
- contact force (fx, fy, fz)
- COM state (pos, vel, acc)
- base orientation (roll, pitch, yaw)
- QP residual and status
- friction cone violation count
6. 快速定位流程
Step 1：检查是否滑步（foot velocity ≠ 0 in world frame）
Step 2：检查QP是否不可行（infeasible / saturated）
Step 3：检查yaw torque是否剧烈振荡
Step 4：检查stance leg force是否突变
Step 5：关闭vy / wz，仅测试vx稳定性
Step 6：逐步增加控制自由度（vx → vx+vy → vx+vy+wz）
7. 常见根因总结
- 力分配未考虑约束（最常见）
- Trot切换不连续
- 摩擦锥未建模
- 接触估计错误
- COM估计漂移
- yaw控制过激或不可观