# 视觉定位不确定下的鲁棒投退位姿优化

随机优化课程项目。目标：用 SAA 求出的补偿量 dp（与观测帧数 n），替代 `arm_control.py` 里手调的压板补偿常数。

## 约定

- 任务坐标系：面朝标签，前 x（接近/按压方向）、左 y、上 z，与 `arm_control.dxy()` 一致，单位 m。
- 测量误差 m = p_hat − p_true；执行误差 e = dp + m。
- 成功：|e_y| ≤ τ_y，|e_z| ≤ τ_z，−τ_x⁻ ≤ e_x ≤ τ_x⁺；e_x > τ_x⁺ 记为过压。

## 文件

| 文件 | 作用 |
|---|---|
| `robust_opt/noise_model.py` | 随机误差模型：参数法（仿真）与 bootstrap（实测） |
| `robust_opt/objective.py` | 容差、成功判定、代价函数 |
| `robust_opt/saa_solver.py` | 模型 A（期望代价）、模型 B（机会约束）、情景法样本量 |
| `robust_opt/evaluate.py` | 样本外评估与 Wilson 置信区间 |
| `robust_opt/simulate.py` | 第 1 步：纯仿真，输出 3 张图和结果表 |
| `robust_opt/fit_from_data.py` | 第 3 步：读真机日志求解，生成补偿配置 |
| `scripts/collect_gt.py` | 第 2 步：真值采集（TCP 触碰标签中心） |
| `config/compensation_baseline.yaml` | 基线：原手调常数（与改动前行为完全一致） |

## 流程

**1. 仿真（不需要硬件）**

```bash
cd ~/Arm_ws/src/njust
python3 -m robust_opt.simulate --out results/sim
```

噪声参数和容差在 `noise_model.ParametricNoise`、`objective.Tolerance` 中，目前都是假设值。

**2. 采集数据**

```bash
roslaunch njust rm_63_ts.launch                        # 默认 comp_profile:=baseline
# 静态重复测量：机械臂停在观测位姿不动，一次测 60 帧，换 3~4 个距离各做一次
rosparam set /robust_opt/direction 0
rosparam set /target_position/n_frames 60
rosservice call /pose "target: 'U4C63V'"
# 真值：10~15 轮，每轮换观测位姿，按提示操作
rosrun njust collect_gt.py _target:=U4C63V _direction:=0 _n_frames:=30 _tip_offset:="[0.0, 0.0, 0.0]"
```

数据写入 `~/robust_opt_logs/`：`vision_*.csv`（逐帧测量）、`gt.csv`（真值）、`exec_*.csv`（每次压板操作）。
`collect_gt.py` 结束时会把 `n_frames` 和 `direction` 恢复原值；手动 `rosparam set` 的记得改回去。

**3. 求解并生成配置**

```bash
python3 -m robust_opt.fit_from_data \
    --vision "~/robust_opt_logs/vision_*.csv" --gt ~/robust_opt_logs/gt.csv \
    --tol 2.5 2.5 3.0 1.5 --out-yaml config/compensation_robust.yaml
```

`--tol` 依次为 τ_y、τ_z、τ_x⁻、τ_x⁺（mm），按压板和夹爪实测尺寸填写。

**4. 真机对比**

```bash
roslaunch njust rm_63_ts.launch comp_profile:=baseline   # 原手调常数
roslaunch njust rm_63_ts.launch comp_profile:=robust     # 优化方案
```

每次操作后在 `exec_*.csv` 的 `outcome` 列填 `success` / `fail` / `over`。

## 注意

- 真值测的是**标签文字框中心**（OCR 定位的点），不是压板。标签到压板的偏移是确定的几何量，不计入随机误差。
- `tip_offset` 是尖头相对 `/rm_driver/ArmCurrentState` 所报位姿的偏移（末端坐标系）。若控制器里已把工具坐标系设到尖头，填 `[0, 0, 0]`。
- 情景法样本量界严格成立的前提是凸问题、且所有样本约束都满足；这里只作为样本量量级的参考。
