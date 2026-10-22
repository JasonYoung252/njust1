# 压板定位深度优化说明

本次修改解决 D435 在压板、标签上深度空洞导致定位失败或静默出错的问题，并为论文实验留出数据接口。原有调用方式和 `/pose`、`/task` 服务定义都没有改变。

## 改了什么

| 文件 | 修改 |
| --- | --- |
| `scripts/depth_localizer.py`（新增） | 纯 numpy 的鲁棒定位：多帧融合、标签 ROI 中位数、面板平面 RANSAC + 射线求交、不确定性输出 |
| `scripts/pose_utils.py`（新增） | `/pose` 返回值的打包和解析；偏移量的偏航修正 |
| `scripts/D435_rgb_depth.py` | 深度流改为 848×480；对齐对象只建一次；阈值和空间滤波；预热帧；`capture_bundle()` 一次采集多帧；单像素取值遇到空洞时改用邻域中位数 |
| `scripts/target_position.py` | 同一次采集完成 OCR 和取深度；用 `DepthLocalizer` 定位；返回 13 维 pose；修复忙碌时返回字符串 `"wait"` 的类型错误；忙等改为事件等待；可选 CSV 日志和原始数据保存 |
| `scripts/arm_control.py` | 定位最多重试 `max_pose_attempts` 次，失败后中止任务、后退 10 cm 并上报 `fail`；可选按实测面板朝向修正偏移方向（默认关闭） |
| `scripts/convert.py` | 拆出变换矩阵函数，新增法向和协方差的坐标变换；`convert()` 结果与原版逐点一致 |
| `scripts/exception.py` | 新增 `TargetNotFoundException` |
| `launch/rm_63_ts.launch` | 增加上述参数 |
| `sim/`（新增） | 合成 D435 深度场景和蒙特卡洛对比；`replay_captures.py` 用真机保存的数据离线重跑定位、统计重复性 |
| `tests/`（新增） | 单元测试，以及用假 rospy / pyrealsense2 / PaddleOCR 跑通节点代码的集成测试 |

`arm_control_nl.py` 没有修改。它引用的 `poseRequest.rank`、`task.angle` 字段在本仓库的 `srv/` 中不存在，与当前 launch 使用的 `arm_control.py` 不是同一套接口。

## 定位流程

1. 机械臂停稳后，丢弃 `settle_frames` 帧旧数据，再连续采集 `n_frames` 帧。
2. 用最后一帧彩色图做 OCR，得到标签文字框。
3. 多帧深度逐像素取中位数，忽略 0 值。
4. 在标签框内取鲁棒中位深度，作为参考深度。
5. 在标签周围半径 `plane_radius_m` 内、深度接近参考值的点上，用 RANSAC 拟合面板平面。标签贴在面板上，所以“标签中心像素的视线”与平面的交点就是标签位置。标签和压板本身全是空洞时也能定位。
6. 平面不可用时（点太少、范围太小、视线过斜），回退到标签 ROI 中位数；两者都不可用则返回失败，不再返回原点。
7. 输出位置、面板法向和不确定性。

## `/pose` 返回值

前 7 个元素与原来相同，后 6 个为新增：

| 下标 | 含义 |
| --- | --- |
| 0–2 | 标签在基座坐标系下的位置（m） |
| 3–6 | 当前末端姿态四元数（沿用原逻辑） |
| 7 | 机械臂面朝面板的实测方向角（rad，0 = 面朝基座 +x）；ROI 回退时为 NaN |
| 8 | 沿面板法向的标准差（m） |
| 9 | 横向标准差（m） |
| 10 | 标签 ROI 有效深度比例 |
| 11 | 面板平面内点比例 |
| 12 | 状态：0 失败，1 面板平面，2 标签 ROI |

失败时返回 13 个 0。旧版 `arm_control` 的检查同样会判为无效。

## 参数

`target_position` 节点：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `~n_frames` | 5 | 每次定位采集的深度帧数 |
| `~settle_frames` | 3 | 采集前丢弃的旧帧数 |
| `~debug_view` | true | 是否弹出 OCR 调试窗口（原行为） |
| `~log_csv` | 空 | 定位日志 CSV 路径 |
| `~save_dir` | 空 | 保存彩色图、原始深度帧（uint16, mm）和元数据的目录 |
| `~depth_width` / `~depth_height` | 848 / 480 | 深度分辨率 |
| `~visual_preset` | 空 | 留空不修改；可选 `high_accuracy`、`high_density`、`default` |
| `~laser_power` | 空 | 留空不修改；`max` 或数值（mW） |
| `~localizer/<字段>` | 见 `LocalizerConfig` | 定位算法参数，例如 `plane_radius_m`、`plane_inlier_thresh_m`、`min_roi_valid_ratio`、`systematic_sigma_m` |

`arm_control` 节点：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `~max_pose_attempts` | 10 | 定位最多重试次数 |
| `~use_yaw_correction` | false | 是否按实测面板朝向旋转偏移量 |
| `~max_yaw_correction_deg` | 10 | 偏航超过该值时不修正（视为测量异常） |

## 现场验证建议

1. **先只看日志，不改动作**。保持 `use_yaw_correction=false`，打开 `log_csv`，正常执行任务。确认日志里 `status` 多为 1，`ROI有效率` 和 `平面内点` 的数值合理。
2. **静态重复性**。打开 `save_dir`，机械臂停在同一位置，对同一标签连续定位 30 次，再运行 `python3 sim/replay_captures.py <save_dir>` 统计位置标准差，与日志里的 σ 对比。保存的数据也可以换参数离线重跑，不必重新上机。
3. **偏航**。让底盘相对屏柜转 0°、5°、10°，看日志中的“面板实测偏航”是否跟着变化、是否稳定在 ±1° 以内。稳定后再开启 `use_yaw_correction`。
4. **相机参数**。在 launch 外用 `D435_rgb_depth.py` 直接运行，可实时看到画面中心的深度有效率。对比 `visual_preset`（`high_accuracy` / `high_density`）和 `laser_power` 的效果，再写进 launch 参数。
5. 原手眼标定的平移 z 有两组相差 11 mm 的值，建议重新标定，并在 `save_dir` 采集的数据上评估。

## 测试与仿真

```bash
pip install pytest            # 只有运行测试需要
python3 -m pytest -q tests    # 不需要 ROS、相机和 OCR
python3 sim/run_simulation.py # 约 10 分钟，结果写入 sim/results/
```

仿真结果见 `sim/results/summary.md`。合成场景是简化模型，用来验证代码和比较方法的相对优劣，论文中的数字必须来自真机实验。
