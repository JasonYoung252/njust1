# njust1
机械臂代码

## 压板定位深度优化

`target_position.py` 现在用多帧融合、标签 ROI 中位数和面板平面先验定位，解决 D435 深度空洞导致的定位失败；`arm_control.py` 增加有限重试、失败中止和可选的偏航修正。说明见 [docs/depth_localization.md](docs/depth_localization.md)。

```bash
python3 -m pytest -q tests        # 不需要 ROS、相机和 OCR
python3 sim/run_simulation.py     # 合成场景对比，结果在 sim/results/
```
