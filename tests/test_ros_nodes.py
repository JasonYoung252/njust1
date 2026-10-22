# -*- coding: utf-8 -*-
"""
节点级集成测试：用假 rospy / pyrealsense2 / PaddleOCR 运行 target_position.py、D435_rgb_depth.py、
arm_control.py 的真实代码。相机数据来自 sim/synthetic_d435.py 的合成场景，有真值可对照。
"""
import importlib
import math
import sys
import threading
import types

import numpy as np
import pytest

import fakes

FAKES = fakes.install()

import convert  # noqa: E402
import pose_utils  # noqa: E402
from synthetic_d435 import SceneParams, SyntheticScene  # noqa: E402

ARM_POSE = [0.30, -0.02, 0.45, 3.10, -1.50, 0.05]   # 末端位姿（m, rad），用于手眼变换


def _fresh(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.fixture
def node():
    rospy, rs, paddle = FAKES["rospy"], FAKES["rs"], FAKES["paddle"]
    rospy.params.clear()
    rospy.logs.clear()
    rng = np.random.default_rng(11)
    p = SceneParams(distance_m=0.36, yaw_deg=4.0, label_dropout=0.8, plate_dropout=0.95, label_offset_px=(40, -20))
    scene = SyntheticScene(p, rng)
    rs.intr = rs.intrinsics(p.fx, p.fy, p.cx, p.cy, p.width, p.height)
    color = np.full((p.height, p.width, 3), 200, np.uint8)
    rs.frame_source = lambda: (scene.render_frames(1)[0], color)
    poly = [[float(a), float(b)] for a, b in scene.label_polygon()]
    paddle.results = [[[[[10.0, 10.0], [60.0, 10.0], [60.0, 30.0], [10.0, 30.0]], ("1KLP2", 0.95)],
                       [poly, ("U4C63V", 0.97)]]]

    D435 = _fresh("D435_rgb_depth")
    tp = _fresh("target_position")
    tp.params["debug_view"] = False
    tp.ocr = paddle.PaddleOCR()
    tp.realsense_processor = D435.RealSenseProcessor(warmup_frames=2)
    from depth_localizer import DepthLocalizer, LocalizerConfig
    tp.localizer = DepthLocalizer(tp.realsense_processor.intrinsics, LocalizerConfig())
    tp.Armstate[:] = ARM_POSE
    tp.Arm_State[:] = [ARM_POSE[0], ARM_POSE[1], ARM_POSE[2], 0.0, 0.0, 0.0, 1.0]
    return types.SimpleNamespace(tp=tp, scene=scene, paddle=paddle, rospy=rospy, rs=rs)


def test_ocr_detection_end_to_end(node):
    pose = node.tp.ocr_detection("U4C63V")
    assert pose is not None and len(pose) == pose_utils.POSE_LEN
    info = pose_utils.unpack_pose(pose)
    assert info["status"] == pose_utils.STATUS_PLANE
    truth_base = np.array(convert.convert(*node.scene.label_center, *ARM_POSE))
    err_mm = np.linalg.norm(np.array([info["x"], info["y"], info["z"]]) - truth_base) * 1000
    assert err_mm < 3.0, err_mm
    n_base = convert.rotate_to_base(node.scene.normal, *ARM_POSE)
    expected_yaw = math.atan2(-n_base[1], -n_base[0])
    assert abs(pose_utils.wrap_angle(info["facing_yaw"] - expected_yaw)) < math.radians(1.0)
    # 同一次采集：settle 3 帧 + n_frames 5 帧 + 预热 2 帧
    assert node.tp.realsense_processor.pipeline.frames_served == 2 + 3 + 5


def test_text_match_rules():
    tp = sys.modules.get("target_position") or _fresh("target_position")
    assert tp.text_matches(" U4C63V ", "U4C63V")
    assert tp.text_matches("I2LP", "12LP")
    assert tp.text_matches("l2LP", "12LP")
    assert not tp.text_matches("U4C63W", "U4C63V")


def test_pose_service_success_failure_and_busy(node):
    tp = node.tp
    threading.Thread(target=tp.position_thread, daemon=True).start()
    req = types.SimpleNamespace(target="U4C63V")
    resp = tp.poseCallback(req)
    assert pose_utils.unpack_pose(resp.pose)["valid"]

    resp = tp.poseCallback(types.SimpleNamespace(target="NOPE"))
    assert resp.pose == pose_utils.failed_pose()

    tp.service_lock.acquire()
    try:
        resp = tp.poseCallback(req)          # 原程序这里返回字符串 "wait"
        assert resp.pose == pose_utils.failed_pose()
    finally:
        tp.service_lock.release()


def test_all_holes_reports_failure_not_origin(node):
    node.rs.frame_source = lambda: (np.zeros((540, 960), np.float32), np.zeros((540, 960, 3), np.uint8))
    assert node.tp.ocr_detection("U4C63V") is None
    assert any("定位失败" in m for lvl, m in node.rospy.logs)


def test_capture_saving_and_csv(node, tmp_path):
    node.tp.params["log_csv"] = str(tmp_path / "loc.csv")
    node.tp.params["save_dir"] = str(tmp_path / "caps")
    assert node.tp.ocr_detection("U4C63V") is not None
    lines = (tmp_path / "loc.csv").read_text().strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("time,target,status")
    saved = sorted(p.name for p in (tmp_path / "caps").iterdir())
    assert any(n.endswith("_depth.npz") for n in saved) and any(n.endswith("_meta.json") for n in saved)
    depth = np.load(next((tmp_path / "caps").glob("*_depth.npz")))["depth_mm"]
    assert depth.shape == (5, 540, 960) and depth.dtype == np.uint16


def test_legacy_camera_api_still_works(node):
    rp = node.tp.realsense_processor
    intr, depth_intrin, color, depth, frame = rp.get_aligned_images()
    u, v = [int(a) for a in node.scene.label_uv]
    avg, _, _, X, Y, Z = rp.process_frame([u, v])
    # 标签 80% 是空洞：中心像素多半为 0，现在应改用邻域中位数，不再返回原点
    assert Z == pytest.approx(node.scene.label_center[2], abs=0.004)


# ----------------------------------------------------------------------------------------------
# arm_control
# ----------------------------------------------------------------------------------------------
class _Rec:
    def __init__(self):
        self.calls = []


@pytest.fixture
def arm():
    rospy = FAKES["rospy"]
    ac = _fresh("arm_control")
    obj = object.__new__(ac.arm_control)
    obj.current_request = types.SimpleNamespace(target="U4C63V", direction=1, type="p", offset=[0.0, 0.03, 0.04])
    obj.current_pose = [0.0, -0.30, 0.45, 0, 0, 0, 1]
    obj.max_pose_attempts = 3
    obj.use_yaw_correction = True
    obj.max_yaw_correction_deg = 10.0
    obj.yaw_delta = 0.0
    obj.target_info = None
    obj.action = False
    moves = []
    obj.get_state = lambda: None
    obj.position_dxyz = lambda x, y, z: moves.append((x, y, z))
    published = []
    obj.result_pub = types.SimpleNamespace(publish=lambda m: published.append(m))
    responses = []
    obj.client = lambda target: responses.pop(0)
    return types.SimpleNamespace(module=ac, obj=obj, moves=moves, published=published, responses=responses,
                                 rospy=rospy)


def _resp(pose):
    return fakes.PoseResponse(pose)


def test_get_target_pose_retries_then_succeeds_with_yaw(arm):
    yaw = pose_utils.nominal_facing_yaw(1) + math.radians(5)
    good = pose_utils.pack_pose((0.02, -0.55, 0.40), (0, 0, 0, 1), yaw, 0.002, 0.001, 0.3, 0.8, 1)
    arm.responses += [_resp(pose_utils.failed_pose()), _resp(pose_utils.failed_pose()), _resp(good)]
    assert arm.obj.get_target_pose() is True
    assert arm.obj.yaw_delta == pytest.approx(math.radians(5))
    assert arm.obj.target_info["status"] == 1
    dx, dy = arm.obj.dxy(0.1, 0.0)
    assert math.atan2(dy, dx) == pytest.approx(yaw)


def test_yaw_correction_disabled_keeps_original_offsets(arm):
    arm.obj.use_yaw_correction = False
    yaw = pose_utils.nominal_facing_yaw(1) + math.radians(5)
    arm.responses.append(_resp(pose_utils.pack_pose((0.02, -0.55, 0.4), (0, 0, 0, 1), yaw, 0, 0, 0, 0, 1)))
    arm.obj.get_target_pose()
    assert arm.obj.yaw_delta == 0.0
    assert arm.obj.dxy(0.1, 0.02) == pytest.approx((0.02, -0.1))


def test_legacy_seven_element_response(arm):
    arm.responses.append(_resp([0.02, -0.55, 0.40, 0, 0, 0, 1]))
    assert arm.obj.get_target_pose() is True
    assert arm.obj.yaw_delta == 0.0


def test_target_not_found_aborts_task_and_retreats(arm):
    arm.responses += [_resp(pose_utils.failed_pose())] * 3

    def dispatch():
        arm.obj.get_target_pose()
        raise AssertionError("定位失败后不应继续执行动作")
    arm.obj._dispatch_task = dispatch
    arm.obj.action = True
    ticks = iter([False, True])
    arm.rospy.is_shutdown = lambda: next(ticks, True)
    try:
        arm.obj.action_thread_func()
    finally:
        arm.rospy.is_shutdown = lambda: False
    assert arm.moves == [(-0.1, 0, 0)]
    assert arm.published and arm.published[-1].startswith("fail")
    assert arm.obj.action is False


def test_dispatch_contains_original_branches(arm):
    import inspect
    src = inspect.getsource(arm.module.arm_control._dispatch_task)
    for t in ("'d'", "'p'", "'button'", "'knob'", "'switch'", "'door'", "'door1'", "'door2'", "'door3'"):
        assert "self.current_request.type == %s" % t in src


def test_main_reads_launch_params(node):
    tp, rospy = node.tp, node.rospy
    rospy.params.update({"~n_frames": 3, "~debug_view": False, "~depth_width": 640, "~depth_height": 480,
                         "~visual_preset": "high_density", "~laser_power": "max",
                         "~localizer/plane_radius_m": 0.05})
    try:
        tp.main()
    finally:
        rospy.params.clear()
    assert tp.params["n_frames"] == 3
    assert tp.localizer.cfg.plane_radius_m == 0.05
    sensor = tp.realsense_processor.profile.get_device().first_depth_sensor()
    assert sensor.options["visual_preset"] == 4 and sensor.options["laser_power"] == 360.0


def test_replay_saved_captures(node, tmp_path):
    import replay_captures
    node.tp.params["save_dir"] = str(tmp_path)
    for _ in range(3):
        assert node.tp.ocr_detection("U4C63V") is not None
    node.tp.params["save_dir"] = ""
    rows = replay_captures.replay(str(tmp_path), replay_captures.LocalizerConfig())
    assert len(rows) == 3 and all(r["status"] == 1 for r in rows)
    s = replay_captures.summarize(rows)["U4C63V"]
    assert s["frame"] == "base" and s["success_rate"] == 1.0
    truth_base = np.array(convert.convert(*node.scene.label_center, *ARM_POSE))
    assert abs(rows[0]["z_base"] - truth_base[2]) < 0.003
