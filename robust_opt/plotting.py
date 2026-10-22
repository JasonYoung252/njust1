"""画图样式：统一配色与中英文字体回退（工控机上未必装了中文字体）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 固定顺序的类别色：基线 / SAA 方案 1 / SAA 方案 2
C_BASE = "#eb6834"
C_SAA = "#2a78d6"
C_SAA2 = "#1baf7a"
C_TEXT = "#3d3d3a"
C_GRID = "#e4e3dd"

_CJK = ["Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "PingFang SC"]


def _setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in _CJK if n in names]
    plt.rcParams.update({
        "font.sans-serif": cjk + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })
    return bool(cjk)


HAS_CJK = _setup()


def L(zh, en):
    """有中文字体用中文，否则用英文，避免出现方块字。"""
    return zh if HAS_CJK else en
