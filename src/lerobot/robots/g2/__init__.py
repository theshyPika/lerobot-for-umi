from .config_g2 import G2RobotConfig

# 尝试导入 G2Robot，但如果 agibot_gdk 不可用，则将其设置为 None
try:
    from .g2 import G2Robot
    G2_AVAILABLE = True
except ImportError:
    G2Robot = None
    G2_AVAILABLE = False

__all__ = [
    "G2Robot",
    "G2RobotConfig"
]
