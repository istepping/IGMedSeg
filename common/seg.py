from model.inter_seg import InterSeg

MODEL = InterSeg(sgd=True, load_model=False, show_loss=False, in_channels=2, fc=False)

OPERATION_IMG = 0  # 加载图像阶段
OPERATION_PRE = 1  # 图像与分割阶段
OPERATION_VIEW = 2  # 图像预览阶段
OPERATION_MODIFY = 3  # 交互阶段

ORIGIN_IMAGE_SIZE = []  # 记录图片大小

IMAGE_NAME = []  # 记录文件名

# IGMedSeg
INITIAL_POINTS = []  # 记录初始交互点
INIT_CONTROL_POS = []  # 原始坐标点位置
CONTROL_POS = []  # 控制点集合
IMAGE_PATH = []  # 分割图片路径
GT_JSON_PATH = []  # 对应GT路径
DEFAULT_DIR = [r"D:\datasets\PA\Image"]  # 默认打开文件路径
INTERACTIVE_POINT = []  # 交互点击点记录
TRUE_CONTROL_INDEX = []  # 用户点击的确定的点对应的索引->都需不会被更改
SHIFT_INDEX = []
# value
FIT_ELLIPSE = 0  # 椭圆拟合
FIT_HULL = 1  # 凸包拟合
FIT_SPLINE = 2  # 样条拟合
VISION_POINT = 0
VISION_LINE = 1
VISION_ALL = 2
# Param
VISION_WAY = VISION_ALL
SHOW_LOSS = False  # 显示Loss曲线