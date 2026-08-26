from typing import Tuple

from rich.color import Color
from rich.style import Style


def lerp_color(val: float, min_val: float, max_val: float, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> Style:
	"""
    生成自然色阶的核心函数：线性插值 (Linear Interpolation)
    val: 当前值
    min_val, max_val: 数据的范围
    start_rgb: 最小值对应的颜色 (R, G, B)
    end_rgb: 最大值对应的颜色 (R, G, B)
    """
	if max_val == min_val:
		ratio = 0.5  # 只有一个数据时，取中间色
	else:
		# 将数值映射到 0.0 - 1.0
		ratio = (val - min_val) / (max_val - min_val)
		ratio = max(0.0, min(1.0, ratio)) # 限制在 0-1 之间

	# 计算插值后的 RGB
	r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
	g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
	b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)

	# 返回 Rich 的 Style 对象
	return Style(color=Color.from_rgb(r, g, b))