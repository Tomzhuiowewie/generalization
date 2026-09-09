import torch


def water_area(water_level, geometry, mask):
    """
    water_level: [B] 或 [B, 1]
    geometry:    [B, P, 2]，最后一维为 [横向距离, 高程]
    mask:        [B, P]，True 表示真实点
    返回:        area [B], perimeter [B]
    """
    x = geometry[..., 0]                 # 横向距离
    ground = geometry[..., 1]            # 地面高程
    level = water_level.reshape(-1, 1)   # 每个断面的水位

    # 1. 每两个相邻地形点组成一条线段。
    segment_width = x[:, 1:] - x[:, :-1]

    # 2. 计算线段两端水深，水面以上的水深记为 0。
    signed_left = level - ground[:, :-1]
    signed_right = level - ground[:, 1:]
    depth_left = signed_left.clamp_min(0)
    depth_right = signed_right.clamp_min(0)

    # 3. 计算每条线段有多少比例位于水下。
    both_wet = (signed_left > 0) & (signed_right > 0)
    one_wet = (signed_left > 0) ^ (signed_right > 0)

    wet_fraction = torch.zeros_like(segment_width)
    wet_fraction[both_wet] = 1.0
    wet_fraction[one_wet] = (
        torch.maximum(signed_left, signed_right)[one_wet]
        / (signed_left - signed_right).abs()[one_wet].clamp_min(1e-8)
    )

    # 4. 梯形面积求和；mask 排除补零点形成的无效线段。
    valid = mask[:, :-1].bool() & mask[:, 1:].bool()
    area = 0.5 * (depth_left + depth_right) * segment_width * wet_fraction

    # 湿周使用地形线段的实际斜边长度，不包含水面宽度。
    segment_height = ground[:, 1:] - ground[:, :-1]
    segment_length = torch.sqrt(segment_width**2 + segment_height**2)
    perimeter = segment_length * wet_fraction

    return (area * valid).sum(dim=1), (perimeter * valid).sum(dim=1)


def water_area_at_x(x, water_level, section_x, geometry, mask):
    """计算任意纵向位置 x 的过水面积和湿周: 基于离散断面数据的线性插值"""
    x = x.reshape(-1)
    right = torch.searchsorted(section_x, x).clamp(1, len(section_x) - 1)
    left = right - 1
    weight = (x - section_x[left]) / (section_x[right] - section_x[left])

    area_left, perimeter_left = water_area(
        water_level, geometry[left], mask[left]
    )
    area_right, perimeter_right = water_area(
        water_level, geometry[right], mask[right]
    )

    area = (1 - weight) * area_left + weight * area_right
    perimeter = (1 - weight) * perimeter_left + weight * perimeter_right
    return area, perimeter

if __name__ == "__main__":
    # 测试
    geometry = torch.tensor([
        [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
        [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
    ])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    water_level = torch.tensor([1.5, 1.5])
    area, perimeter = water_area(water_level, geometry, mask)
    assert torch.allclose(area, torch.tensor([0.25, 0.125]))
    expected_perimeter = torch.tensor([2**0.5, 2**0.5 / 2])
    assert torch.allclose(perimeter, expected_perimeter)
    print("area:", area)
    print("perimeter:", perimeter)