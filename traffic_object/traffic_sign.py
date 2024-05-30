import os

chinese_to_english = {
    # 背景类 非交通标志牌
    '其他': 'others',

    # 速度限制类别
    '最高限速': 'highest_speed_limit',
    '最低限速': 'lowest_speed_limit',
    '解除限速': 'lift_speed_limit',
    '禁止超车': 'no_overtaking',
    '解除禁止超车': 'lift_no_overtaking',

    # 警告类标志
    '停车让行': 'stop',
    '减速让行': 'slow_down_or_stop',
    '慢行': 'slow_down',
    '人行横道': 'pedestrian_crossing',
    '注意行人': 'watch_for_pedestrians',
    '注意儿童': 'watch_for_children',
    '施工': 'watch_for_construction',
    '交通监控设备': 'electronic_capture',
    '警告标志': 'warning_sign',
    '公交车站': 'bus_stop',

    # 行为限制类别
    '禁止停车': 'no_parking',
    '禁止鸣笛': 'no_horn',
    '禁止行人': 'no_pedestrians',
    '禁令标志': 'no_sign',

    # 禁止行驶方向
    '禁止通行': 'no_entry',
    '禁止直行': 'no_straight',
    '禁止左转': 'no_left',
    '禁止右转': 'no_right',
    '禁止掉头': 'no_turn',
    '禁止掉头和向左转弯': 'no_turn_and_no_left',
    '禁止向左和向右转弯': 'no_left_and_no_right',
    '禁止直行和右转': 'no_straight_and_no_right',
    '禁止直行和左转': 'no_straight_and_no_left',

    # 指示类
    '直行': 'straight',
    '直行和向右转弯': 'straight_or_right',
    '向右转弯': 'right',
    '允许掉头': 'turn',
    '向左和向右转弯': 'left_or_right',
    '左侧通行': 'left_side_traffic',
    '右侧通行': 'right_side_traffic',
    '两侧通行': 'side_traffic',
    '靠左侧道路行驶': 'drive_on_left_side',
    '靠右侧道路行驶': 'drive_on_right_side',
    '靠左侧和靠右侧道路行驶': 'drive_on_side',
    '分向行驶车道': 'diverge_lane',
    '线性诱导': 'linear_induction',
    '停车区': 'parking',
    'T形交叉': 'T_shape_cross',
    '步行': 'walk',
    '道路指示牌': 'road_sign',
    '指示标志': 'guide_sign',

    # 车型类限制
    '限高': 'height_limit',
    '限宽': 'width_limit',
    '限重': 'weight_limit',
    '非机动车行驶': 'non_motor_vehicle',
    '机动车行驶': 'motor_vehicle',
    '禁止非机动车通行': 'no_non_motor_vehicle',
    '禁止两轮摩托车通行': 'no_two_wheel_motor',
    '禁止大型客车驶入': 'no_large_bus',
    '禁止载货汽车通行': 'no_truck',
    '公交线路专用车道': 'bus_lane',
    '禁止危险品运输车辆驶入': 'no_chemicals',
    '禁止机动车驶入': 'no_motor_vehicle',


    # 难以处理的corner case
    '复合型标志牌': 'composite_sign',
    '模糊': 'vague_sign',
}

english_to_chinese = {}
tsr_class_names = []
for chinese, english in chinese_to_english.items():
    english_to_chinese[english] = chinese
    tsr_class_names.append(english)

need_ocr_traffic_sign = [
    "highest_speed_limit",
    "lowest_speed_limit",
    "weight_limit",
    "height_limit",
    "width_limit",
]

ocr_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'm', 't']

high_confidence_speed = ['5', '15', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120']
