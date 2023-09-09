## 交通标志牌识别需求

    'others': 0,
    'highest_speed_limit': 1,
    'lowest_speed_limit': 2,
    'height_limit': 3,
    'weight_limit': 4,
    'no_overtaking': 5,
    'no_entry': 6,
    'no_parking': 7,
    'release_speed_limit': 8,
    'release_no_overtaking': 9,
    'direction_sign': 0,

    宇通房车项目暂时只需要识别限速、解除限速、禁止超车、解除禁止超车。
    为了提高分类模型的兼容能力，增加了一些常见类别。
    限速标志牌分为了最高和最低限速，这两种标志牌需要ocr检测：
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'm', 't'
    
### 效率和精度
    客户要求，白天大于0.97，晚上大于0.90。
    推理速度上，分类模型应不超过3ms，ocr模型应不超过5ms。

## 推理流程
    1.初始化分类模型和ocr模型。
    2.一阶段检测，跟踪稳定后，从原图中crop。
    3.resize，归一化后，模型分类,多帧统计得出最终类型。
    4.如果是限速标志牌，则ocr模型检测并组合出速度。
    详细推理细节参考/traffic_sign/traffic_sign_predictor。

## demo
    python traffic_sign/demo.py
    修改video_path路径即可
## 优化计划
    1.目前分类模型精度欠佳，需要清洗数据后调整。
    2.模型效率暂时没有顾及，前期用fp32跑通流程，后期考虑int8或fp16，另外还可以稀疏化。
    3.可设置if_recognition，如果宽高比不正常或统计识别为others不再输入到分类网络。