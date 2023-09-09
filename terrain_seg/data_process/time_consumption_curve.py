import matplotlib.pyplot as plt

if __name__ == "__main__":
    time_list = []

    with open(r'E:\比亚迪全地形分割\全地形+高程2.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')  # 去除文本中的换行符
            if "run time:" in ann:
                time_ms = float(ann.split(": ")[-1].strip("ms"))
                time_list.append(time_ms)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("每帧AI推理耗时(全地形5FPS,包含前处理)")
    plt.xlabel("帧序号")
    plt.ylabel("耗时ms")
    plt.bar(range(100), time_list[:100])
    plt.show()
