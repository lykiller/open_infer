# -*-coding:utf-8-*-
import onnx
import torch
from tools.ModelTools import check_update_model


class Preprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.mean和self.std写在init里是为了让它变成一个确定的东西。如果放到forward函数里，导onnx时就会生成出多的节点
        # self.mean = torch.rand(1, 1, 1, 3)
        # self.std = torch.rand(1, 1, 1, 3)
        # print(type(self.mean))

    def forward(self, x):
        # 输入： x = B × H × W × C   Uint8
        # 输出： y = B × C × H × W   Float32 减去均值除以标准差
        x = x.float()  # 转换成float32，不然默认会变成float64，float64就太慢了
        # x = (x / 255.0 - self.mean) / self.std
        # x = x / 255.0
        # x = x.permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':
    pre = Preprocess()
    # (torch.zeros(1,640,640,3,dtype=torch.uint8),)输入用元组表示，这里表示只有一个输入
    torch.onnx.export(pre,
                      torch.zeros(1, 3, 256, 768, dtype=torch.uint8),
                      "preprocess.onnx",
                      input_names=["pre_input"],
                      output_names=["pre_output"]
                      )

    # 在拿到preprocess.onnx文件后，就把它读进来，和原来的onnx对接起来，这样就完成了把预处理加到原来的onnx里了
    pre_onnx = onnx.load("preprocess.onnx")

    model_path = r"\\10.10.10.136\Publish\1_Business\2_BYDz\3_AI\5_AllTerrain_xRoad\V0.4.4.20231228_Terrain_cls12_256x768\1_onnx\V0.4.4.20240103_Terrain_cls12_256x768.onnx"
    model = onnx.load(model_path)

    # *------------------ 接下来把预处理作为新节点添加进原onnx文件的具体步骤 ------------------* #
    # 第一步 先把pre_onnx的所有节点以及输入输出名称都加上前缀，因为可能和需要合并的onnx文件造成名称冲突
    for node in pre_onnx.graph.node:
        if "pre" not in node.name:
            node.name = "pre_" + node.name
            # node.name = f"pre/{node.name}"  # 加前缀,输出类似为pre/Div_4
            for i in range(len(node.input)):
                if "pre" not in node.input[i]:
                    node.input[i] = "pre_" + node.input[i]
            for i in range(len(node.output)):
                if "pre" not in node.output[i]:
                    node.output[i] = "pre_" + node.output[i]

    # node_name = ""
    # 第二步 先把原onnx中的输入的节点为input的节点，修改为pre_onnx的输出节点
    for node in model.graph.node:
        if node.input[0] == "data":
            node.input[0] = pre_onnx.graph.output[0].name

    # 第三步 把pre_onnx的node全部放到原onnx文件的node中，这是因为pre_onnx.onnx只用到了node，所以只需要把node加到原onnx文件中的node里去，但如果pre_onnx.onnx里除了node还有initializer，这时也得把initializer加到原onnx文件中的node里去。
    # last_pre_node = ""
    for node in pre_onnx.graph.node:
        model.graph.node.append(node)

    # 第四步 把pre_onnx的输入名称作为原onnx文件的input名称
    input_node_name = pre_onnx.graph.input[0].name
    model.graph.input[0].CopyFrom(pre_onnx.graph.input[0])
    model.graph.input[0].name = input_node_name

    # 检查网络
    new_model = check_update_model(model)

    # 节点名改回input
    for input_node in new_model.graph.input:
        if 'pre_input' == input_node.name:
            input_node.name = "data"
    for id, node in enumerate(new_model.graph.node):
        for i, input_node in enumerate(node.input):
            if 'pre_input' == input_node:
                node.input[i] = 'data'
    onnx.checker.check_model(new_model)
    # 修改权重后储存成新的onnx，不然不生效
    onnx.save_model(new_model, model_path)
    print("Done.!")
