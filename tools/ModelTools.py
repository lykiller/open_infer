# -*-coding:utf-8-*-
import onnx
import torch


def check_update_model(_model):
    if _model.graph.name == '':
        _model.graph.name = 'tackle_checker_error'

    try:
        onnx.checker.check_model(_model)
    except onnx.checker.ValidationError as e:
        print("model is invalid: %s" % (e))
        # 目前只考虑这个错误“ Nodes in a graph must be topologically sorted, however input ...”，
        # 就是模型的节点顺序有问题，onnxruntime 是可以推断的，为了其他引擎比如 trt可以推断，调整一下节点顺序。
    else:
        return _model

    print('update model ...')
    _new_model = onnx.ModelProto(ir_version=_model.ir_version,
                                 producer_name=_model.producer_name,
                                 producer_version=_model.producer_version,
                                 opset_import=_model.opset_import)

    _new_model.graph.name = _model.graph.name
    # 只考虑模型是单输入单输出的情况。
    # 这里不需要 copy.deepcopy(model.graph.input[0]), python 是写时copy。
    _new_model.graph.input.append(_model.graph.input[0])
    _new_model.graph.output.append(_model.graph.output[0])

    list_total_input_name = [_model.graph.input[0].name]
    for weight in _model.graph.initializer:
        _new_model.graph.initializer.append(weight)
        list_total_input_name.append(weight.name)

    list_nodes_invalid = []
    for _node in _model.graph.node:
        node_invalid = False
        for node_input_name in _node.input:
            if list_total_input_name.count(node_input_name) == 0:
                list_nodes_invalid.append(_node)
                node_invalid = True
                break
        if node_invalid:
            continue
        for node_output_name in _node.output:
            list_total_input_name.append(node_output_name)
        _new_model.graph.node.append(_node)

    for _node in list_nodes_invalid:
        for node_input_name in _node.input:
            assert list_total_input_name.count(node_input_name) > 0
        for node_output_name in _node.output:
            list_total_input_name.append(node_output_name)
        _new_model.graph.node.append(_node)

    return _new_model
