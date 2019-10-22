with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open('resnet100.onnx', 'rb') as model:
        parser.parse(model.read())
