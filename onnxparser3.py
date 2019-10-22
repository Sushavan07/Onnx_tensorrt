import os
import tensorrt as trt
TRT_LOGGER=trt.Logger()
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder,builder.create_network() as network, trt.OnnxParser(network,TRT_LOGGER) as parser:
        parser.parse(model.read())
