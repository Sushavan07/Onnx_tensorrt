import tensorrt as trt
TRT_LOGGER=TRT.Logger(trt.Logger.WARNING)
builder.max_batch_size=max_batch_size
builder.max_workspace_size=1<<20
with trt.Builder(TRT_LOGGER) as builder,builder.create_builder_config()as config,builder.build_cuda_engine(network,config) as engine:
    h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)
    h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
with engine.create_execution_context() as context:
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
return h_output
with builder=trt.Builder(TRT_LOGGER) as builder,builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open('resnet100.onnx','rb') as model:
        parser.parse(model.read())


        
