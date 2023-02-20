from torch2trt.torch2trt import *
from loguru import logger as lg


__all__ = ["torch2onnx2trt"]


def torch2onnx2trt(module,
                   inputs,
                   input_names=None,
                   output_names=None,
                   log_level=trt.Logger.ERROR,
                   fp16_mode=False,
                   max_workspace_size=1 << 30,
                   strict_type_constraints=False,
                   keep_network=True,
                   int8_mode=False,
                   int8_calib_dataset=None,
                   int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
                   use_onnx=False,
                   default_device_type=trt.DeviceType.GPU,
                   dla_core=0,
                   gpu_fallback=True,
                   device_types=None,
                   min_shapes=None,
                   max_shapes=None,
                   opt_shapes=None,
                   onnx_opset=None,
                   max_batch_size=None,
                   simplify=False,
                   save_onnx=None,
                   save_trt=True,
                   **kwargs):
    """
    rewrite torch2trt
    """
    # capture arguments to provide to context
    kwargs.update(locals())
    kwargs.pop('kwargs')

    if device_types is None:
        device_types = {}

    # handle inputs as dataset of list of tensors
    if issubclass(inputs.__class__, Dataset):
        dataset = inputs
        if len(dataset) == 0:
            raise ValueError('Dataset must have at least one element to use for inference.')
        inputs = dataset[0]
    else:
        dataset = ListDataset()
        dataset.insert(inputs)
        inputs = dataset[0]

    # lg.info(type(dataset))


    outputs = module(*inputs)
    input_flattener = Flattener.from_value(inputs)
    output_flattener = Flattener.from_value(outputs)

    # infer default parameters from dataset

    if min_shapes is None:
        min_shapes_flat = [tuple(t) for t in dataset.min_shapes(flat=True)]
    else:
        min_shapes_flat = input_flattener.flatten(min_shapes)

    if max_shapes is None:
        max_shapes_flat = [tuple(t) for t in dataset.max_shapes(flat=True)]
    else:
        max_shapes_flat = input_flattener.flatten(max_shapes)

    if opt_shapes is None:
        opt_shapes_flat = [tuple(t) for t in dataset.median_numel_shapes(flat=True)]
    else:
        opt_shapes_flat = input_flattener.flatten(opt_shapes)

    # handle legacy max_batch_size
    if max_batch_size is not None:
        min_shapes_flat = [(1,) + s[1:] for s in min_shapes_flat]
        max_shapes_flat = [(max_batch_size,) + s[1:] for s in max_shapes_flat]

    dynamic_axes_flat = infer_dynamic_axes(min_shapes_flat, max_shapes_flat)

    if default_device_type == trt.DeviceType.DLA:
        for value in dynamic_axes_flat:
            if len(value) > 0:
                raise ValueError('Dataset cannot have multiple shapes when using DLA')
    if save_trt:
        logger = trt.Logger(log_level)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

    if input_names is None:
        input_names = default_input_names(input_flattener.size)
    if output_names is None:
        output_names = default_output_names(output_flattener.size)

    if use_onnx:

        import onnx

        module_flat = Flatten(module, input_flattener, output_flattener)
        inputs_flat = input_flattener.flatten(inputs)

        f = io.BytesIO()
        try:
            torch.onnx.export(
                module_flat,
                inputs_flat,
                f,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    name: {int(axis): 'axis_%d' % axis for axis in dynamic_axes_flat[index]}
                    for index, name in enumerate(input_names)
                },
                opset_version=onnx_opset
            )
        except:
            torch.onnx.export(
                module,
                tuple(inputs),
                f,
                input_names=input_names,
                output_names=output_names,
                opset_version=onnx_opset
            )
        f.seek(0)

        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        if simplify:
            try:
                import onnxsim
                lg.info('start to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
                lg.info('simplified ONNX successfully.')
            except Exception as e:
                lg.error(f'Simplifier failure: {e}')

        if save_onnx is not None:
            onnx.save(onnx_model, save_onnx)
            lg.info(f"onnx file saved to {save_onnx}")

        if not save_trt:
            return None
        
        f = io.BytesIO()
        try:
            import onnx_graphsurgeon as gs
            onnx_graph = gs.import_onnx(onnx_model)
            onnx_graph.fold_constants().cleanup()
            onnx.save(gs.export_onnx(onnx_graph), f)
        except:
            onnx.save(onnx_model, f)

        f.seek(0)

        onnx_bytes = f.read()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        parser.parse(onnx_bytes)

    else:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        with ConversionContext(network, torch2trt_kwargs=kwargs, builder_config=config, logger=logger) as ctx:

            inputs_flat = input_flattener.flatten(inputs)

            ctx.add_inputs(inputs_flat, input_names, dynamic_axes=dynamic_axes_flat)

            outputs = module(*inputs)

            outputs_flat = output_flattener.flatten(outputs)
            ctx.mark_outputs(outputs_flat, output_names)

    # set max workspace size
    config.max_workspace_size = max_workspace_size

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    config.default_device_type = default_device_type
    if gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.DLA_core = dla_core

    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    if int8_mode:

        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = dataset

        config.set_flag(trt.BuilderFlag.INT8)

        # Making sure not to run calibration with QAT mode on
        if 'qat_mode' not in kwargs:

            calibrator = DatasetCalibrator(
                int8_calib_dataset, algorithm=int8_calib_algorithm
            )
            config.int8_calibrator = calibrator

    # OPTIMIZATION PROFILE
    profile = builder.create_optimization_profile()
    for index, name in enumerate(input_names):
        profile.set_shape(
            name,
            min_shapes_flat[index],
            opt_shapes_flat[index],
            max_shapes_flat[index]
        )
    config.add_optimization_profile(profile)

    if int8_mode:
        config.set_calibration_profile(profile)

    # BUILD ENGINE

    engine = builder.build_engine(network, config)

    try:
        module_trt = TRTModule(engine, input_names, output_names, input_flattener=input_flattener,
                               output_flattener=output_flattener)
    except:
        module_trt = TRTModule(engine, input_names, output_names)

    if keep_network:
        module_trt.network = network

    return module_trt
