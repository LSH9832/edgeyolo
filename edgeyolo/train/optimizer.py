from torch import nn
from torch.optim import SGD, Adam
from loguru import logger

OPTIMIZER = {
    "sgd": SGD,
    "adam": Adam
}


def get_optimizer(
    model,
    lr,
    momentum=0.9,
    weight_decay=5e-4,
    train_backbone=True,
    head_layer_num=0,
    optimizer_type="SGD"
) -> SGD or Adam:
    optimizer_type = optimizer_type.lower()
    if optimizer_type.lower() not in ["sgd", "adam"]:
        logger.error(f"no optimizer type named {optimizer_type}, use default optimizer SGD.")
        optimizer_type = "sgd"
    # head_layer_num = -1
    # logger.info(f"start layer: {head_layer_num}")
    is_head = False
    if not train_backbone:
        for k, m in model.named_modules():
            ks = k.split(".")
            if len(ks) == 2 and ks[-1].isdigit():
                is_head = int(ks[-1]) >= head_layer_num
            else:
                continue
            for v in m.parameters():
                v.requires_grad = is_head

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    
    for k, v in model.named_modules():
        if "." not in k:
            continue
        
        _, layer_num, *_ = k.split(".")

        if not train_backbone:
            if int(layer_num) < head_layer_num:
                continue

        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.BatchNorm1d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    # import time
    # print(len(pg0), len(pg1), len(pg2))
    # time.sleep(10)
    if optimizer_type == "sgd":
        usual_params = {"lr": lr, "momentum": momentum, "nesterov": True}
    else:
        usual_params = {"lr": lr, "betas": (momentum, 0.999)}
    group_lists = [
        {"params": pg0}, 
        {"params": pg1, "weight_decay": weight_decay},
        {"params": pg2}
    ]
    optimizer = None
    for params_group in group_lists:
        if optimizer is None:
            if not len(params_group["params"]):
                continue
            optimizer = OPTIMIZER[optimizer_type](**params_group, **usual_params)
        else:
            optimizer.add_param_group(params_group)
    # optimizer = SGD(pg0, lr=lr, momentum=momentum, nesterov=True)
    # optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay}) 
    # optimizer.add_param_group({"params": pg2})
    assert optimizer is not None
    return optimizer
