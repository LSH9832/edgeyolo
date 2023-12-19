from torch import nn
from torch.optim import SGD, Adam
from loguru import logger

OPTIMIZER = {
    "sgd": SGD,
    "adam": Adam
}

def get_optimizer(
    model: nn.Module,
    lr: float,
    momentum: float=0.9,
    weight_decay: float=5e-4,
    train_backbone: bool=True,
    head_layer_num: int=0,
    optimizer_type: str="SGD",
    rank=0,
):
    
    optimizer_type = optimizer_type.lower()
    if optimizer_type.lower() not in ["sgd", "adam"]:
        if rank == 0:
            logger.error(f"no optimizer type named {optimizer_type}, use default optimizer SGD.")
        optimizer_type = "sgd"
    # head_layer_num = -1
    # logger.info(f"start layer: {head_layer_num}")
    model.train()
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
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    print("add im")
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    print("add ia")
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if optimizer_type.lower() == "adam":
        optimizer = Adam(pg0, lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(pg0, lr=lr, momentum=momentum, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))

    if rank == 0:
        logger.info(f"length of PG0(BN weights):    {len(pg0)}")
        logger.info(f"length of PG1(other weights): {len(pg1)}")
        logger.info(f"length of PG2(bias):          {len(pg2)}")
    del pg0, pg1, pg2

    assert optimizer is not None

    return optimizer


def get_optimizer_old(
    model,
    lr,
    momentum=0.9,
    weight_decay=5e-4,
    train_backbone=True,
    head_layer_num=0,
    optimizer_type="SGD",
    **kwargs
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
