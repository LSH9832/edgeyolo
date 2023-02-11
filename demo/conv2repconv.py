import time

from edgeyolo import EdgeYOLO, get_model_info, NoPrint
import tqdm


def conv2repconv(model: EdgeYOLO, repmodel: EdgeYOLO):
    ori_model = model.model
    dist_model = repmodel.model

    params = {}
    ori_params = ori_model.state_dict()
    for k, v in dist_model.state_dict().items():
        if k in ori_params:
            v_ = ori_params[k]

        elif "rbr_dense" in k:
            ori_k = k.replace("rbr_dense.0", "conv").replace("rbr_dense.1", "bn")
            if ori_k in ori_params:
                v_ = ori_params[ori_k]
            else:
                print(f"no relative params for repconv params:{k}")
                v_ = v
        else:
            print(f"new layer {k}")
            v_ = v

        if v.shape == v_.shape:
            params[k] = v_
        else:
            print(f"size not match! shape is {v.shape} while input shape is {v_.shape}")

    repmodel.model.load_state_dict(params, strict=False)
    repmodel.now_epoch = -1
    repmodel.class_names = model.class_names
    return repmodel


if __name__ == '__main__':
    import torch

    # model = conv2repconv(
    #     EdgeYOLO(weights="./edgeyolo_coco.pth"),
    #     EdgeYOLO(cfg_file="./params/model/edgeyolo_coco_repconv.yaml")
    # )
    #
    # model.save("edgeyolo_rep_coco.pth")

    for i in tqdm.tqdm(range(100), ncols=200):
        time.sleep(0.2)

    # print(get_model_info(my_model.model, (640, 640)))
    # with NoPrint():
    #     my_model.model.fuse()
    # print(get_model_info(my_model.model, (640, 640)))



# model.84.conv.weight
# model.84.bn.weight
# model.84.bn.bias