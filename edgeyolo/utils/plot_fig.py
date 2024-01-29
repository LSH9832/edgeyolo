from matplotlib import pyplot as plt
import yaml
import os.path as osp
import os
from loguru import logger


DEFAULT_SIZE = (16, 7)


def DEFAULT_ADJUST(size=DEFAULT_SIZE):
    return {
        "left": 0.06 * 16 / size[0],
        "right": 1 - 0.02 * 16 / size[0],
        "top": 1 - 0.03 * 7 / size[1],
        "bottom": 0.08 * 7 / size[1]
    }


def plot(file, plot_type="loss", show=True, figsize=DEFAULT_SIZE, save=False, suffix="pdf"):
    assert osp.isfile(file), f"{file} does not exist!"
    if isinstance(suffix, str):
        suffix = [suffix]
    epochs = []
    losses = {
        "total loss": [],
        "iou loss": [],
        "conf loss": [],
        "class loss": [],
        "l1 loss": []
    }
    lrs = []
    for line in open(file, encoding="utf8").read().split("\n"):
        if "epoch:" in line and "iter:" in line and "loss:" in line:
            if "l1:" not in line:
                losses["l1 loss"].append(0.)
            now_epoch = 0
            for element in line.split("- ")[-1].split():
                if element.startswith("epoch:"):
                    now_epoch = float(element.split(":")[-1].split("/")[0])
                if element.startswith("iter:"):
                    point = float(eval(element.split(":")[-1]))
                    now_value = now_epoch - 1 + point
                    if len(epochs) and now_value <= epochs[-1]:
                        idx = 0
                        while epochs[idx-1] >= now_value:
                            idx -= 1
                        epochs = epochs[:idx]
                        for k in losses.keys():
                            losses[k] = losses[k][:idx]
                        lrs = lrs[:idx]

                    epochs.append(now_value)

                if plot_type == "loss":
                    if "total:" in element:
                        losses["total loss"].append(float(element.split(":")[-1]))
                    if "iou:" in element:
                        losses["iou loss"].append(float(element.split(":")[-1]))
                    if "conf:" in element:
                        losses["conf loss"].append(float(element.split(":")[-1]))
                    if "cls:" in element:
                        losses["class loss"].append(float(element.split(":")[-1][:-1]))
                    if "l1:" in element:
                        losses["l1 loss"].append(float(element.split(":")[-1]))
                elif plot_type == "lr":
                    if "lr:" in element:
                        lrs.append(float(eval(element.split(":")[-1])))

    if plot_type == "loss":
        fig = plt.figure("loss", figsize=figsize)
        plt.subplots_adjust(**DEFAULT_ADJUST(figsize))
        for k, v in losses.items():
            if any(v):
                plt.plot(epochs, v, label=k)
        plt.ylabel("loss")
    elif plot_type == "lr":
        fig = plt.figure("learning rate", figsize=figsize)
        plt.subplots_adjust(**DEFAULT_ADJUST(figsize))
        plt.plot(epochs, lrs, label="learning rate")
        plt.ylabel("learning rate")
    plt.legend(framealpha=1)
    plt.grid(True)
    plt.xlabel("epoch")

    if show:
        plt.show()

    if save:
        path = osp.join(osp.dirname(file), "figures")
        os.makedirs(path, exist_ok=True)
        for s in suffix:
            fig.savefig(osp.join(path, f'{plot_type}.{s}'), format=s, dpi=600)


def plot_ap(file, show=True, figsize=DEFAULT_SIZE, save=False, suffix="pdf"):
    assert osp.isfile(file), f"{file} does not exist!"
    if isinstance(suffix, str):
        suffix = [suffix]
    data: dict = yaml.load(open(file).read(), yaml.SafeLoader)
    epoch = [0]
    ap_50 = [0.]
    ap_50_95 = [0.]

    for k, v in data.items():
        epoch.append(k + 1)
        ap_50.append(v["ap50"])
        ap_50_95.append(v["ap50_95"])

    fig = plt.figure("eval result", figsize=figsize)
    plt.subplots_adjust(**DEFAULT_ADJUST(figsize))
    plt.plot(epoch, ap_50, "b")
    plt.plot(epoch, ap_50_95, "r")
    plt.legend(["ap50", "ap50:95"], framealpha=1)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("AP")

    if save:
        path = osp.join(osp.dirname(file), "figures").replace("\\", "/")
        os.makedirs(path, exist_ok=True)
        for s in suffix:
            fig.savefig(osp.join(path, f'ap.{s}'), format=s, dpi=600)

    if show:
        plt.show()


def plot_all(path, show=False, figsize=DEFAULT_SIZE, save=False, suffix="pdf"):
    if save:
        _p = osp.join(path, 'figures').replace('\\', '/')
        logger.info(f"figs will be saved to {_p}, please wait for seconds and see")
    try:
        plot(osp.join(path, "log.txt"), plot_type="loss", show=False, save=save, suffix=suffix, figsize=figsize)
        plot(osp.join(path, "log.txt"), plot_type="lr", show=False, save=save, suffix=suffix, figsize=figsize)
        plot_ap(osp.join(path, "eval.yaml"), show=False, save=save, suffix=suffix, figsize=figsize)
    except Exception as e:
        logger.error(f"plot error: {e}")

    if show:
        plt.show()
