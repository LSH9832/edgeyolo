from flask import *
import os.path as osp
from glob import iglob
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=12345)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


app = Flask(__name__)


@app.route("/getImageList", methods=["POST"])
def getImageList():
    cfg = request.json
    
    ret = {
        "success": False,
        "value": []
    }
    ret["success"] = osp.isdir(cfg.get("path", ""))
    if ret["success"]:
        img_dir_path = cfg.get("path")
        image_list = []
        for suffix in ["jpg", "png", "jpeg", "bmp", "webp"]:
            for iname in iglob(osp.join(img_dir_path, f"*.{suffix}")):
                image_list.append(osp.basename(iname))
        image_list.sort()
        ret["value"] = image_list

    
    # print(cfg)
    
    return ret


@app.route("/getImage", methods=["POST"])
def getImage():
    cfg = request.json
    
    ret = {
        "success": False,
        "value": None
    }
    ret["success"] = osp.isfile(cfg.get("path", ""))
    if ret["success"]:
        return send_file(cfg["path"])
    
    return ret



if __name__ == "__main__":
    args = get_args()
    app.run("0.0.0.0", args.port, args.debug)

