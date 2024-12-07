from yolo import YOLO
from yolo.capture import setup_source
from yolo.visualize import draw
import yaml
import os.path as osp
import argparse
import cv2
from time import time

# import platform
# print(platform.machine())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="models/agv.yaml")
    parser.add_argument("-s", "--source", type=str, required=True)
    parser.add_argument("-t", "--topic", type=str, default=None)

    return parser.parse_args()


def main():

    args = get_args()

    cfg = yaml.load(open(args.config), yaml.Loader)
    model_path = osp.join(osp.dirname(args.config), cfg["model_path"])
    # print(model_path)
    
    model = YOLO(
        model_path,
        cfg["names"],
        cfg.get("input_name", "input_0"),
        cfg.get("output_names", ["output_0"]),
        cfg["img_size"][1],
        cfg["img_size"][0],
        cfg.get("normalize", False),
        cfg.get("strides", [8, 16, 32]),
        cfg.get("device", 0),
        **cfg.get("kwargs", {})
    )

    print(f"\033[32m\033[1m[INFO] acceleration platform: {model.platform()}\033[0m")
    

    # print(model.length_array, model.num_classes)

    cap, delay = setup_source(args)
    try:
        while cap.isOpened():

            success, frame = cap.read()
            if not success:
                break

            if not all(frame.shape):
                break
            
            t0 = time()
            result = model.infer(frame)
            print(f"\rinfer delay: {(time()-t0) * 1000:.3f}ms, num result={len(result)}", end="             ")

            img_show = draw(frame, [result], model.names, 2)

            cv2.imshow("test", img_show)
            key = cv2.waitKey(delay)
            if key == ord(" "):
                delay = 1 - delay
            elif key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        model.release()
        print()


if __name__ == "__main__":
    main()
