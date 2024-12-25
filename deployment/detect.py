from yolo import YOLO
from yolo.capture import setup_source
from yolo.visualize import draw
import yaml
import os.path as osp
import argparse
import cv2
from time import time, sleep
from threading import Thread

# import platform
# print(platform.machine())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="models/agv.yaml")
    parser.add_argument("-s", "--source", type=str, required=True)
    parser.add_argument("-t", "--topic", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--dist", type=str, default="output.mp4")
    parser.add_argument("--fps", type=int, default=-1)

    return parser.parse_args()






def main():

    args = get_args()

    cfg = yaml.load(open(args.config), yaml.Loader)
    model_path = osp.join(osp.dirname(args.config), cfg["model_path"])
    show = not args.no_show
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
    if not cap.isOpened():
        print("failed to open capture")
        exit(-1)
    if not show:
        fps = args.fps if args.fps > 0 else cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = 30
        imgw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        imgh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # writer = cv2.VideoWriter(args.dist, cv2.VideoWriter_fourcc(*'mp4v'), fps, (imgw, imgh), True)
        

        imgs_list = []
        

        def writeVideo(*wargs):

            writer = cv2.VideoWriter(*wargs)
            if not writer.isOpened():
                print("video writer is not opened")
                return
            
            while True:
                if len(imgs_list):
                    img = imgs_list.pop(0)
                    if img is None:
                        writer.release()
                        return
                    else:
                        writer.write(img)
                else:
                    sleep(0.1)
        
        writer_thread = Thread(target=writeVideo, args=(args.dist, cv2.VideoWriter_fourcc(*'mp4v'), fps, (imgw, imgh), True))
        writer_thread.start()
            
        
    try:
        while cap.isOpened():

            success, frame = cap.read()
            if not success:
                break

            if not all(frame.shape):
                break
            
            t0 = time()
            result = model.infer(frame)
            print(f"\rinfer delay: {(time()-t0) * 1000:.3f}ms, num result={len(result)} ", end="             ")

            img_show = draw(frame, [result], model.names, 2)

            if show:
                cv2.imshow("test", img_show)
                key = cv2.waitKey(delay)
                if key == ord(" "):
                    delay = 1 - delay
                elif key == 27:
                    break
            else:
                imgs_list.append(img_show)
    finally:
        cv2.destroyAllWindows()
        model.release()
        print()
        if not show:
            print("wait", len(imgs_list), "images to write to ", args.dist)
            imgs_list.append(None)
            writer_thread.join()
        


if __name__ == "__main__":
    main()
