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
    
    parser.add_argument("--eval", action="store_true", help="evaluate from source")
    
    parser.add_argument("--number", type=int, default=-1, help="just test number of images and then exit, set -1 to disable, default is -1")
    
    
    parser.add_argument("-t", "--topic", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--dist", type=str, default="output.mp4")
    parser.add_argument("--fps", type=int, default=-1)
    
    # for remote image
    parser.add_argument("--ip", type=str, default="")
    parser.add_argument("--port", type=int, default=12345)

    return parser.parse_args()






def main():

    args = get_args()

    cfg = yaml.load(open(args.config), yaml.Loader)
    model_path = osp.join(osp.dirname(args.config), cfg["model_path"])
    show = not args.no_show and not args.eval
    # print(model_path)
    
    eval_mode = args.eval
    max_number = args.number
    
    kwargs = cfg.get("kwargs", {})
    if eval_mode:
        json_results = {}
        kwargs["conf_threshold"] = 0.001
        kwargs["nms_threshold"] = 0.65
        kwargs["show_info"] = False
        if not len(args.ip):
            assert osp.isdir(args.source), "when use eval mode, image source should be a dir with images for some dataset"
    
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
        **kwargs
    )

    print(f"\033[32m\033[1m[INFO] Acceleration platform: {model.platform()}\033[0m")
    

    # print(model.length_array, model.num_classes)

    cap, delay = setup_source(args)
    if not cap.isOpened():
        print("failed to open capture")
        exit(-1)
    if not show and not eval_mode:
        try:
            fps = args.fps if args.fps > 0 else cap.get(cv2.CAP_PROP_FPS)
            if fps < 1:
                fps = 30
            imgw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            imgh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except:
            fps = 30
            imgw = 640
            imgh = 640
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
            
    elif eval_mode:
        num_imgs = cap.length
    try:
        count = 0
        while cap.isOpened():

            success, frame = cap.read()
            if not success:
                break

            if not all(frame.shape):
                break
            
            count += 1
            if count == max_number:
                break
            t0 = time()
            result = model.infer(frame)
            if eval_mode:
                print(f"\reval: {count}/{num_imgs}", end="    ")
            else:
                print(f"\rinfer delay: {(time()-t0) * 1000:.3f}ms, num result={len(result)} ", end="             ")

            if not eval_mode:
                img_show = draw(frame, [result], model.names, 2)

            if show:
                cv2.imshow("test", img_show)
                key = cv2.waitKey(delay)
                if key == ord(" "):
                    delay = 1 - delay
                elif key == 27:
                    break
            elif not eval_mode:
                imgs_list.append(img_show)
            else:
                if json_results.get(cap.currentFile, None) is None:
                    json_results[cap.currentFile] = []
                for x1, y1, x2, y2, cls, conf in result:
                    json_results[cap.currentFile].append({
                        "category_id": int(cls),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(conf),
                        "segmentation": [],
                    })
                
                
                
    finally:
        cv2.destroyAllWindows()
        model.release()
        print()
        if not show and not eval_mode:
            print("wait", len(imgs_list), "images to write to ", args.dist)
            imgs_list.append(None)
            writer_thread.join()
        elif eval_mode:
            import json
            with open("eval_results.json", "w") as f:
                json.dump(json_results, f)
        


if __name__ == "__main__":
    main()
