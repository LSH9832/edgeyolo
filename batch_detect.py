from multiprocessing import Process, Manager, freeze_support
from datetime import datetime as date
from loguru import logger
from time import time
from glob import glob

import torch.cuda
import argparse
import cv2
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO Detect parser")
    parser.add_argument("-w", "--weights", type=str, default="edgeyolo_coco.pth", help="weight file")
    parser.add_argument("-c", "--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("-n", "--nms-thres", type=float, default=0.45, help="nms threshold")
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--no-fuse", action="store_true", help="do not fuse model")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640], help="input size: [height, width]")
    parser.add_argument("-s", "--source", type=str, default="E:/videos/test.avi", help="video source or image dir")
    parser.add_argument("--trt", action="store_true", help="is trt model")
    parser.add_argument("--legacy", action="store_true", help="if img /= 255 while training, add this command.")
    parser.add_argument("--use-decoder", action="store_true", help="support original yolox model v0.2.0")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--no-label", action="store_true", help="do not draw label")
    parser.add_argument("--save-dir", type=str, default="./imgs/coco", help="image result save dir")
    parser.add_argument("--fps", type=int, default=99999, help="max fps")

    return parser.parse_args()


def inference(msg, results, args):
    from edgeyolo.detect import Detector, TRTDetector
    detector = TRTDetector if args.trt else Detector
    detect = detector(
        weight_file=args.weights,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        input_size=args.input_size,
        fuse=not args.no_fuse,
        fp16=args.fp16,
        use_decoder=args.use_decoder
    )
    if args.trt:
        args.batch = detect.batch_size

    # source loader setup
    if os.path.isdir(args.source):

        class DirCapture:

            def __init__(self, dir_name):
                self.imgs = []
                for img_type in ["jpg", "png", "jpeg", "bmp", "webp"]:
                    self.imgs += sorted(glob(os.path.join(dir_name, f"*.{img_type}")))

            def isOpened(self):
                return bool(len(self.imgs))

            def read(self):
                print(self.imgs[0])
                now_img = cv2.imread(self.imgs[0])
                self.imgs = self.imgs[1:]
                return now_img is not None, now_img

        source = DirCapture(args.source)
        delay = 0
    else:
        source = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        delay = 1

    msg["class_names"] = detect.class_names
    msg["delay"] = delay

    success = True
    while source.isOpened() and success and not msg["end"]:

        frames = []
        for _ in range(args.batch):
            if msg["end"]:
                frames = []
                break
            success, frame = source.read()
            if not success:
                if not len(frames):
                    cv2.destroyAllWindows()
                    break
                else:
                    while len(frames) < args.batch:
                        frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        results.put((frames, [r.cpu() for r in detect(frames, args.legacy)]))

    msg["end"] = True
    torch.cuda.empty_cache()
    msg["end_count"] += 1


def draw_imgs(msg, results, all_imgs, args):
    from edgeyolo.detect import draw

    while "class_names" not in msg:
        pass
    class_names = msg["class_names"]

    while not msg["end"] or not results.empty():
        # print(len(msg["results"]))
        if not results.empty():
            for img in draw(*results.get(), class_names, 2, draw_label=not args.no_label):
                all_imgs.put(img)
                # print(all_imgs.empty())
    torch.cuda.empty_cache()
    msg["end_count"] += 1


def show(msg, all_imgs, args, pid):
    # import platform
    while "delay" not in msg:
        pass
    delay = msg["delay"]
    exist_save_dir = os.path.isdir(args.save_dir)

    all_dt = []
    t0 = time()
    while not msg["end"] or not all_imgs.empty():
        if not all_imgs.empty():
            img = all_imgs.get()
            # print(img.shape)
            while time() - t0 < 1. / args.fps - 0.0004:
                pass

            dt = time() - t0
            all_dt.append(dt)
            if len(all_dt) > 300:
                all_dt = all_dt[-300:]

            mean_dt = sum(all_dt) / len(all_dt) * 1000
            print(f"\r{dt * 1000:.1f}ms --> {1. / dt:.1f}FPS, "
                  f"average:{mean_dt:.1f}ms --> {1000. / mean_dt:.1f}FPS", end="      ")

            t0 = time()
            cv2.imshow("EdgeYOLO result", img)
            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                msg["end"] = True
                cv2.destroyAllWindows()
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                file_name = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, file_name), img)
                logger.info(f"image saved to {file_name}.")

    print()
    print()
    torch.cuda.empty_cache()
    msg["end_count"] += 1

    while not msg["end_count"] == 3:
        pass

    # if platform.system().lower() == "windows":
    #     os.system(F"taskkill /F /PID {pid}")
    # else:
    #     os.system(f"kill -9 {pid}")


def main():

    args = get_args()

    shared_data = Manager().dict()
    shared_data["end"] = False
    shared_data["end_count"] = 0

    results = Manager().Queue()
    all_imgs = Manager().Queue()


    processes = [Process(target=inference, args=(shared_data, results, args)),
                 Process(target=draw_imgs, args=(shared_data, results, all_imgs, args)),
                 Process(target=show, args=(shared_data, all_imgs, args, os.getpid()))]

    [process.start() for process in processes]
    torch.cuda.empty_cache()
    [process.join() for process in processes]


if __name__ == '__main__':
    freeze_support()
    main()
