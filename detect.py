from datetime import datetime as date
from glob import glob
import os
from loguru import logger
import argparse
import cv2

from edgeyolo.detect import Detector, TRTDetector, draw


if __name__ == '__main__':

    parser = argparse.ArgumentParser("EdgeYOLO Detect parser")
    parser.add_argument("-w", "--weights", type=str, default="edgeyolo_coco.pth", help="weight file")
    parser.add_argument("-c", "--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("-n", "--nms-thres", type=float, default=0.55, help="nms threshold")
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--no-fuse", action="store_true", help="do not fuse model")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640], help="input size: [height, width]")
    parser.add_argument("-s", "--source", type=str, default="./test.avi", help="video source or image dir")
    parser.add_argument("--trt", action="store_true", help="is trt model")
    parser.add_argument("--legacy", action="store_true", help="if img /= 255 while training, add this command.")
    parser.add_argument("--use-decoder", action="store_true", help="support original yolox model v0.2.0")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--no-label", action="store_true", help="do not draw label")
    parser.add_argument("--save-dir", type=str, default="./imgs/coco", help="image result save dir")

    args = parser.parse_args()
    exist_save_dir = os.path.isdir(args.save_dir)

    # detector setup
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
        args.batch_size = detect.batch_size

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
    
    all_dt = []
    dts_len = 300 // args.batch_size
    success = True

    # start inference
    while source.isOpened() and success:

        frames = []
        for _ in range(args.batch_size):
            success, frame = source.read()
            if not success:
                if not len(frames):
                    cv2.destroyAllWindows()
                    break
                else:
                    while len(frames) < args.batch_size:
                        frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        results = detect(frames, args.legacy)
        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / args.batch_size:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / args.batch_size * 1000:.1f}ms", end="      ")

        key = -1
        imgs = draw(frames, results, detect.class_names, 2, draw_label=not args.no_label)
        # print([im.shape for im in frames])
        for img in imgs:
            # print(img.shape)
            cv2.imshow("EdgeYOLO result", img)
            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                file_name = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, file_name), img)
                logger.info(f"image saved to {file_name}.")
        if key in [ord("q"), 27]:
            cv2.destroyAllWindows()
            break
