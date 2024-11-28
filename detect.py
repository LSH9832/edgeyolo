from multiprocessing import Process, Manager, freeze_support
from datetime import datetime as date
from loguru import logger

from glob import glob

import torch.cuda
import argparse
import cv2
import os
import numpy as np

from edgeyolo.detect import Detector, TRTDetector, draw
from copy import deepcopy


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO Detect parser")

    parser.add_argument("-w", "--weights", type=str, default="edgeyolo_coco.pth", help="weight file")
    parser.add_argument("-c", "--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("-n", "--nms-thres", type=float, default=0.55, help="nms threshold")

    parser.add_argument("--mp", action="store_true", help="use multi-process to accelerate total speed")

    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--no-fuse", action="store_true", help="do not fuse model")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640], help="input size: [height, width]")
    parser.add_argument("-s", "--source", type=str, default=None, help="video source/image dir/rosbag")
    parser.add_argument("--trt", action="store_true", help="is trt model")
    parser.add_argument("--legacy", action="store_true", help="if img /= 255 while training, add this command.")
    parser.add_argument("--use-decoder", action="store_true", help="support original yolox model v0.2.0")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--no-label", action="store_true", help="do not draw label")
    parser.add_argument("--save-dir", type=str, default="./output/detect/imgs/", help="image result save dir")
    parser.add_argument("--fps", type=int, default=99999, help="max fps")

    parser.add_argument("--topic", type=str, default=None, help="ros or rosbag image topic")

    return parser.parse_args()


class ROSBase:

    def __init__(self):
        import numpy as np
        self.np = np

    def _compressedImage2CVMat(self, msg):
        return cv2.imdecode(self.np.frombuffer(msg.data, dtype=self.np.uint8), cv2.IMREAD_COLOR)


    def _Image2CVMat(self, msg):
        return self.np.reshape(self.np.fromstring(msg.data, dtype="bgr8"), [msg.height, msg.width, 3])
    


class ROSBagCapture(ROSBase):

    def __init__(self, bagfile: str, topic: str):
        super(ROSBagCapture, self).__init__()
        from rosbag import Bag
        
        self.bag = Bag(bagfile)
        self.len = self.bag.get_message_count(topic)
        self.iter = self.bag.read_messages(topic)
        self.idx = 0
        self.first = True
        self.compressed = True

    def isOpened(self):
        return self.idx < self.len - 1

    def read(self):
        if self.isOpened():
            topic, msg, timestamp = next(self.iter)
            self.idx += 1
            if self.first:
                self.first = False
                self.compressed = hasattr(msg, "format")
            
            img = (self._compressedImage2CVMat if self.compressed else self._Image2CVMat)(msg)
            return img is not None, img
        else:
            return False, None
        

class ROSCapture(ROSBase):

    def __init__(self, topic):
        super().__init__()
        import rospy
        import rostopic
        from sensor_msgs.msg import CompressedImage, Image
        img_type, *_ = rostopic.get_topic_type(topic)
        self.img = None
        assert img_type.lower().endswith("image")
        self.compressed = img_type.lower().endswith("compressedimage")
        self.updated = False
        rospy.init_node('edgeyolo_detector', anonymous=True)
        rospy.Subscriber(topic, CompressedImage if self.compressed else Image, self.__imageReceivedCallback)

    def __imageReceivedCallback(self, msg):
        self.img = (self._compressedImage2CVMat if self.compressed else self._Image2CVMat)(msg)
        self.updated = True

    def isOpened(self):
        return True
    
    def read(self):
        while not self.updated:
            pass
        self.updated = False
        return True, self.img


def setup_source(args):
    if isinstance(args.source, str) and os.path.isdir(args.source):

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
        if args.source is not None and args.source.lower().endswith(".bag"):
            cmd_check_topic = f"rosbag info {args.source}"
            if args.topic is None:
                str_show = ""
                topics = []
                count = 0
                for line in os.popen(cmd_check_topic).read().split("topics:")[-1].split("\n"):
                    if len(line):
                        contents = line.split()
                        if contents[-1] not in ["sensor_msgs/CompressedImage", "sensor_msgs/Image"]:
                            continue
                        # print(line)
                        count += 1
                        topics.append(contents[0])
                        str_show += f"{count}. " + topics[-1] + "\n"
                        
                logger.error(f"choose one topic from the following topics:\n{str_show[:-1]}")
                idx = -1
                while True:
                    try:
                        idx = int(input(f"input 1~{count} as your choise and then press enter(-1 to exit):\n>>> "))
                        if idx == -1:
                            break
                        if idx > count or idx < 1:
                            assert False
                        args.topic = topics[idx-1]
                        break
                    except:
                        print("wrong input!")
                        pass
                # return
                if idx == -1:
                    exit(0)
            source = ROSBagCapture(args.source, args.topic)
            delay = 0
        elif args.source is None:
            if args.topic is None:
                str_show = ""
                cmd_check_topic = "rostopic list"
                count = 0
                topics = []
                for line in os.popen(cmd_check_topic).read().split("\n"):
                    if len(line):
                        if os.popen(f"rostopic type {line}").read().split("\n")[0] in ["sensor_msgs/CompressedImage", "sensor_msgs/Image"]:
                            count += 1
                            topics.append(line.split()[0])
                            str_show += f"{count}. " + topics[-1] + "\n"

                logger.error(f"choose one topic from the following topics:\n{str_show[:-1]}")
                while True:
                    try:
                        idx = int(input(f"input 1~{count} as your choise and then press enter(-1 to exit):\n>>> "))
                        if idx == -1:
                            break
                        if idx > count or idx < 1:
                            assert False
                        args.topic = topics[idx-1]
                        break
                    except:
                        print("wrong input!")
                        pass
                # return
                if idx == -1:
                    exit(0)
            source = ROSCapture(args.topic)
            delay = 1
        else:
            source = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            delay = 1
    return source, delay


def detect_single(args):
    import time
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
        args.batch = detect.batch_size

    # source loader setup
    source, delay = setup_source(args)

    all_dt = []
    dts_len = 300 // args.batch
    success = True

    # start inference
    count = 0
    t_start = time.time()
    while source.isOpened() and success:

        frames = []
        for _ in range(args.batch):
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

        results = detect(frames, args.legacy)
        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / args.batch:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / args.batch * 1000:.1f}ms", end="      ")

        key = -1

        # [print(result.shape) for result in results]

        imgs = draw(deepcopy(frames), results, detect.class_names, 2, draw_label=not args.no_label)
        # print([im.shape for im in frames])
        for i, img in enumerate(imgs):
            # print(img.shape)
            cv2.imshow("EdgeYOLO result", img)
            count += 1

            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    exist_save_dir = True
                fn = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}"
                file_name = fn + ".jpg"
                # ori_img_name = fn + "_ori.jpg"
                # output_file = fn + ".npy"
                
                cv2.imwrite(os.path.join(args.save_dir, file_name), img)
                # cv2.imwrite(os.path.join(args.save_dir, ori_img_name), frames[i])

                # np.save(os.path.join(args.save_dir, output_file), detect.net_outputs)
                
                logger.info(f"image saved to {file_name}.")
        if key in [ord("q"), 27]:
            # cv2.destroyAllWindows()
            break
    print()
    cv2.destroyAllWindows()
    logger.info(f"\ntotal frame: {count}, total average latency: {(time.time() - t_start) * 1000 / count - 1}ms")


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
    source, delay = setup_source(args)

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
    from time import time
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


def detect_multi(args):

    freeze_support()

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
    opt = get_args()
    (detect_multi if opt.mp else detect_single)(opt)
    print()
