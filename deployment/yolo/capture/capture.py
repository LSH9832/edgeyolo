import cv2
import os
import os.path as osp
from glob import glob


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
    if hasattr(args, "ip") and hasattr(args, "port") and len(args.ip):
        import requests
        import json
        from PIL import Image
        import io
        import numpy as np
        
        
        class RemoteCapture:
            def __init__(self, ip, port, path):
                self.ip = ip
                self.port = port
                self.path = path
                self.length = 0
                self.__request_url = f"http://{ip}:{port}/getImage"
                
                resp = requests.post(
                    f"http://{ip}:{port}/getImageList",
                    json={
                        "path": path
                    }
                )
                
                result = json.loads(resp.content)
                if result.get("success", False):
                    self.imgs: list = result.get("value")
                    self.length = len(self.imgs)
                
                self.currentFile = None
            
            def isOpened(self):
                return bool(len(self.imgs))
            
            
            def read(self):
                self.currentFile = self.imgs[0]
                resp = requests.post(
                    self.__request_url,
                    json={
                        "path": osp.join(self.path, self.currentFile)
                    }
                )
                try:
                    f = io.BytesIO(resp.content)
                    image = np.array(Image.open(f))
                    if len(image.shape) == 2:
                        # print(image.shape)
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        # cv2.imshow("debug", image)
                        # cv2.waitKey(0)
                    else:
                        image = np.ascontiguousarray(image[..., ::-1])
                    self.imgs = self.imgs[1:]

                    return image is not None, image
                except:
                    return False, None
        try:
            source = RemoteCapture(args.ip, args.port, args.source)
            delay = 0
        except:
            print("failed to connect remote")
            raise
            
    elif isinstance(args.source, str) and os.path.isdir(args.source):

        class DirCapture:

            def __init__(self, dir_name):
                self.imgs = []
                for img_type in ["jpg", "png", "jpeg", "bmp", "webp"]:
                    self.imgs += sorted(glob(os.path.join(dir_name, f"*.{img_type}")))
                self.length = len(self.imgs)
                self.currentFile = None

            def isOpened(self):
                return bool(len(self.imgs))

            def read(self):
                self.currentFile = os.path.basename(self.imgs[0])
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
                        
                print(f"\033[33m\033[1m[WARN] choose one topic from the following topics:\n\033[0m\033[33m{str_show[:-1]}\033[0m")
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

                print(f"\033[33m\033[1m[WARN] choose one topic from the following topics:\n\033[0m\033[33m{str_show[:-1]}\033[0m")
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
