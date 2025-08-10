# bin/detect_one_image --config models/ascend.yaml --source 000000000785.jpg

import os
import requests
import json
import os.path as osp


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
            open(self.currentFile, "wb").write(resp.content)
            self.imgs = self.imgs[1:]
            return True, self.currentFile
        except:
            return False, None
        
if __name__ == "__main__":
    import argparse
    import time
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--path", type=str, required=True)
    
    args = parser.parse_args()
    
    cap = RemoteCapture(args.ip, args.port, args.path)
    if cap.isOpened():
        yaml.dump(
            {"imgs": cap.imgs}, 
            open("path.yaml", "w"), 
            yaml.Dumper
        )
    
    json2save = {}
    
    total_num = cap.length
    
    count = 0
    while cap.isOpened():
        success, real_name = cap.read()
        if not osp.isfile(real_name):
            print("failed to read image!")
            break
        
        jsonfilename = real_name.split(".")[0] + ".json"
        while not osp.isfile(jsonfilename):
            pass
        while True:
            try:
                result = eval(open(jsonfilename).read())
                break
            except:
                print("retry read tmp.json")
                pass
        os.remove(jsonfilename)
        os.remove(real_name)
        
        if osp.isfile(jsonfilename):
            print("failed to delete tmp.json!")
        
        json2save[real_name] = result
        
        count += 1
        print("eval:", f"{count}/{total_num}")
    
    print()
    print("saving json files ...")
    with open("eval_results.json", "w") as f:
        json.dump(json2save, f)
    print("done")
    # result = eval(os.popen(f"bin/detect_one_image --config models/ascend.yaml --source 000000000785.jpg").read().split("\n")[-2])
    # print(result, type(result), type(result[0]))
