import os.path

from edgeyolo.detect.dotaDetector import DOTADetector

from edgeyolo import EdgeYOLO
from edgeyolo.detect import draw

from glob import glob
import cv2
from tqdm import tqdm

from edgeyolo.data.datasets import get_dataset


if __name__ == '__main__':
    # model = EdgeYOLO(weights="../weights/edgeyolo_visdrone.pth")
    model = EdgeYOLO(weights="../edgeyolo_tiny_dota.pth")
    model.model.fuse()
    print(model.ckpt["ap50_95"])
    nc = len(model.class_names)

    detector = DOTADetector(
        model=model.model,
        nc=nc,
        conf=0.001,
        nms=0.65,
        img_size=1280,
        normal=True,
        twice_nms=False
    )

    dataset = get_dataset("../params/dataset/dota.yaml", mode="val")

    # print(dataset.class_ids)

    show = False

    coco_result = []


    img_list = glob("/dataset/DOTA/val/images/*.png")
    for i in tqdm(range(len(dataset)), ncols=100):

        img = dataset.load_image(i)
        _, _, _, img_id, _ = dataset.pull_item(i)
        img_id = int(img_id)

        result = detector(img)

        if result is not None:
            draw(img, [result], model.class_names, 2, True) if show else None
            coco_result.extend(detector.result2coco_json(result, img_id))


        # print(f"{i+1}/{len(img_list)}, image_id: {img_id}")

        if show:
            cv2.imshow("test", img)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break


    if len(coco_result) > 0:
        from pycocotools.cocoeval import COCOeval
        import contextlib
        import io

        cocoGt = dataset.coco

        cocoDt = cocoGt.loadRes(coco_result)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info = redirect_string.getvalue()

        print(cocoEval.stats[0], cocoEval.stats[1], f"\n{info}")
