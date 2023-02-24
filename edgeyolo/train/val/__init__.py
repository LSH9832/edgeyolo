from .coco_evaluator import COCOEvaluator as Evaluator
from .dota_evaluator import DOTAEvaluator


evaluators = {
    "voc": Evaluator,
    "coco": Evaluator,
    "visdrone": Evaluator,
    "yolo": Evaluator,
    "dota": DOTAEvaluator
}
