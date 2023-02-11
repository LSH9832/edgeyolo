from edgeyolo import get_model_info, EdgeYOLO


model = EdgeYOLO("../params/model/edgeyolo_coco_repconv_tiny.yaml").model
# model = EdgeYOLO("../params/model/edgeyolo_coco_repconv_s.yaml").model
print(get_model_info(model, (640, 640)))
model.reparameterize()
print(get_model_info(model, (640, 640)))
