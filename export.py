from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch

yolonas = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

shape = (1, 3, 640, 640)
yolonas.prep_model_for_conversion(input_size=shape)
yolonas.heads.export = True

x = torch.randn(shape, device=next(yolonas.parameters()).device)

with torch.no_grad():
    y = yolonas(x)

    mod = torch.jit.trace(yolonas, x)
    mod.save("yolo_nas_s.pt")
