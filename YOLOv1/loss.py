from xml.sax.saxutils import prepare_input_source
import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse(nn.MSELoss(reduction="sum")) # original paper use sum instead of avg.
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord= 5
    
    def forward(self, predictions, target):
        predictions = predictions(-1, self.S, self.S, self.C + self.B * 5)
        
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0) # bestbox will be the argmax one
        exists_box = target[..., 20].unsqueeze(3) # Iobj i: Is there an object in cell i

        # =================== #
        # FOR BOX COORDINATES #
        # =================== #
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30] + (1-best_box) * predictions[..., 21:25]
        )

        # =============== #
        # FOR OBJECT LOSS #
        # =============== #



        # ================== #
        # FOR NO OBJECT LOSS #
        # ================== #



        # ============== #
        # FOR CLASS LOSS #
        # ============== #


