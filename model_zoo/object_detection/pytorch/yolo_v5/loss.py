import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_iou(box1, box2):
    box1_x1 = box1[...,0] - box1[...,2]/2
    box1_y1 = box1[...,1] - box1[...,3]/2
    box1_x2 = box1[...,0] + box1[...,2]/2
    box1_y2 = box1[...,1] + box1[...,3]/2

    box2_x1 = box2[...,0] - box2[...,2]/2
    box2_y1 = box2[...,1] - box2[...,3]/2
    box2_x2 = box2[...,0] + box2[...,2]/2
    box2_y2 = box2[...,1] + box2[...,3]/2

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - inter_area + 1e-6
    return inter_area / union


class Custom_loss(nn.Module):
    def __init__(self, num_classes=10, num_boxes=2):
        super().__init__()
        self.C = num_classes
        self.B = num_boxes
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.2
        self.lambda_class = 0.5

    def forward(self, preds, targets):
        BATCH, S1, S2, total_channels = preds.shape

        expected_channels = self.C + 5*self.B
        if total_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels but got {total_channels}."
            )

        # Split predictions and targets
        pred_cls = preds[..., :self.C]            # (B,7,7,12)
        pred_boxes = preds[..., self.C:].view(BATCH, S1, S2, self.B,5)

        target_cls = targets[..., :self.C]        # (B,7,7,12)
        target_boxes = targets[..., self.C:].view(BATCH, S1, S2, self.B,5)

        pred_xywh = pred_boxes[...,:4]
        pred_obj = pred_boxes[...,4]

        target_xywh = target_boxes[...,:4]
        target_obj = target_boxes[...,4]

        # IoU loss
        iou = bbox_iou(pred_xywh, target_xywh).detach()
        box_loss = self.lambda_box * torch.sum(target_obj * (1.0 - iou))

        # Objectness loss
        obj_loss = self.lambda_obj * F.binary_cross_entropy_with_logits(
            pred_obj, target_obj.float(), reduction="sum"
        )

        # No-objectness loss
        noobj_loss = self.lambda_noobj * F.binary_cross_entropy_with_logits(
            pred_obj, target_obj.float(), reduction="sum"
        )

        # Class loss (per-cell, not per-box)
        class_loss = self.lambda_class * F.binary_cross_entropy_with_logits(
            pred_cls, target_cls.float(), reduction="sum"
        )

        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        return total_loss
