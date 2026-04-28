import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops


class MambaYOLORTDETRDataset(YOLODataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=False)
        else:
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class MambaYOLORTDETRValidator(DetectionValidator):

    def build_dataset(self, img_path, mode="val", batch=None):
        return MambaYOLORTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred
        return outputs

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)
            bbox[..., [0, 2]] *= ori_shape[1]
            bbox[..., [1, 3]] *= ori_shape[0]
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz
        return predn.float()
