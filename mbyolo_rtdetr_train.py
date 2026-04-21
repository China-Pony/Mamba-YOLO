from ultralytics.models.mamba_yolo_rtdetr import MambaYOLORTDETR
import argparse
import os

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-rtdetr.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='AdamW', help='SGD, Adam, AdamW')
    parser.add_argument('--lr0', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate factor (lr0 * lrf)')
    parser.add_argument('--warmup_epochs', type=float, default=5.0, help='warmup epochs')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.001, help='warmup bias learning rate')
    parser.add_argument('--close_mosaic', type=int, default=0, help='disable mosaic for last N epochs (0=disabled)')
    parser.add_argument('--amp', action='store_true', help='open amp (CAUTION: may cause NaN with RT-DETR)')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco_rtdetr', help='save to project/name')
    parser.add_argument('--name', default='mambayolo_rtdetr', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    args = {
        "data": ROOT + opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "lr0": opt.lr0,
        "lrf": opt.lrf,
        "warmup_epochs": opt.warmup_epochs,
        "warmup_bias_lr": opt.warmup_bias_lr,
        "close_mosaic": opt.close_mosaic,
        "device": opt.device,
        "amp": opt.amp,
        "project": ROOT + opt.project,
        "name": opt.name,
    }
    model_conf = ROOT + opt.config
    model = MambaYOLORTDETR(model_conf)
    task_type = {
        "train": model.train(**args),
        "val": model.val(**args),
    }
    task_type.get(opt.task)
