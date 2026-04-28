from ultralytics.models.mamba_yolo_rtdetr import MambaYOLORTDETR
import argparse
import os
import torch

ROOT = os.path.abspath('.') + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml')
    parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-rtdetr.yaml')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco_rtdetr')
    parser.add_argument('--name', default='mambayolo_rtdetr_v2')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    
    # ====== 阶段1：只训练Decoder头（冻结backbone）====== 
    print("=" * 60)
    print("阶段1：冻结backbone，只训练RTDETRDecoder头")
    print("=" * 60)
    
    model = MambaYOLORTDETR(ROOT + opt.config)
    
    # 冻结所有backbone参数（模型中除最后一个模块外的所有层）
    for name, param in model.model.named_parameters():
        if 'model.-1' not in name:  # -1 是最后一层（RTDETRDecoder）
            param.requires_grad = False
    
    # 冻结训练20个epochs，使用较高学习率
    model.train(
        data=ROOT + opt.data,
        epochs=20,
        workers=opt.workers,
        batch=opt.batch_size,
        optimizer="AdamW",
        lr0=0.001,                 # 训练头用较高学习率
        lrf=0.1,
        weight_decay=0.0001,
        warmup_epochs=1.0,
        warmup_bias_lr=0.01,
        close_mosaic=0,
        device=opt.device,
        amp=True,
        project=ROOT + opt.project,
        name=opt.name + "_stage1",
        exist_ok=True,
    )
    
    # 保存阶段1的权重
    stage1_weights = ROOT + opt.project + f"/{opt.name}_stage1/weights/best.pt"
    
    # ====== 阶段2：联合微调 ======
    print("=" * 60)
    print("阶段2：加载阶段1权重，联合微调整个模型")
    print("=" * 60)
    
    # 重新加载模型（恢复所有参数为可训练）
    model = MambaYOLORTDETR(ROOT + opt.config)
    
    # 加载阶段1的最佳权重（decoder头已经初始化好）
    if os.path.exists(stage1_weights):
        model.load(stage1_weights)
    
    # 联合微调
    model.train(
        data=ROOT + opt.data,
        epochs=opt.epochs,
        workers=opt.workers,
        batch=opt.batch_size,
        optimizer="AdamW",
        lr0=0.0001,                # 联合微调用较低学习率
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_bias_lr=0.001,
        close_mosaic=10,
        device=opt.device,
        amp=True,
        project=ROOT + opt.project,
        name=opt.name + "_stage2",
        exist_ok=True,
    )