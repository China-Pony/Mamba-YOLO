import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from pathlib import Path

def test_yaml_parse():
    print("=" * 60)
    print("Step 1: Testing YAML config parsing...")
    print("=" * 60)

    for variant in ['T', 'B', 'L']:
        yaml_path = f"ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-{variant}-rtdetr.yaml"
        try:
            with open(yaml_path, 'r') as f:
                d = yaml.safe_load(f)
            head_last = d["head"][-1]
            print(f"  {variant}-rtdetr: head[-1] = {head_last}")
            assert head_last[-2] == "RTDETRDecoder", f"Expected RTDETRDecoder, got {head_last[-2]}"
            assert head_last[0] == [14, 17, 20], f"Expected [14,17,20], got {head_last[0]}"
            print(f"  [OK] {variant}-rtdetr YAML valid (RTDETRDecoder with P3/P4/P5)")
        except Exception as e:
            print(f"  [FAIL] {variant}-rtdetr YAML: {e}")
    print()

def test_imports():
    print("=" * 60)
    print("Step 2: Testing module imports...")
    print("=" * 60)

    tests = [
        ("ultralytics.nn.modules.head.RTDETRDecoder", "RTDETRDecoder in head.py"),
        ("ultralytics.nn.modules.transformer.DeformableTransformerDecoder", "DeformableTransformerDecoder"),
        ("ultralytics.nn.modules.transformer.MSDeformAttn", "MSDeformAttn"),
        ("ultralytics.nn.modules.transformer.MLP", "MLP"),
    ]

    for module_path, desc in tests:
        try:
            parts = module_path.rsplit('.', 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls = getattr(mod, parts[1])
            print(f"  [OK] {desc}: {cls}")
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")
    print()

def test_file_structure():
    print("=" * 60)
    print("Step 3: Testing file structure...")
    print("=" * 60)

    expected_files = [
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-rtdetr.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-rtdetr.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B-rtdetr.yaml",
        "ultralytics/models/mamba_yolo_rtdetr/__init__.py",
        "ultralytics/models/mamba_yolo_rtdetr/model.py",
        "ultralytics/models/mamba_yolo_rtdetr/train.py",
        "ultralytics/models/mamba_yolo_rtdetr/val.py",
        "ultralytics/models/mamba_yolo_rtdetr/predict.py",
        "mbyolo_rtdetr_train.py",
    ]

    for f in expected_files:
        exists = os.path.isfile(f)
        status = "[OK]" if exists else "[FAIL]"
        print(f"  {status} {f}")
    print()

def test_tasks_py_modifications():
    print("=" * 60)
    print("Step 4: Testing tasks.py modifications...")
    print("=" * 60)

    with open("ultralytics/nn/tasks.py", 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ("class MambaYOLORTDETRModel" in content, "MambaYOLORTDETRModel class defined"),
        ('"detect", "rtdetrdecoder"' in content, "cfg2task handles RTDETRDecoder"),
        ("RTDETRDecoder" in content and "Detect, WorldDetect, RTDETRDecoder" in content,
         "guess_model_task handles RTDETRDecoder in module check"),
    ]

    for condition, desc in checks:
        status = "[OK]" if condition else "[FAIL]"
        print(f"  {status} {desc}")
    print()

def test_model_module_consistency():
    print("=" * 60)
    print("Step 5: Testing module consistency...")
    print("=" * 60)

    with open("ultralytics/models/mamba_yolo_rtdetr/__init__.py", 'r') as f:
        init_content = f.read()
    with open("ultralytics/models/mamba_yolo_rtdetr/model.py", 'r') as f:
        model_content = f.read()
    with open("ultralytics/models/mamba_yolo_rtdetr/train.py", 'r') as f:
        train_content = f.read()
    with open("ultralytics/models/mamba_yolo_rtdetr/val.py", 'r') as f:
        val_content = f.read()
    with open("ultralytics/models/mamba_yolo_rtdetr/predict.py", 'r') as f:
        predict_content = f.read()

    checks = [
        ("MambaYOLORTDETR" in init_content, "__init__.py exports MambaYOLORTDETR"),
        ("MambaYOLORTDETRModel" in model_content, "model.py uses MambaYOLORTDETRModel"),
        ("MambaYOLORTDETRTrainer" in train_content, "train.py defines MambaYOLORTDETRTrainer"),
        ("MambaYOLORTDETRValidator" in val_content, "val.py defines MambaYOLORTDETRValidator"),
        ("MambaYOLORTDETRPredictor" in predict_content, "predict.py defines MambaYOLORTDETRPredictor"),
        ("RTDETRDetectionLoss" in train_content or "MambaYOLORTDETRModel" in train_content,
         "train.py uses RTDETR-compatible model"),
        ("stretch=True" in val_content, "val.py uses stretch augmentation for RT-DETR"),
        ("scaleFill=True" in predict_content, "predict.py uses scaleFill for RT-DETR"),
        ("giou_loss" in train_content, "train.py uses RT-DETR loss names"),
        ("postprocess" in val_content and "xywh2xyxy" in val_content, "val.py has RT-DETR postprocess"),
    ]

    for condition, desc in checks:
        status = "[OK]" if condition else "[FAIL]"
        print(f"  {status} {desc}")
    print()

if __name__ == '__main__':
    test_yaml_parse()
    test_imports()
    test_file_structure()
    test_tasks_py_modifications()
    test_model_module_consistency()
    print("All static checks completed!")
