import argparse
from pathlib import Path
from ultralytics import DSVP
from ultralytics.models.dsvp.detect import SceneDetectionTrainer, StreamDetectionTrainer

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSVP with causal multi-scale selective memory.")

    parser.add_argument("--data", default=f"dataset yaml path")
    parser.add_argument("--spatial-model", default=f"spatial model yaml path")
    parser.add_argument("--stream-model", default=f"stream model yaml path")
    parser.add_argument("--stage1-epochs", type=int, default=60)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--stage3-epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--prefix", default="EA_UAV_DSVP")
    parser.add_argument("--stage3-lr", type=float, default=2e-6)
    parser.add_argument("--start-stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--weights", default=r"")
    parser.add_argument("--stream-lr-scale", type=float, default=50.0)
    parser.add_argument("--backbone-stream-lr-scale", type=float, default=500.0)
    parser.add_argument("--stage3-scope", choices=("backbone", "temporal", "head", "all"), default="all")
    parser.add_argument("--backbone-temporal-layers", default="")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--stage3-amp", action="store_true")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def best_weight(model):
    trainer = model.trainer
    return trainer.best if trainer.best.exists() else trainer.last


def spatial_train(weights, args, stage, mosaic):
    model = DSVP(str(weights))
    model.train(
        trainer=SceneDetectionTrainer,
        data=args.data,
        epochs=args.stage1_epochs if stage == 1 else args.stage2_epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        fraction=args.fraction,
        project=args.project,
        name=f"{args.prefix}_stage{stage}_{'mosaic' if mosaic else 'local'}",
        mosaic=1.0 if mosaic else 0.0,
        mixup=0.0,
        copy_paste=0.1 if mosaic else 0.0,
        close_mosaic=0,
        optimizer="auto" if mosaic else "AdamW",
        lr0=0.01 if mosaic else 5e-4,
        warmup_epochs=3.0 if mosaic else 1.0,
    )
    return best_weight(model)


def stream_train(weights, args):
    model = DSVP(args.stream_model).load(weights)
    model.train(
        trainer=StreamDetectionTrainer,
        data=args.data,
        epochs=args.stage3_epochs,
        batch=1,
        imgsz=args.imgsz,
        workers=0,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        amp=args.stage3_amp,
        fraction=args.fraction,
        project=args.project,
        name=f"{args.prefix}_stage3_stream",
        optimizer="AdamW",
        lr0=args.stage3_lr,
        lrf=0.1,
        warmup_epochs=0.5,
        nbs=64,
        weight_decay=1e-4,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        stream=True,
        temporal_only=args.stage3_scope == "temporal",
        stream_head_only=args.stage3_scope == "head",
        backbone_temporal_only=args.stage3_scope == "backbone",
        backbone_temporal_layers=args.backbone_temporal_layers,
        stream_lr_scale=args.stream_lr_scale,
        backbone_stream_lr_scale=args.backbone_stream_lr_scale or None,
    )
    return best_weight(model)


def main():
    args = parse_args()
    weights = Path(args.weights)
    if args.start_stage == 1:
        weights = spatial_train(args.spatial_model, args, stage=1, mosaic=True)
    if args.start_stage <= 2:
        weights = spatial_train(weights, args, stage=2, mosaic=False)
    weights = stream_train(weights, args)
    print(f"Best DSVP streaming checkpoint: {weights}")


if __name__ == "__main__":
    main()
