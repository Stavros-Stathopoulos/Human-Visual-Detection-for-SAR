import os
import argparse

import cv2

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from src.config.config import _DERIVED_DIR


class FusedDataset(YOLODataset):
    '''YOLODataset that keeps the 4th channel of RGBA PNGs.

    ultralytics 8.4.102 reads with IMREAD_COLOR for any channels != 1
    (data/base.py:116), which silently strips the IR (alpha) channel.
    IMREAD_UNCHANGED returns the image exactly as stored.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.channels == 4:
            self.cv2_flag = cv2.IMREAD_UNCHANGED


class FusedTrainer(DetectionTrainer):
    '''DetectionTrainer that builds FusedDataset and verifies 4-channel batches.'''

    _channels_verified = False

    def build_dataset(self, img_path: str, mode: str = 'train', batch: int | None = None):
        # mirrors build_yolo_dataset (data/build.py) with FusedDataset swapped in
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        rect = mode == 'val'
        return FusedDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',
            hyp=self.args,
            rect=self.args.rect or rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == 'train' else 0.5,
            prefix=colorstr(f'{mode}: '),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == 'train' else 1.0,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not FusedTrainer._channels_verified:
            n_ch = batch['img'].shape[1]
            assert n_ch == 4, (
                f'Expected 4-channel batches (B,4,H,W), got {tuple(batch["img"].shape)} — '
                f'the IR channel was dropped somewhere in the pipeline.')
            print(f'Verified: batch shape {tuple(batch["img"].shape)} — IR channel present.')
            FusedTrainer._channels_verified = True
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train YOLO on the fused 4-channel dataset')
    parser.add_argument('--data', default=os.path.join(_DERIVED_DIR, 'yolo_fused', 'dataset.yaml'))
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = FusedTrainer(overrides={
        'data': args.data,
        'model': args.model,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
    })
    trainer.train()


if __name__ == '__main__':
    main()
