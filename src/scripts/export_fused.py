import os
import cv2
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd

from src.config.config import (_DERIVED_DIR,
                               _PROJECT_ROOT,
                               IR_WIDTH as IR_W,
                               IR_HEIGHT as IR_H,
                               AFFINE_X_SCALE as AX,
                               AFFINE_X_OFFSET as BX,
                               AFFINE_Y_SCALE as AY,
                               AFFINE_Y_OFFSET as BY)

SMOKE_SPLIT_WARNING = (
    'WARNING: smoke split (last N frames of the same sequence as val). '
    'Same-sequence leakage — metrics from this dataset are NOT valid for reporting.'
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export fused 4-channel (RGB+IR) dataset')
    parser.add_argument('--manifest', default=os.path.join(_DERIVED_DIR, 'manifest.csv'))
    parser.add_argument('--output-dir', default=os.path.join(_DERIVED_DIR, 'yolo_fused'))
    parser.add_argument('--val-frames', type=int, default=50,
                        help='number of trailing frames used as the smoke val split (default: 50)')
    return parser.parse_args()

def fuse_pair(
        rgb_path: str,
        ir_path: str
) -> np.ndarray:
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None:
        raise FileNotFoundError(f'RGB image not found: {rgb_path}')
    if ir is None:
        raise FileNotFoundError(f'IR image not found: {ir_path}')

    H, W = rgb.shape[:2]
    x1, x2 = round(BX * W), round((AX + BX) * W)
    y1, y2 = round(BY * H), round((AY + BY) * H)

    crop = rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (IR_W, IR_H), interpolation=cv2.INTER_AREA)

    assert ir.shape == (IR_H, IR_W), f'IR image shape mismatch: {ir.shape} != ({IR_H}, {IR_W})'

    # channel order: B, G, R, IR
    fused = np.dstack([crop, ir])
    return fused

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df = df.sort_values(['sequence_id', 'frame_id'])

    train_rows = df[:-args.val_frames]
    val_rows = df[-args.val_frames:]

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output_dir, 'images', split))
        os.makedirs(os.path.join(args.output_dir, 'labels', split))

    for split, rows in [('train', train_rows), ('val', val_rows)]:
        for _, row in rows.iterrows():
            fused = fuse_pair(
                os.path.join(_PROJECT_ROOT, row['rgb_image']),
                os.path.join(_PROJECT_ROOT, row['ir_image'])
            )
            # fused frame has IR geometry -> IR basename, PNG for lossless 4-channel
            name = os.path.splitext(os.path.basename(row['ir_image']))[0] + '.png'
            cv2.imwrite(os.path.join(args.output_dir, 'images', split, name), fused)

            # IR labels are already in fused-frame coordinates
            os.symlink(
                os.path.join(_PROJECT_ROOT, row['ir_label']),
                os.path.join(args.output_dir, 'labels', split,
                             os.path.splitext(name)[0] + '.txt')
            )

    dataset = {
        'path': os.path.abspath(args.output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person'},
        'channels': 4,
    }
    with open(os.path.join(args.output_dir, 'dataset.yaml'), 'w') as f:
        yaml.safe_dump(dataset, f, sort_keys=False)

    print(f'Exported fused: {len(train_rows)} train / {len(val_rows)} val -> {args.output_dir}')
    print(SMOKE_SPLIT_WARNING)

if __name__ == '__main__':
    main()
