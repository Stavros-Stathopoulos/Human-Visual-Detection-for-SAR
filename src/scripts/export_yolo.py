import os
import yaml
import shutil
import argparse
import pandas as pd

from src.config.config import _DERIVED_DIR, _PROJECT_ROOT

SMOKE_SPLIT_WARNING = (
    'WARNING: smoke split (last N frames of the same sequence as val). '
    'Same-sequence leakage — metrics from this dataset are NOT valid for reporting.'
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export manifest to YOLO format.')
    parser.add_argument('--manifest',
                        default=os.path.join(_DERIVED_DIR, 'manifest.csv'))
    parser.add_argument('--modality',
                        choices=['rgb', 'ir'],
                        default='rgb',
                        help='modality to export (default: rgb)')
    parser.add_argument('--val-frames',
                        type=int,
                        default=50,
                        help='number of trailing frames used as the smoke val split (default: 50)')
    parser.add_argument('--output-dir',
                        default=None,
                        help='output directory (default: data/derived/yolo_<modality>)')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(_DERIVED_DIR, f'yolo_{args.modality}')
    return args

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df = df.sort_values(['sequence_id', 'frame_id'])

    if args.modality == 'rgb':
        img_col, lbl_col = 'rgb_image', 'rgb_label'
    else:
        img_col, lbl_col = 'ir_image', 'ir_label'

    train_rows = df[:-args.val_frames]
    val_rows = df[-args.val_frames:]

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output_dir, 'images', split))
        os.makedirs(os.path.join(args.output_dir, 'labels', split))

    for split, split_rows in [('train', train_rows), ('val', val_rows)]:
        for _, row in split_rows.iterrows():
            img_path = os.path.join(_PROJECT_ROOT, row[img_col])
            lbl_path = os.path.join(_PROJECT_ROOT, row[lbl_col])
            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                raise FileNotFoundError(
                    f'Manifest row {row["sequence_id"]}/{row["frame_id"]} points to a '
                    f'missing file — rebuild and validate the manifest first.')

            os.symlink(img_path, os.path.join(
                args.output_dir, 'images', split, os.path.basename(img_path)))
            os.symlink(lbl_path, os.path.join(
                args.output_dir, 'labels', split, os.path.basename(lbl_path)))

    dataset = {
        'path': os.path.abspath(args.output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person'},
    }
    with open(os.path.join(args.output_dir, 'dataset.yaml'), 'w') as f:
        yaml.safe_dump(dataset, f, sort_keys=False)

    print(f'Exported {args.modality}: {len(train_rows)} train / {len(val_rows)} val '
          f'-> {args.output_dir}')
    print(SMOKE_SPLIT_WARNING)

if __name__ == '__main__':
    main()
