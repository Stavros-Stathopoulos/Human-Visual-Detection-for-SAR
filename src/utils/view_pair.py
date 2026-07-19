
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from src.config.config import _PROJECT_ROOT, _DERIVED_DIR
from src.utils.ground_truth import load_image_with_boxes


def view_pair(row: pd.Series, save_dir: str | None = None) -> None:
    '''Show (or save) one manifest row's RGB and IR images side by side with boxes.'''
    rgb = load_image_with_boxes(
        os.path.join(_PROJECT_ROOT, row['rgb_image']),
        os.path.join(_PROJECT_ROOT, row['rgb_label']))
    ir = load_image_with_boxes(
        os.path.join(_PROJECT_ROOT, row['ir_image']),
        os.path.join(_PROJECT_ROOT, row['ir_label']))
    if rgb is None or ir is None:
        return

    fig, (ax_rgb, ax_ir) = plt.subplots(1, 2, figsize=(18, 7))
    ax_rgb.imshow(rgb)
    ax_rgb.set_title(f'RGB {rgb.shape[1]}x{rgb.shape[0]} — {os.path.basename(row["rgb_image"])}')
    ax_rgb.axis('off')
    ax_ir.imshow(ir)
    ax_ir.set_title(f'IR {ir.shape[1]}x{ir.shape[0]} — {os.path.basename(row["ir_image"])}')
    ax_ir.axis('off')
    fig.suptitle(f'{row["sequence_id"]} frame {row["frame_id"]}')
    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, f'{row["sequence_id"]}_{row["frame_id"]:08}.png')
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f'Saved {out_path}')
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='View RGB/IR pairs from the manifest side by side.')
    parser.add_argument('--manifest', default=os.path.join(_DERIVED_DIR, 'manifest.csv'))
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='frame_id values to show; default is 6 frames spread over the sequence')
    parser.add_argument('--save-dir', default=None,
                        help='save PNGs here instead of opening windows')
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.frames is not None:
        rows = df[df['frame_id'].isin(args.frames)]
        missing = set(args.frames) - set(rows['frame_id'])
        if missing:
            raise ValueError(f'frame_id(s) not in manifest: {sorted(missing)}')
    else:
        step = max(len(df) // 6, 1)
        rows = df.sort_values(['sequence_id', 'frame_id']).iloc[::step]

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for _, row in rows.iterrows():
        view_pair(row, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
