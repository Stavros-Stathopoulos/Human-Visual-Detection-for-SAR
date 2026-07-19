
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config.config import _PROJECT_ROOT, _DERIVED_DIR
from src.utils.ground_truth import get_ground_truth

RGB_WIDTH, RGB_HEIGHT = 3840, 2160

def match_boxes(
        rgb_boxes: list[list[float]],
        ir_boxes: list[list[float]],
        threshold: float = 0.1
) -> list[tuple]:
    '''Match boxes between modalities by nearest normalized centers.

    Returns ((x_ir, y_ir), (x_rgb, y_rgb)) pairs; empty list when the frame
    is skipped (unequal box counts) or no match is within the threshold.
    '''
    if len(rgb_boxes) != len(ir_boxes):
        return []
    centers_rgb = [(x, y) for _, x, y, w, h in rgb_boxes]
    centers_ir = [(x, y) for _, x, y, w, h in ir_boxes]

    candidates = []
    for i, ir_center in enumerate(centers_ir):
        for j, rgb_center in enumerate(centers_rgb):
            distance = ((ir_center[0] - rgb_center[0]) ** 2 + (ir_center[1] - rgb_center[1]) ** 2) ** 0.5
            candidates.append((distance, i, j))

    candidates.sort(key=lambda x: x[0])

    pairs = []
    used_ir, used_rgb = set(), set()
    for distance, i, j in candidates:
        if distance > threshold:
            break
        if i not in used_ir and j not in used_rgb:
            pairs.append((centers_ir[i], centers_rgb[j]))
            used_ir.add(i)
            used_rgb.add(j)

    return pairs


def fit_axis(
        ir_values: list[float],
        rgb_values: list[float]
):
    return np.polyfit(ir_values, rgb_values, 1)

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(_DERIVED_DIR, 'manifest.csv'))
    all_pairs = []
    skipped = 0

    for _, row in df.iterrows():
        rgb_boxes = get_ground_truth(os.path.join(_PROJECT_ROOT, row['rgb_label']))
        ir_boxes = get_ground_truth(os.path.join(_PROJECT_ROOT, row['ir_label']))
        pairs = match_boxes(rgb_boxes, ir_boxes)
        if not pairs and rgb_boxes:
            skipped += 1

        for ir_c, rgb_c in pairs:
            all_pairs.append((row['frame_id'], ir_c, rgb_c))

    a, b = fit_axis([ir_c[0] for _, ir_c, _ in all_pairs], [rgb_c[0] for _, _, rgb_c in all_pairs])
    c, d = fit_axis([ir_c[1] for _, ir_c, _ in all_pairs], [rgb_c[1] for _, _, rgb_c in all_pairs])

    errors = []
    frame_errors = {}
    for frame_id, ir_c, rgb_c in all_pairs:
        err_x = (a * ir_c[0] + b - rgb_c[0]) * RGB_WIDTH
        err_y = (c * ir_c[1] + d - rgb_c[1]) * RGB_HEIGHT
        err = np.hypot(err_x, err_y)
        errors.append(err)
        frame_errors.setdefault(frame_id, []).append(err)

    print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
    print(f"IR spans RGB x in [{b:.3f}, {a + b:.3f}], y in [{d:.3f}, {c + d:.3f}]")
    print(f"Total pairs: {len(all_pairs)}, skipped frames: {skipped}")
    print(f"Error (RGB px): mean={np.mean(errors):.1f}, median={np.median(errors):.1f}, p95={np.percentile(errors, 95):.1f}")

    frame_ids = sorted(frame_errors)
    mean_errs = [np.mean(frame_errors[f]) for f in frame_ids]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter([ir_c[0] for _, ir_c, _ in all_pairs], [rgb_c[0] for _, _, rgb_c in all_pairs], alpha=0.5)
    plt.plot([0, 1], [b, a + b], color='red', label=f'y={a:.4f}x + {b:.4f}')
    plt.xlabel('IR center x')
    plt.ylabel('RGB center x')
    plt.title('IR vs RGB Center X')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter([ir_c[1] for _, ir_c, _ in all_pairs], [rgb_c[1] for _, _, rgb_c in all_pairs], alpha=0.5)
    plt.plot([0, 1], [d, c + d], color='red', label=f'y={c:.4f}x + {d:.4f}')
    plt.xlabel('IR center y')
    plt.ylabel('RGB center y')
    plt.title('IR vs RGB Center Y')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(frame_ids, mean_errs)
    plt.xlabel('frame_id')
    plt.ylabel('mean error (RGB px)')
    plt.title('Registration error over sequence')

    plt.tight_layout()
    output_plot = os.path.join(_DERIVED_DIR, 'registration_fit.png')
    plt.savefig(output_plot, dpi=100)
    print(f'Saved plot: {output_plot}')
    plt.show()
