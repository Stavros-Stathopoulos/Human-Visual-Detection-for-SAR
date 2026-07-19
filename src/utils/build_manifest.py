import os
import csv

from src.typedefs.frame import Frame
from src.config.config import _IR_DIR, _VIS_DIR, _PROJECT_ROOT, _DERIVED_DIR

def get_manifest(
        folder_vis: str,
        folder_ir: str
):
    frames = []
    for vis_file in os.listdir(folder_vis):

        if vis_file.startswith('count'):
            continue
        elif not vis_file.endswith('.jpeg'):
            continue

        vis_file_name = vis_file.split('_')
        print(f'VIS file name: {vis_file}')
        frame_id = int(vis_file_name[5].split('.')[0])
        ir_file = '_'.join(
            [str(vis_file_name[0]),
            vis_file_name[1],
            vis_file_name[2],
            'IR',
            str(int(vis_file_name[4]) + 1).zfill(4),
            str(frame_id + 1).zfill(8) + '.jpeg'])
        if not os.path.exists(os.path.join(folder_ir, ir_file)):
            raise FileNotFoundError(f'IR file not found: {ir_file}')

        frame = Frame(
            sequence_id=str(vis_file_name[0] + '_' + vis_file_name[1] + '_' + vis_file_name[2]),
            frame_id=frame_id,
            rgb_image=os.path.relpath(os.path.join(folder_vis, vis_file), _PROJECT_ROOT),
            rgb_label=os.path.relpath(os.path.join(folder_vis, vis_file.split('.')[0] + '.txt'), _PROJECT_ROOT),
            ir_image = os.path.relpath(os.path.join(folder_ir, ir_file), _PROJECT_ROOT),
            ir_label=os.path.relpath(os.path.join(folder_ir, ir_file.split('.')[0] + '.txt'), _PROJECT_ROOT)
        )

        print(f'Frame: {frame} \n')
        frames.append(frame)
    print(f'Total frames processed: {len(frames)}')
    return frames

def build_manifest(
        folder_vis: str,
        folder_ir: str,
        output_file: str
):
    
    headers = list(Frame.model_fields.keys())
    frames = get_manifest(folder_vis, folder_ir)
    frames = sorted(frames, key=lambda f: (f.sequence_id, f.frame_id))
    with open(output_file, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

        for frame in frames:
            writer.writerow(frame.model_dump())

if __name__ == '__main__':
    output_file = os.path.join(_DERIVED_DIR, 'manifest.csv')
    build_manifest(_VIS_DIR, _IR_DIR, output_file)