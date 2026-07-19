
import os
import sys
import cv2
import pandas as pd

from src.config.config import _PROJECT_ROOT


def parse_frame_number(path: str) -> int:
    '''Extract the trailing frame number from a WiSARD file path.'''
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.split('_')[-1])


def validate_labels(label_path: str) -> tuple[list[str], list[str], int]:
    '''
    Validate a YOLO label file.

    Returns:
        (errors, warnings, n_boxes) for this file. An empty file is valid
        (frame with no person) and returns n_boxes == 0.
    '''
    errors = []
    warnings = []
    n_boxes = 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 5:
            errors.append(f'{label_path}:{line_no}: expected 5 tokens, got {len(tokens)}')
            continue
        try:
            cls = int(tokens[0])
            x, y, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            errors.append(f'{label_path}:{line_no}: non-numeric token in "{line}"')
            continue
        if cls != 0:
            errors.append(f'{label_path}:{line_no}: class must be 0, got {cls}')
        for name, value in (('x', x), ('y', y), ('w', w), ('h', h)):
            if not 0.0 <= value <= 1.0:
                errors.append(f'{label_path}:{line_no}: {name}={value} outside [0, 1]')
        if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
            warnings.append(f'{label_path}:{line_no}: box extends past image edge')
        n_boxes += 1

    return errors, warnings, n_boxes


def validate_manifest(manifest_file: str) -> bool:
    print('_' * 45)
    print(f"Validating manifest file: {manifest_file}")
    print('_' * 45)

    errors = []
    warnings = []

    if not os.path.exists(manifest_file):
        print('Test 1: Manifest file does not exists - FAILED')
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
    print('Test 1: Manifest file exists - PASSED')

    df = pd.read_csv(manifest_file)
    required_columns = ['sequence_id', 'frame_id', 'rgb_image', 'rgb_label', 'ir_image', 'ir_label']

    for col in required_columns:
        if col not in df.columns:
            print(f'Test 2: Missing required column: {col} - FAILED')
            raise ValueError(f"Missing required column: {col}")
    print('Test 2: All required columns are present - PASSED')

    test_errors = []
    for index, row in df.iterrows():
        for col in required_columns:
            if pd.isnull(row[col]) or row[col] == '':
                test_errors.append(f'row {index}: missing value in column {col}')
    errors += test_errors
    print(f'Test 3: No missing values in required columns - {"PASSED" if not test_errors else f"FAILED ({len(test_errors)} errors)"}')

    test_errors = []
    for index, row in df.iterrows():
        for col in ['rgb_image', 'rgb_label', 'ir_image', 'ir_label']:
            file_path = os.path.join(_PROJECT_ROOT, row[col])
            if not os.path.exists(file_path):
                test_errors.append(f'row {index}: file does not exist: {row[col]}')
    errors += test_errors
    print(f'Test 4: All referenced files exist - {"PASSED" if not test_errors else f"FAILED ({len(test_errors)} errors)"}')

    test_errors = []
    rgb_resolutions = set()
    ir_resolutions = set()
    for index, row in df.iterrows():
        rgb_image = cv2.imread(os.path.join(_PROJECT_ROOT, row['rgb_image']))
        if rgb_image is None:
            test_errors.append(f'row {index}: unable to read RGB image: {row["rgb_image"]}')
        else:
            rgb_resolutions.add((rgb_image.shape[1], rgb_image.shape[0]))
        ir_image = cv2.imread(os.path.join(_PROJECT_ROOT, row['ir_image']))
        if ir_image is None:
            test_errors.append(f'row {index}: unable to read IR image: {row["ir_image"]}')
        else:
            ir_resolutions.add((ir_image.shape[1], ir_image.shape[0]))
    errors += test_errors
    print(f'Test 5: All images are readable - {"PASSED" if not test_errors else f"FAILED ({len(test_errors)} errors)"}')

    test_errors = []
    test_warnings = []
    rgb_empty = 0
    ir_empty = 0
    rgb_boxes = 0
    ir_boxes = 0
    for index, row in df.iterrows():
        label_errors, label_warnings, n_boxes = validate_labels(os.path.join(_PROJECT_ROOT, row['rgb_label']))
        test_errors += label_errors
        test_warnings += label_warnings
        rgb_boxes += n_boxes
        if n_boxes == 0:
            rgb_empty += 1

        label_errors, label_warnings, n_boxes = validate_labels(os.path.join(_PROJECT_ROOT, row['ir_label']))
        test_errors += label_errors
        test_warnings += label_warnings
        ir_boxes += n_boxes
        if n_boxes == 0:
            ir_empty += 1
    errors += test_errors
    warnings += test_warnings
    print(f'Test 6: All label files are valid YOLO - {"PASSED" if not test_errors else f"FAILED ({len(test_errors)} errors)"}')

    test_errors = []
    for index, row in df.iterrows():
        for col in ['rgb_image', 'rgb_label', 'ir_image', 'ir_label']:
            basename = os.path.basename(row[col])
            if not basename.startswith(row['sequence_id']):
                test_errors.append(f'row {index}: {col} filename does not match sequence_id {row["sequence_id"]}')
        if parse_frame_number(row['rgb_image']) != row['frame_id']:
            test_errors.append(f'row {index}: rgb_image frame number does not match frame_id {row["frame_id"]}')
        if parse_frame_number(row['rgb_label']) != row['frame_id']:
            test_errors.append(f'row {index}: rgb_label frame number does not match frame_id {row["frame_id"]}')
        if parse_frame_number(row['ir_image']) != parse_frame_number(row['ir_label']):
            test_errors.append(f'row {index}: ir_image and ir_label frame numbers disagree')
    errors += test_errors
    print(f'Test 7: Filenames consistent with sequence_id/frame_id - {"PASSED" if not test_errors else f"FAILED ({len(test_errors)} errors)"}')

    duplicates = df.duplicated(subset=['sequence_id', 'frame_id'], keep=False)
    test_errors = [f'row {index}: duplicate (sequence_id, frame_id)' for index in df[duplicates].index]
    errors += test_errors
    print(f'Test 8: No duplicate (sequence_id, frame_id) pairs - {"PASSED" if not test_errors else f"FAILED ({len(test_errors)} errors)"}')

    print('_' * 45)
    print('Summary')
    print('_' * 45)
    print(f'Rows checked          : {len(df)}')
    print(f'Errors                : {len(errors)}')
    print(f'Warnings              : {len(warnings)}')
    print(f'RGB resolutions (WxH) : {sorted(rgb_resolutions)}')
    print(f'IR resolutions (WxH)  : {sorted(ir_resolutions)}')
    print(f'RGB boxes / empty     : {rgb_boxes} boxes, {rgb_empty} empty frames')
    print(f'IR boxes / empty      : {ir_boxes} boxes, {ir_empty} empty frames')
    for sequence_id, group in df.groupby('sequence_id'):
        frame_ids = sorted(group['frame_id'])
        gaps = [f for f in range(frame_ids[0], frame_ids[-1] + 1) if f not in set(frame_ids)]
        print(f'Sequence {sequence_id}: frames {frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)} rows, {len(gaps)} gaps)')

    for message in errors:
        print(f'ERROR: {message}')
    for message in warnings:
        print(f'WARNING: {message}')

    return len(errors) == 0


if __name__ == '__main__':
    manifest_file = os.path.join(_PROJECT_ROOT, 'data', 'derived', 'manifest.csv')
    if not validate_manifest(manifest_file):
        sys.exit(1)
