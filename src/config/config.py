import os

import yaml

_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
_DERIVED_DIR = os.path.join(os.path.dirname(_ROOT_DIR), 'data', 'derived')
_PROJECT_ROOT = os.path.dirname(_ROOT_DIR)
_DATA_DIR = os.path.join(os.path.dirname(_ROOT_DIR), 'data', 'WiSARD_Multi_Modal_Sample')
_IR_DIR = os.path.join(_DATA_DIR, '210417_MtErie_Enterprise_IR_0004')
_VIS_DIR = os.path.join(_DATA_DIR, '210417_MtErie_Enterprise_VIS_0003')
_PREFIX_IR = '210417_MtErie_Enterprise_IR_0004_'
_PREFIX_VIS = '210417_MtErie_Enterprise_VIS_0003_'

with open(os.path.join(os.path.dirname(__file__), 'registration.yaml')) as _f:
    _REGISTRATION = yaml.safe_load(_f)

# IR -> RGB affine (normalized coords): rgb = scale * ir + offset
AFFINE_X_SCALE = _REGISTRATION['affine']['x']['scale']
AFFINE_X_OFFSET = _REGISTRATION['affine']['x']['offset']
AFFINE_Y_SCALE = _REGISTRATION['affine']['y']['scale']
AFFINE_Y_OFFSET = _REGISTRATION['affine']['y']['offset']
IR_WIDTH = _REGISTRATION['ir_size']['width']
IR_HEIGHT = _REGISTRATION['ir_size']['height']