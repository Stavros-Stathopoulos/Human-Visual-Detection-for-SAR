import os

_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
_DERIVED_DIR = os.path.join(os.path.dirname(_ROOT_DIR), 'data', 'derived')
_PROJECT_ROOT = os.path.dirname(_ROOT_DIR)
_DATA_DIR = os.path.join(os.path.dirname(_ROOT_DIR), 'data', 'WiSARD_Multi_Modal_Sample')
_IR_DIR = os.path.join(_DATA_DIR, '210417_MtErie_Enterprise_IR_0004')
_VIS_DIR = os.path.join(_DATA_DIR, '210417_MtErie_Enterprise_VIS_0003')
_PREFIX_IR = '210417_MtErie_Enterprise_IR_0004_'
_PREFIX_VIS = '210417_MtErie_Enterprise_VIS_0003_'