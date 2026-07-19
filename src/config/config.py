import os

_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_ROOT_DIR), 'data', 'WiSARD')
_IR_DIR = os.path.join(_DATA_DIR, '_IR_')
_VIS_DIR = os.path.join(_DATA_DIR, '_VIS_')
_PREFIX_IR = '__ir__'
_PREFIX_VIS = '__vis__'