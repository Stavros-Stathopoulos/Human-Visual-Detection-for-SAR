import os

from src.config.config import _IR_DIR, _VIS_DIR

def rename_files(
        data_dir: str, 
        prefix: str) -> None:
    '''
    Rename files in the specified directory by adding a prefix to their names.
    
    Args:
        data_dir (str): The directory containing the files to be renamed.
        prefix (str): The prefix to add to the file names.
    '''

    for file in os.listdir(data_dir):
        if file.startswith('count'):
            continue
        else:
            name = f'{prefix}{file.split("_")[-1]}'
            os.rename(os.path.join(data_dir, file), os.path.join(data_dir, name))
            print(f'Renamed {file} to {name}')

if __name__ == "__main__":
    rename_files(os.path.join(_IR_DIR), '__ir__')
    rename_files(os.path.join(_VIS_DIR), '__vis__')