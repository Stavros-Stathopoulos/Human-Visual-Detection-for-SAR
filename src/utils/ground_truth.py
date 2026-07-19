import os
import cv2
import matplotlib.pyplot as plt

from src.config.config import _IR_DIR, _PREFIX_VIS, _VIS_DIR, _PREFIX_IR

def get_ground_truth(
        file: str
)-> list:
    persons = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            persons.append([float(line[0]),
                            float(line[1]),
                            float(line[2]),
                            float(line[3]),
                            float(line[4])])
    return persons

def load_image_with_boxes(
        image_file: str,
        ground_truth_file: str
):
    '''Read an image, draw its ground-truth boxes, return it as RGB (or None).'''
    points = get_ground_truth(ground_truth_file)
    image = cv2.imread(image_file)
    if image is None:
        print(f'Error: Could not read image file {image_file}')
        return None

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape
    for point in points:
        _, x, y, bw, bh = point
        x1 = int((x - bw / 2) * img_w)
        y1 = int((y - bh / 2) * img_h)
        x2 = int((x + bw / 2) * img_w)
        y2 = int((y + bh / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def plot_ground_truth(
        image_file: str,
        ground_truth_file: str
)-> None:

    img = load_image_with_boxes(image_file, ground_truth_file)
    if img is None:
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Ground Truth')
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    #get_ground_truth(os.path.join(_IR_DIR, '__ir__00000264.txt'))
    plot_ground_truth(
        os.path.join(_VIS_DIR, f'{_PREFIX_VIS}00000263.jpeg'),
        os.path.join(_VIS_DIR, f'{_PREFIX_VIS}00000263.txt')
    )
    plot_ground_truth(
        os.path.join(_IR_DIR, f'{_PREFIX_IR}00000264.jpeg'),
        os.path.join(_IR_DIR, f'{_PREFIX_IR}00000264.txt')
    )