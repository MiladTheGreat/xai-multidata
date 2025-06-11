import os
from collections import Counter
from PIL import Image

def check_image(data_root):
    class_count = Counter()
    corrupted_files = []
    for class_name in os.listdir(data_root):
        class_path = os.path.join(data_root,class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        class_count[class_name] = len(images)

        for img_name in images:
            img_path = os.path.join(class_path,img_name)
            
            try:
                img = Image.open(img_path)
                img.verify()
            except:
                corrupted_files.append(img_path)

    print('Images For Each Class:')
    for class_name,count in class_count.items():
        print(f'    {class_name} : {count}')

    print(f'Courrupted files found: {len(corrupted_files)}')
    print('Sanity check is done!')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='sanity check for dataset.')
    parser.add_argument("--data_root",type=str,required=True)
    args = parser.parse_args()
    check_image(args.data_root)