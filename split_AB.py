import os
from PIL import Image

def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths

def align_images(file_paths, a_path, b_path, data_type='train'):
    if not os.path.exists(os.path.join(a_path, data_type)):
        os.makedirs(os.path.join(a_path, data_type))
    for i in range(len(file_paths)):
        img = Image.open(file_paths[i])
        a = img.crop(box=(0,0,img.size[0]/2, img.size[1]))
        b = img.crop(box=(img.size[0]/2, 0, img.size[0], img.size[1]))
        assert(a.size == b.size)
        a.save(os.path.join(a_path, data_type, '{:04d}.jpg'.format(i)))
        b.save(os.path.join(b_path, data_type, '{:04d}.jpg'.format(i)))
        
for data_type in ['train', 'val', 'test']:
    file_paths = get_file_paths('/data/jehyuk/imgdata/datasets/facades/{}'.format(data_type))
    a_path = '/data/jehyuk/imgdata/datasets/facades/a'
    b_path = '/data/jehyuk/imgdata/datasets/facades/b'
    align_images(file_paths, a_path, b_path, data_type)