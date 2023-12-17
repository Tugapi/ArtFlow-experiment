import os
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

import random
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def transform(img_size):
    transform_list = [
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class UnAlignedDataSet(data.Dataset):
    """
    This dataset class can load unaligned datasets.

    It requires two directories to host training images from content domain and from style domain respectively.
    """
    def __init__(self, args):
        super(UnAlignedDataSet, self).__init__()
        self.content_dir = args.content_dir
        self.style_dir = args.style_dir
        self.transform = transform(args.img_size)
        self.content_paths = make_dataset(self.content_dir)
        self.content_num = len(self.content_paths)
        if self.content_num == 0:
            raise(RuntimeError("Found 0 images in: " + self.content_dir + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.style_paths = make_dataset(self.style_dir)
        self.style_num = len(self.style_paths)
        if self.style_num == 0:
            raise(RuntimeError("Found 0 images in: " + self.style_dir + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        content_path = self.content_paths[index % self.content_num]  # make sure index is within then range
        index_style = random.randint(0, self.style_num - 1)
        style_path = self.style_paths[index_style]
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')
        content_input = self.transform(content_img)
        style_input = self.transform(style_img)

        return {"content": content_input, "style": style_input}

    def __len__(self):
        return self.content_num



