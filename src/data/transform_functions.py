import math
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

#Pytorchによる発展ディープラーニング9章参考
class GroupToTensor():
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    def __call__(self, frames):
        return [self.to_tensor(frame) for frame in frames]

class GroupToPIL():
    def __init__(self):
        self.to_pil = transforms.ToPILImage()
    def __call__(self, frames):
        return [self.to_pil(frame) for frame in frames]  

class GroupImgNormalize():
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)
    def __call__(self, frames):
        return [self.normalize(frame) for frame in frames]

class GroupImgRandomHorizontalFlip():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    def __call__(self, frames):
        if random.random() > self.threshold:
            return [TF.hflip(frame)for frame in frames]
        else:
            return frames

class GroupImgRandomRotation():
    def __init__(self, degrees, fluctuation=0, fl_threshold=1.0):
        self.degrees = degrees
        self.fluctuation, self.fl_threshold = fluctuation, fl_threshold
    def __call__(self, frames):
        degree = random.randint(-self.degrees, self.degrees)
        if random.random() > self.fl_threshold:
            retrun_list = []
            for frame in frames:
                retrun_list.append(TF.rotate(frame, degree + random.randint(-self.fluctuation, self.fluctuation)))
            return retrun_list
        else:
            return [TF.rotate(frame, degree) for frame in frames]

class GroupResize():
    def __init__(self, resize, interpolation=Image.BILINEAR):
        self.rescaler = transforms.Resize(resize, interpolation)
    def __call__(self, frames):
        return [self.rescaler(frame) for frame in frames]

class GroupImgRandomCrop():
    def __init__(self, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=Image.BILINEAR, fluctuation=0, fl_threshold=1.0):
        self.scale = scale
        self.ratio = ratio  
        self.interpolation = interpolation
        self.fluctuation, self.fl_threshold = fluctuation, fl_threshold

    def get_params(self, img, scale, ratio):
        width, height = img.size
        area = height * width
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w, height, width

        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, frames):
        i, j, h, w, height, width = self.get_params(img=frames[0], scale=self.scale, ratio=self.ratio)
        if random.random() > self.fl_threshold:
            retrun_list = []
            for frame in frames:
                f_i, f_j = random.randint(-self.fluctuation, self.fluctuation), random.randint(-self.fluctuation, self.fluctuation)
                # f_h, f_w = random.randint(-self.fluctuation, self.fluctuation), random.randint(-self.fluctuation, self.fluctuation)

                t_i, t_j = i - f_i if i - f_i > 0 else 0, j - f_j if j - f_j > 0 else 0
                # t_h, t_w = h + f_h if h + f_h < height else height - 1, w + f_w if w + f_w < width else width - 1

                retrun_list.append(TF.resized_crop(frame, t_i, t_j, h, w, size=(height,width) , interpolation=self.interpolation))
            return retrun_list
        else:
            return [TF.resized_crop(frame, i, j, h, w, size=(height,width) , interpolation=self.interpolation) for frame in frames]

# class GroupImgRandomCrop():
#     def __init__(self, resize, interpolation=Image.BILINEAR):
#         self.resize = resize
#         self.interpolation = interpolation
#     def __call__(self, frames):
#         return [TF.resized_crop(frame, size=self.resize, interpolation=self.interpolation) for frame in frames]

class GroupImgGrayscale():
    def __init__(self):
        self.color2gray = transforms.Grayscale(num_output_channels=1)
    def __call__(self, frames):
        return [self.color2gray(frame) for frame in frames]

class Stack():
    def __call__(self, frames):
        return torch.stack(frames, axis=1)

# class MyRotationTransform:
#     """Rotate by one of the given angles."""

#     def __init__(self, angles, fluctuation=0):
#         self.angles = angles
#         self.fluctuation = fluctuation

#     def __call__(self, x):
#         angle = self.angles + random.randint(-self.fluctuation, self.fluctuation)
#         return TF.rotate(x, angle)