import random
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.transform_functions import GroupToTensor, GroupResize, GroupImgRandomRotation, GroupImgRandomHorizontalFlip, GroupImgNormalize, GroupImgRandomCrop, GroupImgGrayscale, Stack

class SSSD2019Dataset(Dataset):
    def __init__(self, path, mode, val_speakers=None, in_channels=1, max_timesteps=100, augmentations=False):
        assert mode in ['train', 'val', 'test']
        self.max_timesteps = max_timesteps
        self.in_channels = in_channels
        self.augmentations = augmentations if mode in ['train', 'pretrain'] else False
        self.folder_list, self.folder_paths, self.labels = self.build_folder_list(path, mode, val_speakers)

        self.transform = self.build_transform(augmentations)

        self.label_list = ('zero','one','two','three','four','five','six','seven',
                           'eight','nine','thank','no','morning','cong','sleep','sorry',
                           'hello','night','seeyou','excuse','welcome','yes','hazime','matane','moshimoshi')
        self.int2label = dict(enumerate(self.label_list))
        self.label2int = {char: index for index, char in self.int2label.items()}

    def build_folder_list(self, path, mode, val_speakers=None):
        folder_list, paths, labels = [], [], []

        if mode in ['train','val']:
            assert not val_speakers is None
            folders = sorted(listdir(join(path, 'LF-ROI')))
            if mode == 'train':
                folder_list = [folder for folder in folders if folder.split('_')[0] not in val_speakers]
            elif mode == 'val':
                folder_list = [folder for folder in folders if folder.split('_')[0] in val_speakers]
            paths = [join(*[path, 'LF-ROI', folder])for folder in folder_list]
            labels = [int(folder.split("_")[1]) -1 for folder in folder_list]
        else:
            folder_list = sorted(listdir(join(*[path, 'challenge2019', 'LF-ROI'])))
            paths = [join(*[path, 'challenge2019', 'LF-ROI', folder]) for folder in folder_list]
            with open(join(*[path, 'challenge2019', 'test.txt'])) as f:
                labels = f.readlines()
            labels = [int(label.rstrip()) for label in labels]
        
        return folder_list, paths, labels

    def __len__(self):
        return len(self.folder_paths)

    def build_transform(self, augmentations):
        if augmentations:
            augmentations_compose = transforms.Compose([
                GroupImgRandomHorizontalFlip(threshold=0.5),
                GroupImgRandomCrop(scale=(0.5,1.0), fl_threshold=0.5, fluctuation=5),
                GroupImgRandomRotation(degrees=30, fluctuation=5),
            ])
        else:
            augmentations_compose = transforms.Compose([])

        if self.in_channels == 1:
            transform = transforms.Compose([
                GroupResize(64),
                augmentations_compose,
                GroupImgGrayscale(),
                GroupToTensor(),
                # GroupImgNormalize(mean=[0.4161, ], std=[0.1688, ]),
                Stack(),
            ])
        elif self.in_channels == 3:
            transform = transforms.Compose([
                GroupResize(64),
                augmentations_compose,
                GroupToTensor(),
                # GroupImgNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Stack()
            ])     
        return transform

    def build_tensor(self, frames):
        speech_length = len(frames)
        if speech_length >= self.max_timesteps:
            idxs = [round(t*speech_length/self.max_timesteps) for t in range(self.max_timesteps)]
            frames = [frames[idx] for idx in idxs]
        else:
            frames.extend([frames[-1]]*(self.max_timesteps-speech_length))
        frames = self.transform(frames)
        return frames

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        folder_path = self.folder_paths[idx]
        label = self.labels[idx]

        # print([join(folder_path, i)for i in sorted(listdir(folder_path))])
        frames = list(map(Image.open, [join(folder_path, i)for i in sorted(listdir(folder_path))]))
        frames = self.build_tensor(frames)

        return folder_name, frames, label


