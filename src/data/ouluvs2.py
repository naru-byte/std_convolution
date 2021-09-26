from os import listdir
from os.path import join

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.transform_functions import GroupToPIL, GroupToTensor, GroupResize, GroupImgRandomRotation, GroupImgRandomHorizontalFlip, GroupImgNormalize, GroupImgRandomCrop, GroupImgGrayscale, Stack

train_speaker_ids = [1,2,3,10,11,12,13,18,19,20,21,22,23,24,25,27,33,35,36,37,38,39,45,46,47,48,50,53]
val_speaker_ids = [4,5,7,14,16,17,28,31,32,40,41,42]
test_speaker_ids = [6,8,9,15,26,30,34,43,44,49,51,52]

class OuluVS2Dataset(Dataset):
    def __init__(self, path, mode='train', data_mode='p_d', in_channels=1, max_timesteps=32, augmentations=False):
        self.path = path
        self.in_channels = in_channels
        self.max_timesteps = max_timesteps
        # self.speakers = {
        #     'train': (0, 40),
        #     'val': (43, 48),
        #     'test': (48, 53),
        # }[mode]
        self.speakers = {
            'train': train_speaker_ids,
            'val': val_speaker_ids,
            'test': test_speaker_ids,
        }[mode]
        self.data_types = {
            'p_d': ('cropped_mouth_mp4_digit', 'cropped_mouth_mp4_phrase'),
            'p'  : ['cropped_mouth_mp4_phrase'],
            'd'  : ['cropped_mouth_mp4_digit'],
        }[data_mode]        
        self.videos, self.video_paths, self.labels = self.build_file_list(path)

        self.transform = self.build_transform(augmentations)

        digit_label_list = [
            '1 7 3 5 1 6 2 6 6 7',
            '4 0 2 9 1 8 5 9 0 4',
            '1 9 0 7 8 8 0 3 2 8',
            '4 9 1 2 1 1 8 5 5 1',
            '8 6 3 5 4 0 2 1 1 2',
            '2 3 9 0 0 1 6 7 6 4',
            '5 2 7 1 6 1 3 6 7 0',
            '9 7 4 4 4 3 5 5 8 7',
            '6 3 8 5 3 9 8 5 6 5',
            '7 3 2 4 0 1 9 9 5 0',
        ]
        phrase_label_list = [
            "Excuse me",
            "Goodbye",
            "Hello",
            "How are you",
            "Nice to meet you",
            "See you",
            "I am sorry",
            "Thank you",
            "Have a good time",
            "You are welcome",
        ]
        self.label_list = {
            'p_d': digit_label_list + phrase_label_list,
            'p'  : phrase_label_list,
            'd'  : digit_label_list,
        }[data_mode]
        self.int2label = dict(enumerate(self.label_list))
        self.label2int = {char: index for index, char in self.int2label.items()}

    def build_file_list(self, path):
        videos, paths, labels = [], [], []

        for cnt, data_type in enumerate(self.data_types):
            data_path = join(self.path, data_type)
            speakers_num = sorted(listdir(data_path))
            for speaker in speakers_num:
                if int(speaker) in self.speakers:
                # if int(speaker) >= self.speakers[0] and int(speaker) < self.speakers[1]:
                    for degree in sorted(listdir(join(data_path,speaker))):
                        for video_file in sorted(listdir(join(*[data_path,speaker,degree]))):
                            videos.append(video_file)
                            paths.append(join(*[data_path,speaker,degree,video_file]))
                            label = video_file[:-4].split("_")[-1]
                            label = ((int(label[1:]) -1) %30) // 3 + 10*cnt
                            labels.append(label)
        return videos, paths, labels

    def __len__(self):
        return len(self.videos)

    def build_transform(self, augmentations):
        if augmentations:
            augmentations_compose = transforms.Compose([
                GroupImgRandomHorizontalFlip(threshold=0.5),
                GroupImgRandomRotation(degrees=30, fluctuation=5),
            ])
        else:
            augmentations_compose = transforms.Compose([])

        if self.in_channels == 1:
            transform = transforms.Compose([
                GroupToPIL(),
                GroupResize((100,120)),
                augmentations_compose,
                GroupImgGrayscale(),
                GroupToTensor(),
                # GroupImgNormalize(mean=[0.4161, ], std=[0.1688, ]),
                Stack(),
            ])
        elif self.in_channels == 3:
            transform = transforms.Compose([
                GroupToPIL(),
                GroupResize((100,120)),
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
            frames = torch.cat([frames, frames[-1].repeat(self.max_timesteps - speech_length,1,1,1)])
        frames = self.transform(frames)
        return frames

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames, _, _ = torchvision.io.read_video(video_path)
        frames = frames.permute(0, 3, 1, 2) 
        frames = self.build_tensor(frames)

        return video_name, frames, label