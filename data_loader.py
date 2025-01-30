
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os
from torchvision import transforms
from torchvision import transforms as video_transforms

class FetchData():
    def __init__(self, dataset, video_dir, task='train', segment_duration=1, segment_interval=0.5, target_size=(224, 224)):
        # Load dataset CSV file
        self.df = pd.read_csv(f'Datasets\\{dataset}_{task}.csv')
        self.df = self.df.reset_index()
        self.video_dir = video_dir
        self.task = task
        self.segment_duration = segment_duration
        self.segment_interval = segment_interval
        self.target_size = target_size
        self.frames_list = []
        self.labels = []
        self.length_videos = len(self.df)
        print("Number of Vids: ", self.length_videos)
        self.prepare_data()

    def __len__(self):
        return len(self.frames_list)
    
    def __getitem__(self, id):
        if torch.is_tensor(id):
            id = id.tolist()
        frames = self.frames_list[id]
        label = self.labels[id]
        return frames, label

    def create_time_segments(self, vid_path, accident_start_time):
        """Creates time segments from video before an accident start time."""
        time_segments = []
        vid_segments = []
        current_time = 0.0
        
        # Start segmenting 6 seconds before accident, if possible
        if accident_start_time > 6 + self.segment_duration:
            current_time = accident_start_time - 6 - self.segment_duration
        
        # Create overlapping time segments
        while current_time + self.segment_duration <= accident_start_time:
            time_segments.append(accident_start_time - current_time - self.segment_duration)    
            vid_segments.append([vid_path, current_time, current_time + self.segment_duration, None, None, None])
            current_time += self.segment_interval
        
        if not time_segments:
            return None
        
        time_segments = torch.stack([torch.tensor(x, dtype=torch.float32) for x in time_segments])
        return time_segments, vid_segments
    
    def prepare_data(self):
        """Processes the dataset and prepares video segments and labels."""
        for i in tqdm(range(self.length_videos)):
            # Ensure segment duration is valid and the video file exists
            if 5 >= self.segment_duration and os.path.exists(f"{self.video_dir}\\{self.df['File Name'][i]}.mp4"):
                temp = self.create_time_segments(f"{self.video_dir}\\{self.df['File Name'][i]}.mp4", self.df['ttc'][i])
                if temp is not None:
                    self.frames_list.append(temp[1])
                    self.labels.append(temp[0])
        
        # Flatten list and convert labels to tensor
        self.labels = torch.cat(self.labels)
        self.frames_list = [item for sublist in self.frames_list for item in sublist]
        self.frames_list = [self.frames_list[i] + [self.labels[i]] for i in range(len(self.labels))]
        
        print("Frames List Length: ", len(self.frames_list), "Labels Length: ", len(self.labels))
        print("Mean Duration: ", torch.mean(self.labels), "Std Dev of Duration: ", torch.std(self.labels))
        
        if self.task == 'train':
            """Handles class balancing by oversampling underrepresented bins."""
            bin_edges = np.arange(min(self.labels), max(self.labels), 0.5)
            video_groups = {i: [] for i in range(len(bin_edges))}
            
            # Group videos into bins based on time-to-collision (TTC)
            for i, label in enumerate(self.frames_list):
                bin_index = np.digitize([label[-1]], bin_edges)[0] - 1
                video_groups[bin_index].append(label)
            
            print("BEFORE: ", [len(video_groups[x]) for x in video_groups.keys()])
            max_occurrence = max([len(video_groups[x]) for x in video_groups.keys()])
            
            # Oversample underrepresented bins
            for key in video_groups.keys():
                if len(video_groups[key]) < max_occurrence:
                    while len(video_groups[key]) < max_occurrence:
                        duplicated_sublist = [sublist[:-4] + [random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.8, 1.2), random.uniform(0, 0.2)] + [sublist[-1]] for sublist in video_groups[key]]
                        video_groups[key] += duplicated_sublist
                    video_groups[key] = video_groups[key][:max_occurrence]
            
            # Shuffle and finalize labels
            video_groups = [value for values in video_groups.values() for value in values]
            random.shuffle(video_groups)
            self.labels = [inner_list[-1] for inner_list in video_groups]
            self.frames_list = [inner_list[:-1] for inner_list in video_groups]
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
            
            print("Mean Duration: ", torch.mean(self.labels), "Std Dev of Duration: ", torch.std(self.labels))
            print("Frames List Length: ", len(self.frames_list), "Labels Length: ", len(self.labels))
            print(self.frames_list[0])


# Crops the tensor to desired shape
def crop_tensor(video_tensor, target_frames):
    frame_indices = torch.linspace(0, 30 - 1, target_frames).long()
    return video_tensor[frame_indices]


# Load and augment video segments
def load_and_augment_video(x, num_frames=30, target_frames=30):
    video_path = x[0]
    start = x[1]
    end = x[2]
    segment_frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(start * frame_rate)
    end_frame = int(end * frame_rate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # print("TEST -> ", x[0], start_frame, end_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = transforms.ToTensor()(frame)
        if x[3]:
            frame = transforms.ColorJitter(brightness=x[3], contrast=x[4], saturation = x[5], hue=x[-1])(frame)
        segment_frames.append(frame)   

    video = torch.stack(segment_frames)    
    video = crop_tensor(video, target_frames)
    return video



# Data collator for Training data
class TrainCollator():
    def __init__(self, model_type, target_size):
        self.model_type = model_type
        self.target_size = target_size

    def __call__(self, batch):
        # Adjust dimensions for specific models
        if self.model_type in ['VideoSwin', 'Hiera']:
            dims_shape = [0, 2, 1, 3, 4] # FOR Hiera, VideoSwin Transformers
        else:
            dims_shape = [0, 1, 2, 3, 4] # FOR Rest of the models
        if self.model_type in ['VideoFocalNets', 'VideoSwin']:
            target_frames = 8
        else:
            target_frames = 16

        # Process Pipeline for the video segments 
        train_trans = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                video_transforms.RandomCrop(700),
                video_transforms.Resize(self.target_size, antialias=True),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Load and Process the Videos
        temp = [load_and_augment_video(video_path, target_frames=target_frames) for video_path, label in batch]
        transformed_video = torch.stack([train_trans(video) for video in temp])
        return transformed_video.permute(*dims_shape), torch.tensor([label for _, label in batch])


# Data collator for Testing data
class TestCollator():
    def __init__(self, model_type, target_size):
        self.model_type = model_type
        self.target_size = target_size

    def __call__(self, batch):
        # Adjust dimensions for specific models
        if self.model_type in ['VideoSwin', 'Hiera']: 
            dims_shape = [0, 2, 1, 3, 4] # FOR Hiera, VideoSwin Transformers
        else:
            dims_shape = [0, 1, 2, 3, 4] # FOR Rest of the models
        if self.model_type in ['VideoFocalNets', 'VideoSwin']:
            target_frames = 8
        else:
            target_frames = 16

        # Process Pipeline for the video segments    
        test_trans = video_transforms.Compose([
            video_transforms.Resize(self.target_size, antialias=True),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load and Process the Videos
        temp = [load_and_augment_video(video_path, target_frames=target_frames) for video_path, label in batch]
        transformed_video = torch.stack([test_trans(video) for video in temp])

        return transformed_video.permute(*dims_shape), torch.tensor([label for _, label in batch])