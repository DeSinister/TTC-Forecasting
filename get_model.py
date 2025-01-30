import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Create a class for the Model with Downstream task.
class VideoEncoder(nn.Module):
    def __init__(self, model_type='VideoSwin', clamp_limit=6.0):
        super().__init__() 
        self.clamp_limit = clamp_limit
        embed_dims = None

        # Load the appropriate video transformer model and set the embedding dimension
        if model_type == 'VideoFocalNets':  
            from Video_FocalNets.video_focalnet import videofocalnet_b
            self.vid_trans = videofocalnet_b  # Load Video FocalNet model
            embed_dims = 1024  # Set embedding dimension
        
        if model_type == 'VideoSwin':
            from Video_SwinTransformer.swin import swin_base
            self.vid_trans = swin_base  # Load Video Swin Transformer model
            embed_dims = 1024  # Set embedding dimension
        
        if model_type == 'Hiera':
            from Hiera.hiera import hiera_b
            self.vid_trans = hiera_b  # Load Hiera model
            embed_dims = 768  # Set embedding dimension
    

        # Define the final regression head for predicting Time-to-Collision (TTC)
        self.final_layer = nn.Sequential(
            nn.ReLU(),  # Apply ReLU activation
            nn.Linear(embed_dims, 64),  # Linear layer to reduce embedding size
            nn.ReLU(),  # Apply ReLU activation
            nn.Dropout(),  # Apply dropout for regularization
            nn.Linear(64, 1)  # Final linear layer to output a single scalar value
        )
    
    
    # Forward Method
    def forward(self, x):
        # Extract video features using the selected transformer model
        x = self.vid_trans(x)       
        
        # Pass features through the regression head to predict the TTC score
        x = self.final_layer(x)
        x = torch.clamp(x, min=0.0, max=self.clamp_limit)
        return x
