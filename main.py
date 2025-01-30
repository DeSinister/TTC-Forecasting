import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from data_loader import FetchData, TestCollator, TrainCollator
from get_model import VideoEncoder
from utils import train_model, evaluate_model, epoch_time
import time
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set the Seed
torch.manual_seed(42)
np.random.seed(42)



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4, help="Batch Size")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Number of Epochs")
    parser.add_argument('-e_p', '--early_stop_patience', type=int, default=15, help="Patience for Early Stopping")
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-6, help="Learning Rate")
    parser.add_argument('-lr_p', '--learning_rate_patience', type=int, default=5, help="Patience for Decaying Learning Rate")
    parser.add_argument('-dev', '--device', type=str, default='cuda', help="Device type: 'cpu', 'cuda'")
    parser.add_argument('-w', '--num_workers', type=int, default=1, help="Number of Worker Threads")
    parser.add_argument('-img_size', '--target_size', type=str, default='224x224', help="Image Dimension required by the model")
    parser.add_argument('-seg_len', '--segment_length', type=float, default=1, help="Video Sequence Segment Length in Seconds")
    parser.add_argument('-seg_over', '--segment_overlap', type=float, default=0.5, help="Video Sequence Overlap Length in Seconds")
    parser.add_argument('-model', '--model_type', type=str, default='Hiera', help="Type of the model: ['VideoFocalNets', 'Hiera', 'VideoSwin']")
    parser.add_argument('-dataset', '--dataset', default='DoTA', type=str, help="Name of teh Dataset: ['CCD', 'DoTA', 'DAD']")
    parser.add_argument('-vid_dir', '--video_dir', type=str, help="Path for the Video Directory")
    args = parser.parse_args()

    # Store the Extracted inforamtion
    batch_size = args.batch_size
    num_of_epochs = args.epochs
    learning_rate = args.learning_rate
    learning_rate_patience = args.learning_rate_patience
    early_stop_patience = args.early_stop_patience
    device = torch.device(args.device)
    num_workers = args.num_workers
    dataset = args.dataset
    target_size = [int(x) for x in args.target_size.lower().split('x')]
    segment_length = args.segment_length
    segment_interval = args.segment_overlap
    model_type= args.model_type


    # Verification of the Parsed Arguments
    if len(target_size) != 2:
        print("INVALID IMAGE DIMENSION")
    if args.video_dir:
        video_dir = args.video_dir
    else:
        print("VIDEO DIRECTORY IS MISSING")
   
    if dataset == 'CCD':
        max_vid_duration = 5.0
    elif dataset == 'DAD':
        max_vid_duration = 4.0
    elif dataset == 'DoTA':
        max_vid_duration = 6.0
    else:
        print("INVALID DATASET")


    # SET UP DATALOADER
    train_dataset = FetchData(dataset, video_dir, task = 'train', segment_duration=segment_length, segment_interval=segment_interval, target_size=target_size)
    train_collate_fn = TrainCollator(model_type=model_type, target_size=target_size)
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last = True,
            pin_memory=False,
            collate_fn=train_collate_fn,
        )   
    test_dataset = FetchData(dataset, video_dir, task = 'val', segment_duration= segment_length, segment_interval=segment_interval, target_size=target_size)
    test_collate_fn = TestCollator(model_type=model_type, target_size=target_size)
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last = True,
            pin_memory=False,
            collate_fn=test_collate_fn,
        )
    

    # --- Model Configurations ---
    model = VideoEncoder(model_type = model_type, clamp_limit=max_vid_duration).to(device)
    model = nn.DataParallel(model)
    
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=learning_rate_patience, verbose=True)

    # Set up the Metrics
    loss_function = nn.MSELoss()
    highest_score = highest_score2 = np.inf
    lowest_loss = np.inf
    bad_epochs = 0
    early_stop = False
    end_epoch = num_of_epochs

    # Loop for Every epochs
    for epoch in range(num_of_epochs):
        # Start Counting time
        start = time.time()

        # Train the Model for Every epoch
        train_value = train_model(model, train_loader, optimizer, loss_function, device, scaler, opt_step_size=4).detach().cpu()

        # Evaluate the Model using the test Split
        test_value = evaluate_model(model, test_loader, loss_function, device).detach().cpu()
        
        # Save the Model If the Model is Performing better on test set while Training
        if test_value < lowest_loss:
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"test Loss Decreased from {lowest_loss} to {test_value}")
            # Changing the Lowest Loss to Current test Loss
            lowest_loss = test_value
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_TTC_{epoch}_loss{test_value}.pt")
        else:
            # Model not performning better for this epoch
            bad_epochs+=1
        # Stop the Counting Time
        end = time.time()

        # Estimate the Duration
        minutes, seconds = epoch_time(start, end)

        # Report Training and test Loss
        print(f"Epoch Number: {epoch+1}")
        print(f"Duration: {minutes}m {seconds}s")
        print(f"Training Loss: {train_value}")
        print(f"test Loss: {test_value}")
        print()

        # If Patience Level reached for Model not Performing better
        if bad_epochs == early_stop_patience:
            print("Stopped Early. The Model is not improving over test loss")
            end_epoch = epoch
            break
