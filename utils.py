from tqdm import tqdm
import torch
from torchmetrics.regression import MeanSquaredError, R2Score, ConcordanceCorrCoef
from torchmetrics.classification import Accuracy, MulticlassF1Score


# Converts the String Timestamp to hours, minutes, seconds, centiseconds
def parse_timestamp(timestamp_str):
    # Split the timestamp string into components
    components = timestamp_str.split(':')
    # Extract and convert components to integers
    hours = int(components[0])
    minutes = int(components[1])
    seconds = int(components[2])
    centiseconds = int(components[3])
    return hours, minutes, seconds, centiseconds

# Calculates the difference between 2 given timestamps
def subtract_timestamps(timestamp_str1, timestamp_str2):
    # Parse both timestamps
    hours1, minutes1, seconds1, centiseconds1 = parse_timestamp(timestamp_str1)
    hours2, minutes2, seconds2, centiseconds2 = parse_timestamp(timestamp_str2)
    # Calculate the differences for each component
    hours_diff = hours2 - hours1
    minutes_diff = minutes2 - minutes1
    seconds_diff = seconds2 - seconds1
    centiseconds_diff = centiseconds2 - centiseconds1
    
    # Handle negative differences (borrowing)
    if centiseconds_diff < 0:
        centiseconds_diff += 100
        seconds_diff -= 1
    if seconds_diff < 0:
        seconds_diff += 60
        minutes_diff -= 1
    if minutes_diff < 0:
        minutes_diff += 60
        hours_diff -= 1
    result = hours_diff*3600 + minutes_diff*60 + seconds_diff + centiseconds_diff/100
    return abs(result)

# Calculates time taken for an epoch
def epoch_time(start_time, end_time):
    duration = end_time - start_time
    minutes = duration//60
    seconds = duration - (60*minutes)
    return int(minutes), int(seconds)

# Function to train the model for a given epoch
def train_model(model, loader, optimizer, loss_fn, device, scaler, opt_step_size = 4):
    # Intializing starting Epoch loss as 0
    loss_for_epoch = 0.0     
    model.train()
    iters = 0
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float16)
        y = y.to(device, dtype=torch.float16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            y = torch.unsqueeze(y, 1)
            loss = loss_fn(y_pred, y)/opt_step_size

        # Scale Loss Backwards
        scaler.scale(loss).mean().backward()

        # Unscale the Gradients in Optimizer
        if iters % opt_step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Add the Loss for every sample in a Batch
        loss_for_epoch += loss.item()
        iters+=1

    
    if iters % opt_step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Calculating The Average Loss for the Epoch
    loss_for_epoch =  torch.div(loss_for_epoch, len(loader))
    return loss_for_epoch

# Function to Evaluate the Model
def evaluate_model(model, loader, loss_fn, device):
    # Intializing starting Epoch loss as 0
    total_loss = 0.0

    # Model to be used in Evaluation Mode
    model.eval()

    # Gradients are not calculated
    with torch.no_grad():
        
        # For every Input Image, Label Image in a Batch
        for x, y in tqdm(loader):

            # Storing the Images to the Device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            y = torch.unsqueeze(y, 1)
            total_loss += loss_fn(y_pred, y)

        # Calculating The Average Loss for the Epoch
        total_loss =  torch.div(total_loss, len(loader))
        
    return total_loss
