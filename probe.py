import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# Custom Dataset Class for Lazy Loading CSV files
class AudioDataset(Dataset):
    def __init__(self, block_folder_base, label_folder_base, boolean_folder_base, severity_mapping, severity_level, block_num, feature_column='F0semitoneFrom27.5Hz_sma3nz', num_frames=10):
        self.block_folder_base = block_folder_base
        self.label_folder_base = label_folder_base
        self.boolean_folder_base = boolean_folder_base  # Path to the folder containing boolean arrays
        self.severity_mapping = severity_mapping
        self.severity_level = severity_level
        self.block_num = block_num
        self.num_frames = num_frames  # Number of random frames to sample
        self.target_feature = feature_column
        self.file_pairs = self._gather_file_pairs()

    def _gather_file_pairs(self):
        pairs = []
        block_folder = os.path.join(self.block_folder_base, f'block_{self.block_num}')
        if not os.path.exists(block_folder):
            return pairs  # Return empty if block folder doesn't exist

        for speaker_id in os.listdir(block_folder):
            speaker_severity = self.severity_mapping.get(speaker_id)
            if speaker_severity != self.severity_level:
                continue  # Skip speakers that don't match the current severity
            
            speaker_data_folder = os.path.join(block_folder, speaker_id)
            if not os.path.exists(speaker_data_folder):
                continue
            
            for session in os.listdir(speaker_data_folder):
                session_data_folder = os.path.join(speaker_data_folder, session)
                for data_file in os.listdir(session_data_folder):
                    data_file_path = os.path.join(session_data_folder, data_file)
                    
                    # Find corresponding label and boolean files
                    label_file_path = os.path.join(self.label_folder_base, speaker_id, session, data_file)
                    boolean_file_path = os.path.join(self.boolean_folder_base, speaker_id, session) + ".csv"
                    
                    print(f"Data: {data_file_path}, Label: {label_file_path}, Boolean: {boolean_file_path}")
                    if not os.path.exists(label_file_path) or not os.path.exists(boolean_file_path):
                        print(f"Label or boolean file not found for {data_file_path}, skipping...")
                        continue

                    # Read the boolean DataFrame and filter for speech frames
                    boolean_df = pd.read_csv(boolean_file_path, header=None)
                    boolean_df[4] = pd.to_numeric(boolean_df[4], errors='coerce') 
                    boolean_df[3] = pd.to_numeric(boolean_df[3], errors='coerce') 
                    
                    speech_file = os.path.basename(data_file_path).replace('.csv', '')
                    
                    # Get indices of speech frames
                    speech_indices = boolean_df[(boolean_df[4] == 1.0) & (boolean_df[2] == speech_file)][3].tolist()

                    # Check if the number of speech frames meets the requirement
                    if len(speech_indices) == 0 :
                        print(f"Warning: Not enough speech frames in {data_file_path}. Found {len(speech_indices)}, required {self.num_frames}.")
                        continue  # Skip this pair if not enough speech frames

                    # Add pair to the list only if it passes the above checks
                    pairs.append((data_file_path, label_file_path, boolean_file_path))

        random.shuffle(pairs)
        print(f"Gathered {len(pairs)} file pairs for severity {self.severity_level}")  # Debugging

        return pairs
    
    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        # Get the input, label, and boolean file paths
        input_file_path, label_file_path, boolean_file_path = self.file_pairs[idx]

        # Load input, label, and boolean CSV files
        try:
            input_df = pd.read_csv(input_file_path, header=None)
            label_df = pd.read_csv(label_file_path)
            boolean_df = pd.read_csv(boolean_file_path, header=None)

            speech_file = os.path.basename(input_file_path).replace('.csv', '')
            # Filter out non-speech frames
            boolean_df[4] = pd.to_numeric(boolean_df[4], errors='coerce') 
            boolean_df[3] = pd.to_numeric(boolean_df[3], errors='coerce') 
            speech_indices = boolean_df[(boolean_df[4] == 1.0) &  (boolean_df[2] == speech_file)][3].tolist()

            if len(speech_indices) < self.num_frames:
                print(f"Warning: Less than {self.num_frames} speech frames found in {input_file_path}. Found {len(speech_indices)}. Using all available indices.")
                selected_indices = speech_indices  # Use all available speech indices
            else:
                # Select random indices if there are enough speech frames
                selected_indices = random.sample(speech_indices, self.num_frames)

            # Select the corresponding frames from input and label
            X_selected = input_df.iloc[selected_indices].values
            y_selected = label_df[self.target_feature].iloc[selected_indices].values

            # Convert selected data to tensors
            X_tensor = torch.tensor(X_selected).float()
            y_tensor = torch.tensor(y_selected).float()

            return X_tensor, y_tensor  # Return the input and label tensors

        except Exception as e:
            print(f"Error loading data from {input_file_path}, {label_file_path}, or {boolean_file_path}: {e}")
            return None  # In case of error, return None

# Function to split dataset into train and test
def train_test_split(dataset, test_size=0.2):
    total_size = len(dataset)
    test_size = int(total_size * test_size)
    train_size = total_size - test_size
    
    train_indices = random.sample(range(total_size), train_size)
    test_indices = list(set(range(total_size)) - set(train_indices))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
# Define your model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Function to append results to CSV
def append_to_csv(result_list, severity):
    df = pd.DataFrame(result_list)
    file_name = f'results_{severity}.csv'
    
    if not os.path.isfile(file_name):
        df.to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, mode='a', header=False, index=False)

# Function to evaluate the model on the test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if X_batch is None:
                continue

            X_batch = X_batch.squeeze(0)  # Shape will become [rows, 1024]
            y_batch = y_batch.squeeze(0)
            y_batch = y_batch.unsqueeze(1)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader)
    return average_loss

# Example function to train model per severity and block with patience and appending mechanism
def train_model(block_folder_base, label_folder_base, bool_folder_base, severity_mapping, severities, features, num_epochs=10, patience=5, test_size=0.2):
    for severity in severities:
        for block_num in range(1, 26):  # Iterate over each block
            print(f"Training for Severity: {severity}, Block: {block_num}")

            for feature in features:  # Iterate over each feature
                print(f"Training for Feature: {feature}")

                # Initialize dataset for the current severity, block, and feature
                dataset = AudioDataset(block_folder_base, label_folder_base, bool_folder_base, severity_mapping, severity, block_num, feature_column=feature)
                print(f"Dataset has been loaded len: {len(dataset)}")

                if len(dataset) == 0:
                    print("Dataset is empty, skipping...")
                    continue
                
                # Split the dataset into train and test sets
                train_dataset, test_dataset = train_test_split(dataset, test_size)
                print(f"Train Dataset Size: {len(train_dataset)}, Test Dataset Size: {len(test_dataset)}")

                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=my_collate)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate)

                print("Data loaders are set")
                
                model = SimpleNN(1024, hidden_size=128, output_size=1).to(device)  # Adjust input size as needed
                print("Model initialized")
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                print("Optimizer set up")
                
                best_loss = float('inf')
                no_improvement_epochs = 0
                feature_results = []

                # Training loop for each severity, block, and feature
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0

                    for X_batch, y_batch in train_loader:
                        if X_batch is None:
                            print("Fetched X_batch is None, skipping...")
                            continue

                        X_batch = X_batch.squeeze(0)  # Shape will become [rows, 1024]
                        y_batch = y_batch.squeeze(0)
                        y_batch = y_batch.unsqueeze(1)
                        
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    epoch_loss = running_loss / len(train_loader)
                    print(f"Block: {block_num}, Severity: {severity}, Feature: {feature}, Epoch {epoch+1}, Loss: {epoch_loss}")

                    # Evaluate the model on the test dataset
                    test_loss = evaluate_model(model, test_loader, criterion)
                    print(f"Test Loss after Epoch {epoch + 1}: {test_loss}")

                    # Store results in memory
                    feature_results.append({
                        'Block': block_num,
                        'Severity': severity,
                        'Feature': feature,
                        'Epoch': epoch + 1,
                        'Loss': epoch_loss,
                        'Test Loss': test_loss
                    })

                    # Early stopping based on test loss
                    if test_loss < best_loss:
                        best_loss = test_loss
                        no_improvement_epochs = 0
                    else:
                        no_improvement_epochs += 1
                    
                    if no_improvement_epochs >= patience:
                        print("Early stopping triggered")
                        break

                # Append the results to a CSV file after training
                append_to_csv(feature_results, severity)

# Entry point
if __name__ == "__main__":
    block_folder_base = './data_folder_whole'
    label_folder_base = './labels/TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO'
    boolean_folder_base = './voice_activity_output'  # Update this with your actual path

    severity_mapping = {
        'FC01': 'Normal', 'FC02': 'Normal', 'FC03': 'Normal', 'FC04': 'Normal',
        'FC05': 'Mild', 'FC06': 'Mild', 'FC07': 'Mild', 'FC08': 'Mild',
        'FC09': 'Moderate', 'FC10': 'Moderate', 'FC11': 'Moderate', 'FC12': 'Moderate',
        'FC13': 'Severe', 'FC14': 'Severe', 'FC15': 'Severe', 'FC16': 'Severe'
    }

    features = [
    'F0semitoneFrom27.5Hz_sma3nz',
    'spectralFlux_sma3',
    'HNRdBACF_sma3nz',
    'F2frequency_sma3nz',
    'F2bandwidth_sma3nz',
    'F3bandwidth_sma3nz',
    'logRelF0-H1-A3_sma3nz',
    'logRelF0-H1-H2_sma3nz',
    'slope500-1500_sma3',
    'Loudness_sma3',
    'mfcc3_sma3'
    ]

    severities = ['Normal', 'Very low', 'Low', 'Medium']  # List of severities to train on
    train_model(block_folder_base, label_folder_base, boolean_folder_base, severity_mapping, severities, features, num_epochs=100)
