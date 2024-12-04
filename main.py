import os
import pandas as pd
import numpy as np
from encoding_extractor import load_model_and_hooks, process_audio
from wav_loader import list_wav_files
#from f_extractor import extract_features
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Mapping of speaker IDs to severity levels
severity_mapping = {
    'FC01': 'Normal', 'FC02': 'Normal', 'FC03': 'Normal',
    'MC01': 'Normal', 'MC02': 'Normal', 'MC03': 'Normal', 'MC04': 'Normal',
    'F04': 'Very low', 'M03': 'Very low',
    'F03': 'Low', 'M05': 'Low',
    'F01': 'Medium', 'M01': 'Medium', 'M02': 'Medium', 'M04': 'Medium'
}

def save_to_csv(data, csv_filename, headers=None):
    """Saves block data to a CSV file."""
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(csv_filename, mode='w', header=headers is not None, index=False)

def process_and_save_files(group_files, group_name, model, output_dir):
    total_files = len(group_files)
    print("Total files:", total_files)
    for index, file_path in enumerate(group_files):
        if (index + 1) % 10 == 0 or index + 1 == total_files:
            remaining_files = total_files - index - 1
            print("Processed {}/{} files. {} files remaining.".format(index + 1, total_files, remaining_files))
        
        file_id = os.path.basename(file_path).replace('.wav', '')
        session_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
        severity = severity_mapping.get(speaker_id, 'Unknown')
        print(file_path)
        print(file_id)
        hidden_states, num_frames = process_audio(file_path)

        if hidden_states is None or num_frames is None:
            print("Skipping file {} due to processing error.".format(file_path))
            continue

        for i, block_output in enumerate(hidden_states):
            # Save the entire sequence of hidden states
            block_dir = os.path.join(output_dir, "block_{}".format(i + 1), speaker_id, session_id)
            os.makedirs(block_dir, exist_ok=True)

            csv_filename = os.path.join(block_dir, "{}.csv".format(file_id))
            if os.path.exists(csv_filename):
                print("File {} already processed. Skipping.".format(csv_filename))
                continue

            valid_data = block_output[0, :num_frames, :].cpu().numpy()  # Use only valid frames
            headers = ['frame'] + ['feature_{}'.format(j) for j in range(valid_data.shape[1])]

            save_to_csv(valid_data, csv_filename)

def load_processed_files(file_path):
    """
    Loads processed files from a text file.

    Parameters:
    - file_path (str): Path to the processed files text file.

    Returns:
    - set: A set of tuples representing already processed files.
    """
    processed_files = set()
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                speaker_id, session_id, file_id = parts
                processed_files.add((speaker_id, session_id, file_id))
    
    return processed_files

def filter_wav_files(wav_files, processed_files):
    """
    Filters out processed files from the list of WAV files.

    Parameters:
    - wav_files (dict): Dictionary containing lists of WAV file paths categorized by 'control' and 'experimental'.
    - processed_files (set): Set of tuples representing already processed files.

    Returns:
    - dict: Dictionary with filtered lists of WAV file paths.
    """
    filtered_files = {'control': [], 'experimental': []}
    
    for group, files in wav_files.items():
        for file_path in files:
            file_name = os.path.basename(file_path)  # Get the file name with extension
            file_id, _ = os.path.splitext(file_name)  # Remove the .wav extension
            session_id = os.path.basename(os.path.dirname(file_path))  # Extract session ID
            speaker_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Extract speaker ID
            
            # Prepare tuple for comparison
            file_tuple = (speaker_id, session_id, file_id)
            
            if file_tuple not in processed_files:
                filtered_files[group].append(file_path)
    
    return filtered_files

def main():
    base_directory = './TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO'
    output_directory = './data_folder_whole'
    processed_files_path = './processed_files.txt'
    
    # Load processed files
    processed_files = load_processed_files(processed_files_path)
    
    # Load model
    model = load_model_and_hooks('medium')
    
    # List and filter WAV files
    wav_files = list_wav_files(base_directory)
    wav_files = filter_wav_files(wav_files, processed_files)
    print("len after filter")
    print(len(wav_files["control"]))
    
    # Process control group files
    print("Processing control group files...")
    #process_and_save_files(wav_files['control'], 'control', model, output_directory)

    # Process experimental group files
    print("Processing experimental group files...")
    process_and_save_files(wav_files['experimental'], 'experimental', model, output_directory)

if __name__ == "__main__":
    main()
