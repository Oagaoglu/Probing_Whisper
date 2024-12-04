import os
import pandas as pd

def list_wav_files(base_directory):
    """
    Walks through the directory structure starting from the base_directory,
    identifies WAV files within 'wav_arrayMic' subdirectories, and categorizes them
    into 'control' and 'experimental' based on folder naming conventions.

    Parameters:
    - base_directory (str): The root directory to start the search from.

    Returns:
    - dict: A dictionary with keys 'control' and 'experimental', each containing a list of unprocessed WAV file paths.
    """
    processed_files = check_processed_files()
    print(f"Processed files: {len(processed_files)} entries found")

    audio_files = {'control': [], 'experimental': []}

    # Traverse the directory structure based on the provided logic
    for speaker_id in os.listdir(base_directory):
        speaker_path = os.path.join(base_directory, speaker_id)
        if os.path.isdir(speaker_path):
            for session_id in os.listdir(speaker_path):
                session_path = os.path.join(speaker_path, session_id)
                if os.path.isdir(session_path) and session_id.startswith('Session'):
                    wav_dir_path = os.path.join(session_path, 'wav_arrayMic')
                    if os.path.exists(wav_dir_path):
                        # Determine the group type based on the presence of 'C' in the directory name
                        is_control = speaker_id in ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04']
                        group_type = 'control' if is_control else 'experimental'

                        # List all WAV files in the 'wav_arrayMic' directory
                        for file in os.listdir(wav_dir_path):
                            if file.endswith('.wav'):
                                file_id = os.path.splitext(file)[0]

                                if (speaker_id, session_id, file_id) not in processed_files:
                                    audio_files[group_type].append(os.path.join(wav_dir_path, file))

    # Print the counts for verification
    print(f"Control files: {len(audio_files['control'])}")
    print(f"Experimental files: {len(audio_files['experimental'])}")
    
    return audio_files

def check_processed_files(base_directory="./data_folder_whole"):
    """
    Scans the directory structure under the base_directory to identify already processed files.

    Parameters:
    - base_directory (str): The root directory to start the search from.

    Returns:
    - set: A set of tuples (speaker_id, session_id, file_id) representing already processed files.
    """
    processed_files = set()

    # Traverse the directory structure
    for block_id in os.listdir(base_directory):
        block_path = os.path.join(base_directory, block_id)
        if os.path.isdir(block_path) and block_id.startswith('block_'):
            for speaker_id in os.listdir(block_path):
                speaker_path = os.path.join(block_path, speaker_id)
                if os.path.isdir(speaker_path):
                    for session_id in os.listdir(speaker_path):
                        session_path = os.path.join(speaker_path, session_id)
                        if os.path.isdir(session_path):
                            for file in os.listdir(session_path):
                                if file.endswith('.csv'):
                                    file_id = os.path.splitext(file)[0]
                                    processed_files.add((speaker_id, session_id, file_id))
    
    return processed_files

def main():
    base_directory = './TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO/'  # Update this path as necessary
    audio_files = list_wav_files(base_directory)
    print(audio_files)  # Optional: Print the output for verification during development

if __name__ == "__main__":
    main()
