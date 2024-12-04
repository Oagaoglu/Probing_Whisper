import os
import pandas as pd
import opensmile
from wav_loader import list_wav_files

def extract_features(wav_file, output_file):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    try:
        y = smile.process_file(wav_file)
        features_to_keep = [
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
        
        y_filtered = y[features_to_keep]
        y_reset = y_filtered.reset_index()
        y_reset.to_csv(output_file, index=False)
        return True  # Return success
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return False  # Return failure

def process_all_wav_files(base_directory, output_directory):
    audio_files = list_wav_files(base_directory)
    unprocessed_count = 0
    unmatched_files = []  # To track unmatched files
    missing_csvs = []  # To track files missing CSVs

    for group_type, files in audio_files.items():
        for file_path in files:
            path_parts = file_path.split(os.sep)
            speaker_id = path_parts[-4]
            session_id = path_parts[-3]
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            speaker_dir = os.path.join(output_directory, speaker_id)
            session_dir = os.path.join(speaker_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)

            output_file = os.path.join(session_dir, f"{file_name}.csv")

            if not os.path.exists(output_file):
                # If the CSV does not exist, attempt to extract features
                if extract_features(file_path, output_file):
                    unprocessed_count += 1
                else:
                    unmatched_files.append(file_path)  # Log unmatched file
            
    print(f"Number of unprocessed files processed: {unprocessed_count}")

    # Check for missing CSVs
    expected_csvs = {os.path.join(session_dir, f"{os.path.splitext(os.path.basename(f))[0]}.csv")
                     for group_type, files in audio_files.items()
                     for f in files
                     for session_dir in [os.path.join(output_directory, path_parts[-4], path_parts[-3])]}

    existing_csvs = {os.path.join(session_dir, f"{os.path.splitext(os.path.basename(f))[0]}.csv")
                     for group_type, files in audio_files.items()
                     for f in files
                     for session_dir in [os.path.join(output_directory, path_parts[-4], path_parts[-3])]}

    missing_csvs = expected_csvs - existing_csvs

    # Report unmatched files and missing CSVs
    if unmatched_files:
        print("Unmatched WAV files that could not be processed:")
        for unmatched in unmatched_files:
            print(unmatched)

    if missing_csvs:
        print("WAV files missing corresponding CSVs, attempting to process:")
        for missing in missing_csvs:
            # Attempt to process missing WAV files again
            original_wav_file = os.path.splitext(missing)[0] + ".wav"  # Assuming original WAV file exists
            if os.path.exists(original_wav_file):
                if extract_features(original_wav_file, missing):
                    print(f"Successfully processed {original_wav_file} into {missing}.")
                else:
                    print(f"Failed to process {original_wav_file}.")
            else:
                print(f"Original WAV file does not exist for {missing}.")

if __name__ == "__main__":
    base_directory = "./TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO/"
    output_directory = "./labels/"
    process_all_wav_files(base_directory, output_directory)
    print("Feature extraction completed.")
