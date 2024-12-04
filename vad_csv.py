import os
import csv
import soundfile as sf
from vad import EnergyVAD  # Ensure this is the correct import for your EnergyVAD implementation

def process_audio_file(file_path):
    try:
        # Load the audio file
        audio, sample_rate = sf.read(file_path)
        
        # Check if audio data is empty
        if len(audio) == 0:
            print(f"Warning: {file_path} is empty. Skipping.")
            return []  # Return an empty list to indicate no voice activity

        # Initialize the EnergyVAD
        vad = EnergyVAD(sample_rate=16000, frame_length=20, frame_shift=10, energy_threshold=0.01)

        # Process the audio to detect voice activity
        voice_activity = vad(audio)
        return voice_activity
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []  # Return an empty list for files that cannot be processed

def write_to_csv(output_file, speaker_id, session_id, voice_activity):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Speaker ID', 'Session ID', 'File Name', 'Frame', 'Voice Activity'])
        for frame_index, is_active in enumerate(voice_activity):
            writer.writerow([speaker_id, session_id, os.path.basename(output_file).replace('.csv', ''), frame_index, is_active])

def process_directory(base_directory, output_directory):
    for speaker_id in os.listdir(base_directory):
        speaker_path = os.path.join(base_directory, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        
        for session_id in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session_id)
            if not os.path.isdir(session_path):
                continue
            
            # Check for the wav_arrayMic subdirectory
            wav_array_path = os.path.join(session_path, 'wav_arrayMic')
            if not os.path.isdir(wav_array_path):
                continue  # Skip if the wav_arrayMic directory does not exist
            
            # Prepare the output CSV file path for the session
            csv_filename = os.path.join(output_directory, speaker_id, f"{session_id}.csv")
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

            # Initialize a list to collect all voice activity data for this session
            session_voice_activity = []

            for file_name in os.listdir(wav_array_path):
                if file_name.endswith(('.wav', '.flac', '.ogg')):  # Supported formats
                    file_path = os.path.join(wav_array_path, file_name)
                    print(f"Processing file: {file_path}")
                    
                    # Process audio file
                    voice_activity = process_audio_file(file_path)

                    # Collect the voice activity for this session
                    session_voice_activity.append((os.path.splitext(file_name)[0], voice_activity))

            # Write results to CSV for this session
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Speaker ID', 'Session ID', 'File Name', 'Frame', 'Voice Activity'])
                for file_name, voice_activity in session_voice_activity:
                    for frame_index, is_active in enumerate(voice_activity):
                        writer.writerow([speaker_id, session_id, file_name, frame_index, is_active])

def main():
    base_directory = './TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO'
    output_directory = './voice_activity_output'
    
    process_directory(base_directory, output_directory) 

if __name__ == "__main__":
    main()