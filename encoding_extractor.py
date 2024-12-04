import torch
import soundfile as sf
from scipy.signal import resample
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperFeatureExtractor

def load_model_and_hooks(model_size='medium'):
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
    model.eval()
    model.encoder_only = True
    return model

def adjust_model_config(num_frames, model_size='medium'):
    model = load_model_and_hooks(model_size)
    state_dict = model.state_dict()
    
    # Adjust positional embeddings to match the num_frames
    state_dict["model.encoder.embed_positions.weight"] = state_dict["model.encoder.embed_positions.weight"][:num_frames, :]
    config = WhisperConfig.from_pretrained(f"openai/whisper-{model_size}", max_source_positions=num_frames)
    
    # Load the model with the new configuration
    new_model = WhisperForConditionalGeneration(config)
    new_model.load_state_dict(state_dict)
    new_model.eval()
    new_model.encoder_only = True
    return new_model

def process_audio(audio_path, model_size='medium', target_sr=16000):
    print(f"Processing audio file: {audio_path}")

    # Load audio
    audio, sr = sf.read(audio_path)

    # Resample if the sample rate doesn't match target_sr
    if sr != target_sr:
        audio = resample(audio, int(len(audio) * float(target_sr) / sr))
        sr = target_sr

    # Convert to tensor and handle stereo channels
    audio = torch.tensor(audio).float()
    if audio.ndim > 1:
        audio = torch.mean(audio, dim=1)

    # Whisper expects 100 frames per second, calculate num_frames
    duration = len(audio) / sr
    num_frames = int(duration * 100)  
    if duration * 100 - num_frames >= 0.5:
        num_frames += 1 

    # Adjust model configuration
    model = adjust_model_config(num_frames, model_size)

    decoder_input_ids = torch.tensor([[1]] * model.config.decoder_start_token_id)
    feature_extractor = WhisperFeatureExtractor(chunk_length=len(audio)//sr)

    try:
        # Feature extraction
        inputs = feature_extractor(audio.flatten(), return_tensors="pt", sampling_rate=sr)

        # Inference
        with torch.no_grad():
            outputs = model(inputs.input_features, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

        hidden_states = outputs.encoder_hidden_states
        return hidden_states, num_frames

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        if "size of tensor" in str(e):
            error_msg = str(e)
            # Extract sizes from the error message
            try:
                a_size = int(error_msg.split('tensor a (')[1].split(')')[0])    
                b_size = int(error_msg.split('tensor b (')[1].split(')')[0])

                # Use the smaller size to retry
                num_frames = b_size
                print(f"Retrying with corrected num_frames={num_frames}")
                model = adjust_model_config(num_frames, model_size)
                retry_attempts -= 1  # Decrement the retry attempts

            except (IndexError, ValueError) as inner_e:
                print(f"Failed to parse sizes from error: {inner_e}")
        else:
            print(f"Error: {e}")

    return None, None