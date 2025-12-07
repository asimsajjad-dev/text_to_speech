import gradio as gr
import numpy as np
import scipy.io.wavfile as wav
import io
import os
import re
from typing import Tuple, List

# --- COQUI TTS IMPORTS ---
try:
    from TTS.api import TTS
    from pydub import AudioSegment # Re-introduce for combining pauses
except ImportError as e:
    print(f"FATAL ERROR: TTS or pydub not found. Check Dockerfile/requirements.txt. Error: {e}")
    exit()

# --- MODEL LOADING (VITS Multi-Speaker) ---

# We load the model globally so it only loads once when the app starts.
MODEL_NAME = "vits--multilingual" 
tts_model = None
try:
    print(f"Loading VITS Multi-Speaker model ({MODEL_NAME})... This may take a moment.")
    tts_model = TTS(MODEL_NAME)
    print("VITS model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load VITS model: {e}")
    # The app may still launch, but the function will fail.

# --- UTILITY FUNCTIONS ---

def insert_pause(duration_ms):
    """Creates a silent pydub AudioSegment."""
    return AudioSegment.silent(duration=duration_ms)

def process_text_with_tags(text: str) -> Tuple[List[str], List[int]]:
    """Splits text by [pause Xs] tags."""
    pause_pattern = r'\[pause (\d+)s\]'
    segments = re.split(pause_pattern, text)
    processed_segments = []
    pauses = []
    i = 0
    while i < len(segments):
        segment = segments[i].strip()
        if segment:
            processed_segments.append(segment)
        i += 1
        if i < len(segments):
            pause_sec = int(segments[i])
            pauses.append(pause_sec * 1000)
            i += 1
    return processed_segments, pauses


# --- GRADIO CORE FUNCTION ---
def generate_voiceover(audio_file_path: str, text_with_tags: str) -> str:
    """
    Main function called by Gradio. 
    Returns the path to the generated WAV file.
    """
    if tts_model is None:
        raise gr.Error("TTS Model failed to load. Check logs.")
        
    if not audio_file_path:
        raise gr.Error("Please upload a Reference Voice Audio file.")
    if not text_with_tags.strip():
        raise gr.Error("Text input is required.")

    print(f"Processing audio for text: {text_with_tags[:30]}...")

    # 1. Process Text
    text_segments, pauses_ms = process_text_with_tags(text_with_tags)
    
    combined_audio = AudioSegment.empty()
    SAMPLE_RATE_VITS = 22050 # VITS models often use 22.05 kHz

    # 2. Generate and Combine Audio Segments
    for i, segment in enumerate(text_segments):
        
        # Synthesis call
        wav_array = tts_model.synthesize(
            segment,             
            speaker_wav=audio_file_path,  # Reference voice audio path
            language="en"
        )
        
        # Convert NumPy array to pydub AudioSegment
        buffer = io.BytesIO()
        wav.write(buffer, SAMPLE_RATE_VITS, np.array(wav_array, dtype=np.int16)) 
        buffer.seek(0)
        audio_seg = AudioSegment.from_wav(buffer)
        
        # Add audio segment
        combined_audio += audio_seg
        
        # Add pause AFTER the segment (if it's not the last one)
        if i < len(pauses_ms):
            combined_audio += insert_pause(pauses_ms[i])

    # 3. Export Audio to a temporary file
    output_filename = "vits_voiceover.wav"
    combined_audio.export(output_filename, format="wav")
    
    # 4. Return the file path for Gradio
    return output_filename

# --- GRADIO INTERFACE DEFINITION ---

iface = gr.Interface(
    fn=generate_voiceover,
    inputs=[
        gr.Audio(label="Reference Voice Audio (Voice Cloning Sample)", type="filepath"),
        gr.Textbox(
            label="Text to Speak (Use [pause Xs] tags)",
            value="This is a free voiceover using the VITS model. [pause 1s] I hope you enjoy the quality!",
            info="The [pause Xs] tag (where X is seconds) will insert silence."
        )
    ],
    outputs=gr.Audio(label="Generated Voiceover", type="filepath"),
    title="VITS Multi-Speaker Voice Cloning (Gradio)",
    description="Deploying VITS via Docker on Hugging Face Spaces for fast, free voice cloning."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0") # Use 0.0.0.0 for compatibility with HF Spaces
