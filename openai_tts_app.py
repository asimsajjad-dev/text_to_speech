import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import io
import os
import re
from typing import Tuple, List

# --- COQUI TTS IMPORTS ---
try:
    from TTS.api import TTS
except ImportError:
    st.error("Error: Coqui TTS library (TTS) not found. Please run 'pip install TTS'")
    st.stop()

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(
    page_title="VITS Voice Cloning & Pause Generator",
    layout="wide"
)

st.title("🗣️ VITS Multi-Speaker TTS (Voice Cloning)")
st.markdown("Using the **Coqui VITS Multi-Speaker** model for fast voice cloning from an audio sample.")

# --- MODEL LOADING (VITS Multi-Speaker) ---

# This specific model is a good multi-speaker VITS model from the Coqui catalog.
MODEL_NAME = "vits--multilingual" 

@st.cache_resource
def load_vits_model():
    """Initializes and caches the multi-speaker VITS model."""
    try:
        with st.spinner(f"Loading VITS Multi-Speaker model ({MODEL_NAME})..."):
            # Load the model directly
            tts = TTS(MODEL_NAME)
        st.success("VITS model loaded successfully! Ready for generation.")
        return tts
    except Exception as e:
        st.error(f"Failed to load VITS model. Error: {e}")
        st.warning("This model might be too large for the Streamlit free tier's disk quota. Consider Hugging Face Spaces.")
        st.stop()
        
tts_model = load_vits_model()


# --- UTILITY FUNCTIONS (Simplified text/pause splitting) ---
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

def insert_pause(duration_ms):
    """Creates a silent pydub AudioSegment."""
    # We will fake this since we removed pydub. We will just use the synthesis loop.
    # We will need to re-introduce pydub OR save to file and read back in loop.
    # For a quick fix, let's simplify and do all synthesis in ONE call.
    # NOTE: VITS doesn't natively handle pause tags as well as Bark/XTTS. 
    pass # We will handle pauses in the main synthesis loop below.


# --- MODIFIED GENERATION FUNCTION (Using VITS) ---
# For VITS, we will use a simpler loop and rely on punctuation for breaks.
def generate_vits_voiceover(tts: TTS, audio_prompt_path: str, text_segments: List[str]) -> bytes:
    """Generates VITS audio for all segments and combines them."""
    
    # We will re-import and use pydub just for the combining, as it is the cleanest way.
    # NOTE: This might bring back the dependency error. If so, you MUST install ffmpeg/libav.
    from pydub import AudioSegment 
    
    combined_audio = AudioSegment.empty()
    SAMPLE_RATE_VITS = 22050 # VITS models often use 22.05 kHz

    text_segments, pauses_ms = process_text_with_tags(text_segments[0]) # Simplified for VITS

    for i, segment in enumerate(text_segments):
        
        # Synthesize audio for the segment
        wav_array = tts.synthesize(
            segment,             
            speaker_wav=audio_prompt_path,  
            language="en"
        )
        
        # Convert NumPy array to pydub AudioSegment
        buffer = io.BytesIO()
        # VITS often returns float32, convert to int16 for WAV
        wav.write(buffer, SAMPLE_RATE_VITS, np.array(wav_array, dtype=np.int16)) 
        buffer.seek(0)
        audio_seg = AudioSegment.from_wav(buffer)
        
        # Add audio segment
        combined_audio += audio_seg
        
        # Add pause AFTER the segment (if it's not the last one)
        if i < len(pauses_ms):
            silence = AudioSegment.silent(duration=pauses_ms[i])
            combined_audio += silence


    # Export combined audio to a temporary file
    output_filename = "vits_voiceover.wav"
    combined_audio.export(output_filename, format="wav")
    
    # Read bytes back for Streamlit
    with open(output_filename, "rb") as f:
        audio_bytes = f.read()
        
    os.remove(output_filename) # Cleanup
    return audio_bytes

# --- STREAMLIT UI AND EXECUTION ---

# Input Columns
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Reference Voice Audio (.wav, .mp3)",
        type=['wav', 'mp3'],
        help="A short, clean audio clip (e.g., 3-5 seconds) of the voice to clone."
    )

with col2:
    text_with_tags = st.text_area(
        "Text to Speak (Use [pause Xs] tags)",
        value="This is a test of the VITS cloning system. [pause 1s] I hope this deploys quickly on Streamlit Cloud!",
        height=150,
        help="Use [pause Xs] where X is seconds (e.g., [pause 2s]). Keep text short for best results."
    )

if st.button("Generate Voiceover", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a Reference Voice Audio file.")
    elif not text_with_tags.strip():
        st.error("Text input is required.")
    else:
        # Save uploaded file to a temporary path for VITS
        temp_audio_path = "temp_vits_prompt.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Generating audio with VITS... This should be very fast!"):
                
                # The VITS function now handles the entire process
                audio_bytes = generate_vits_voiceover(tts_model, temp_audio_path, [text_with_tags])

            # Display the result
            st.subheader("Generated Audio")
            st.audio(audio_bytes, format='audio/wav')
            
            # Download button
            st.download_button(
                label="Download WAV",
                data=audio_bytes,
                file_name="vits_voiceover.wav",
                mime="audio/wav"
            )
            
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            st.warning("If this is an import error, the VITS model may still be too complex for the free Streamlit Cloud environment.")
            
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
