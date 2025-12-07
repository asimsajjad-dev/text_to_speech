import streamlit as st
import asyncio
# import io
import os
#from pydub import AudioSegment

# --- OpenAI and Asynchronous Imports ---
from openai import AsyncOpenAI, OpenAI
from openai.types.audio import AudioSpeechResponse
# We will use a synchronous client for simplicity in the non-streaming mode
# Async is tricky to run directly inside Streamlit's main thread.

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(
    page_title="OpenAI Advanced TTS with Instructions",
    layout="wide"
)

st.title("🗣️ OpenAI Advanced TTS with Custom Instructions")
st.markdown("Use the `instructions` parameter to guide the voice style and delivery.")

# --- INITIALIZATION AND CACHING ---

# 1. API Key Setup
# The client needs the API key. We get it from environment or user input.
api_key = st.sidebar.text_input(
    "OpenAI API Key (Required)", 
    type="password",
    value=os.environ.get("OPENAI_API_KEY", "")
)

# 2. Synchronous Client Initialization (Easier for Streamlit)
@st.cache_resource
def get_openai_client(key):
    """Initializes and caches the synchronous OpenAI client."""
    if not key:
        return None
    try:
        # Use the synchronous client for a simpler integration
        return OpenAI(api_key=key) 
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

client = get_openai_client(api_key)

# --- CORE TTS FUNCTION (Synchronous) ---

def generate_tts_audio(tts_client: OpenAI, text_input: str, instructions_input: str) -> bytes:
    """Calls the synchronous OpenAI TTS API and returns raw audio bytes."""
    
    # We will use the common MP3 format for simple delivery via Streamlit
    # The dedicated 'instructions' parameter works with the synchronous client too.
    
    response = tts_client.audio.speech.create(
        model="gpt-4o-mini-tts",  # Use the powerful model
        voice="ash",              # The base voice
        input=text_input,
        instructions=instructions_input, # Pass the instructions
        response_format="mp3",    # MP3 is ideal for web/Streamlit
    )
    
    # The response object contains the raw audio content
    return response.content

# --- STREAMLIT UI ---

# Default values for inputs
default_input = """Hello, and welcome back to the channel. I'm so glad you decided to stop by tonight. Go ahead and get yourself comfortable. [PAUSE] Tonight, we're taking a journey back in time to the deep winter of 1922."""
default_instructions = """Voice Style: Calm, composed, reassuring Midwestern American male—quiet authority with sincere empathy, steady moderate pacing (110-130 wpm), clear precise pronunciation, and brief supportive pauses after offers of help."""

# Input Fields
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area(
        "Text to Speak (Use [PAUSE], [BREATHE], etc.)",
        value=default_input,
        height=300
    )

with col2:
    user_instructions = st.text_area(
        "Voice Delivery Instructions",
        value=default_instructions,
        height=300
    )
    
if st.button("Generate Audio", type="primary"):
    if not client:
        st.error("Please enter a valid OpenAI API Key in the sidebar.")
    elif not user_input.strip():
        st.error("Text input cannot be empty.")
    else:
        with st.spinner("Generating and customizing audio..."):
            try:
                # Call the synchronous function
                audio_bytes = generate_tts_audio(client, user_input, user_instructions)

                # Display the result
                st.subheader("Generated Audio")
                st.audio(audio_bytes, format='audio/mp3')

                # Optional: Provide a download button
                st.download_button(
                    label="Download MP3",
                    data=audio_bytes,
                    file_name="custom_openai_tts.mp3",
                    mime="audio/mp3"
                )

            except Exception as e:
                error_message = str(e)
                if "API key is required" in error_message:
                     st.error("Invalid or missing OpenAI API Key. Please check the key in the sidebar.")
                else:
                    st.error(f"An unexpected error occurred: {error_message}")

st.markdown("---")
st.info("💡 **Why synchronous?** Streamlit's core processing runs synchronously. Wrapping the original asynchronous streaming code is complex, while the synchronous call (`response_format='mp3'`) is fast enough for excellent user experience.")
