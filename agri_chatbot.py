import os
import tempfile
import requests
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from google import genai
from dotenv import load_dotenv
import time

# ---------------------------
# Load API keys and initialize clients
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# ---------------------------
# Audio Recording Settings
# ---------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds per recording

# ---------------------------
# Helper Functions
# ---------------------------
def record_audio(duration=DURATION):
    """Record audio from microphone and visualize waveform."""
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    
    # Create temporary file and ensure it's closed after writing
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')
    tmp_file.close()
    
    # Verify file exists
    if not os.path.exists(tmp_file.name):
        print(f"Error: Audio file {tmp_file.name} was not created.")
        return None
    
    # Check audio level
    audio_data, _ = sf.read(tmp_file.name)
    max_amplitude = np.max(np.abs(audio_data))
    print(f"Audio level: Max amplitude = {max_amplitude:.4f}")
    if max_amplitude < 0.01:
        print("Warning: Audio is too quiet. Speak louder next time.")
    
    # Play back to confirm audio
    try:
        print(f"Playing back recorded audio: {tmp_file.name}")
        playsound(tmp_file.name.replace('\\', '/'))
    except Exception as e:
        print(f"Error playing audio: {e}")
    
    # Visualize waveform
    try:
        plt.figure(figsize=(10, 4))
        time_axis = np.linspace(0, duration, len(audio_data))
        plt.plot(time_axis, audio_data[:, 0] if audio_data.ndim > 1 else audio_data, color='blue')
        plt.title("Recorded Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    except Exception as e:
        print(f"Error visualizing waveform: {e}")
    
    return tmp_file.name

def transcribe_audio(audio_file):
    """Transcribe audio using Deepgram Speech-to-Text with retry mechanism."""
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} does not exist.")
        return ""

    max_retries = 3
    retry_delay = 5  # Increased delay to handle timeouts
    for attempt in range(max_retries):
        try:
            print(f"Attempting to transcribe {audio_file}, size: {os.path.getsize(audio_file)} bytes")
            with open(audio_file, "rb") as audio:
                source = {"buffer": audio}
                options = PrerecordedOptions(
                    model="nova-2",
                    language="en-IN",  # English (Indian accent)
                    smart_format=True
                )
                response = deepgram.listen.rest.v("1").transcribe_file(source, options)
                print("DEBUG Deepgram STT full response:", response)
                transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                if not transcript:
                    print("WARNING: Empty transcript from Deepgram Speech-to-Text.")
                return transcript.strip()
        except Exception as e:
            print(f"Deepgram Speech-to-Text error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Transcription failed.")
                return ""

def generate_text(prompt):
    """Generate text response from Gemini."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def generate_speech(text, lang='en'):
    """Generate speech using Deepgram Text-to-Speech and play it."""
    try:
        options = SpeakOptions(
            model="aura-asteria-en",  # English TTS model
            encoding="mp3",
            sample_rate=44100
        )
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response = deepgram.speak.v("1").stream(
            {"text": text},
            options
        )
        with open(tmp_file.name, "wb") as f:
            f.write(response.stream.getvalue())
        
        playsound(tmp_file.name.replace('\\', '/'))
        os.unlink(tmp_file.name)
    except Exception as e:
        print(f"Deepgram Text-to-Speech error: {e}")

# ---------------------------
# Main Chatbot Loop
# ---------------------------
def chatbot():
    print("🤖 Welcome to Kerala Agri Chatbot!")
    print("Note: Voice mode currently supports English only due to Deepgram limitations.")
    while True:
        mode = input("Choose input mode: [1] Voice, [2] Text, [q] Quit: ").strip().lower()
        if mode == 'q':
            print("Goodbye!")
            break
        elif mode == '1':
            audio_file = record_audio()
            if audio_file is None:
                print("Failed to record audio. Try again.")
                continue
            user_input = transcribe_audio(audio_file)
            print(f"🗣 You said: {user_input}")
            try:
                os.unlink(audio_file)
            except:
                pass
        elif mode == '2':
            user_input = input("You: ").strip()
        else:
            print("Invalid option.")
            continue

        if not user_input:
            print("No input detected. For voice, ensure you're speaking clearly in English.")
            continue

        print("🔎 Asking Gemini...")
        try:
            bot_response = generate_text(user_input)
            print("🤖 Bot:", bot_response)
            lang = 'en'  # Deepgram TTS supports English; Malayalam not supported
            print("🔊 Speaking...")
            generate_speech(bot_response, lang=lang)
        except Exception as e:
            print(f"Error: {e}")

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    print("DEEPGRAM_API_KEY:", "Set" if os.getenv("DEEPGRAM_API_KEY") else "Not set")
    print("GEMINI_API_KEY:", "Set" if os.getenv("GEMINI_API_KEY") else "Not set")
    chatbot()