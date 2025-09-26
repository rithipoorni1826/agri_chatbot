import os
import tempfile
import asyncio
import requests
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pygame
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from google import genai
from dotenv import load_dotenv
from datetime import datetime
import time
import logging

# ---------------------------
# Load API keys and initialize clients
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not GEMINI_API_KEY or not DEEPGRAM_API_KEY:
    print("Error: API keys not set. Please check your .env file.")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# Initialize pygame mixer
pygame.mixer.init()

# ---------------------------
# Audio Recording Settings
# ---------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 20  # seconds per recording

# ---------------------------
# Helper Functions
# ---------------------------
def play_audio(file_path):
    """Play audio file using pygame."""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for playback to finish
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio with pygame: {e}")
    finally:
        pygame.mixer.music.unload()

def record_audio(duration=DURATION):
    """Record audio from microphone and visualize waveform."""
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    
    # Create temporary file
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
        play_audio(os.path.normpath(tmp_file.name))
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
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempting to transcribe {audio_file}, size: {os.path.getsize(audio_file)} bytes")
            with open(audio_file, "rb") as audio:
                source = {"buffer": audio}
                options = PrerecordedOptions(
                    model="nova-2",
                    language="en-IN",
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
    # Handle greetings
    if prompt.lower().strip() in ["hello", "hello?", "hi", "hi.", "hiiiii"]:
        return "Hi! I'm your Kerala Agri Chatbot. Ask me about farming, weather, or anything else!"
    
    # Include current date for date-related queries
    if "today" in prompt.lower():
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        prompt = f"{prompt} (Current date is {current_date})"
    
    # Add instruction to keep responses concise
    prompt = f"{prompt} Please keep the response concise and under 2000 characters."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

async def async_generate_speech(text, filename):
    """Async helper to generate speech using Deepgram TTS."""
    try:
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="mp3"
        )
        payload = {"text": text}
        response = await deepgram.speak.asyncrest.v("1").save(filename, payload, options)
        print(f"TTS Response: {response.to_json(indent=4)}")
    except Exception as e:
        print(f"Deepgram TTS async error: {e}")
        raise

def generate_speech(text, lang='en'):
    """Generate speech using Deepgram Text-to-Speech and play it."""
    if lang != 'en':
        print("Warning: Only English TTS supported.")
    
    MAX_CHAR_LIMIT = 2000  # Deepgram's character limit
    
    # Truncate or split text if it exceeds the limit
    if len(text) > MAX_CHAR_LIMIT:
        print(f"Warning: Text exceeds {MAX_CHAR_LIMIT} characters ({len(text)}). Truncating to fit.")
        text = text[:MAX_CHAR_LIMIT]  # Simple truncation
        # Alternatively, for splitting into chunks (uncomment if preferred):
        # chunks = [text[i:i+MAX_CHAR_LIMIT] for i in range(0, len(text), MAX_CHAR_LIMIT)]
        # for i, chunk in enumerate(chunks):
        #     filename = tempfile.mktemp(suffix=f"_{i}.mp3")
        #     try:
        #         asyncio.run(async_generate_speech(chunk, filename))
        #         if os.path.exists(filename):
        #             play_audio(os.path.normpath(filename))
        #         else:
        #             print(f"Error: Audio file {filename} was not generated.")
        #     except Exception as e:
        #         print(f"Deepgram Text-to-Speech error for chunk {i}: {e}")
        #     finally:
        #         if os.path.exists(filename):
        #             try:
        #                 os.unlink(filename)
        #             except Exception as e:
        #                 print(f"Failed to delete temporary file {filename}: {e}")
        # return  # Exit after processing chunks

    filename = tempfile.mktemp(suffix=".mp3")
    try:
        asyncio.run(async_generate_speech(text, filename))
        if os.path.exists(filename):
            play_audio(os.path.normpath(filename))
        else:
            print("Error: Audio file was not generated.")
    except Exception as e:
        print(f"Deepgram Text-to-Speech error: {e}")
    finally:
        if os.path.exists(filename):
            try:
                os.unlink(filename)
            except Exception as e:
                print(f"Failed to delete temporary file {filename}: {e}")

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
            except Exception as e:
                print(f"Failed to delete temporary file {audio_file}: {e}")
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
            print(f"Response length: {len(bot_response)} characters")
            lang = 'en'
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