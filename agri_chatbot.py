import sqlite3
import whisper
import pyaudio
import wave
from datetime import datetime
from transformers import pipeline
from geopy.geocoders import Nominatim
import requests
import os
import re

# Initialize Whisper model
try:
    whisper_model = whisper.load_model("tiny")
except Exception as e:
    print(f"Error loading Whisper model: {e}. Ensure ffmpeg is installed and in PATH.")
    exit(1)

# Initialize Hugging Face pipeline
try:
    llm = pipeline("text2text-generation", model="google/flan-t5-base")
    translator = pipeline("translation", model="facebook/m2m100_418M")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    exit(1)

# Initialize SQLite database
def init_db():
    with sqlite3.connect("agri.db") as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query TEXT,
            response TEXT,
            feedback INTEGER,
            timestamp TEXT
        )
        """)
        conn.commit()

# Language detection
def detect_language(text):
    if re.search(r'[\u0D00-\u0D7F]', text):
        return "malayalam"
    return "english"

# Weather API
def get_weather(location):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather data unavailable: API key not set."
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            # Fallback to Kerala if location not found
            url = f"https://api.openweathermap.org/data/2.5/weather?q=Kerala&appid={api_key}"
            response = requests.get(url).json()
            if response.get("cod") != 200:
                return f"Weather data unavailable: {response.get('message', 'Unknown error')}"
        return response.get("weather", [{}])[0].get("description", "N/A")
    except Exception as e:
        return f"Weather data unavailable: {str(e)}"

# Location detection
geolocator = Nominatim(user_agent="agri_chatbot")
def get_location(query):
    try:
        location = geolocator.geocode(query, country_codes="IN")
        return location.address if location else "Kerala"
    except:
        return "Kerala"

# Record audio
def record_audio(filename="temp.wav", duration=5):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        print("Recording... Speak now.")
        frames = []
        for _ in range(0, int(16000 / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        print("Recording stopped.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))
        wf.close()
        return filename
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

# Transcribe audio
def transcribe_audio(audio_file, lang):
    try:
        result = whisper_model.transcribe(audio_file, language="ml" if lang == "malayalam" else "en")
        os.remove(audio_file)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

# Process query
def process_query(query, lang, user_id):
    location = get_location(query)
    weather = get_weather(location)
    season = datetime.now().strftime("%B")  # September 2025
    context = f"Location: {location}, Weather: {weather}, Season: {season}"

    # Translate Malayalam to English
    if lang == "malayalam":
        try:
            query_en = translator(query, src_lang="ml", tgt_lang="en")[0]["generated_text"]
        except:
            query_en = query
    else:
        query_en = query

    # Enhanced prompt
    prompt = (
        f"You are an expert in Kerala's agriculture, knowledgeable about crops, farming techniques, and local practices. "
        f"Provide a concise, practical answer to the query, focusing on actionable farming advice. "
        f"Query: {query_en}. Context: {context}"
    )

    try:
        result = llm(prompt, max_new_tokens=256)
        response = result[0]["generated_text"]
        confidence = 0.8  # Adjusted for flan-t5-base
        if lang == "malayalam":
            try:
                response = translator(response, src_lang="en", tgt_lang="ml")[0]["generated_text"]
            except:
                pass
    except Exception as e:
        response = f"Error generating response: {e}"
        confidence = 0.1

    # Escalation
    if confidence < 0.7 or "unknown" in response.lower() or len(response.strip()) < 10:
        with open("escalated_queries.txt", "a", encoding="utf-8") as f:
            f.write(f"Query: {query}\nContext: {context}\nSuggested Response: {response}\n\n")
        escalation_msg = (
            "നിന്റെ ചോദ്യം സങ്കീർണ്ണമാണ്, കാർഷിക ഓഫീസർക്ക് കൈമാറിയിരിക്കുന്നു."
            if lang == "malayalam"
            else "Your query is complex and has been escalated to an agricultural officer."
        )
        response += f"\n\n{escalation_msg}"

    return response, confidence

# Log query
def log_query(user_id, query, response, confidence):
    with sqlite3.connect("agri.db") as conn:
        conn.execute(
            "INSERT INTO query_logs (user_id, query, response, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, response, None, datetime.now().isoformat())
        )
        conn.commit()

# Main function
def main():
    init_db()
    user_id = "farmer_001"
    print("Agriculture Chatbot (കാർഷിക ചാറ്റ്ബോട്ട്) - September 20, 2025")
    print("Ask questions via text or voice (type 'voice' to record).")
    print("Type 'exit' to quit.")
    print("For voice input, speak clearly for 5 seconds.")

    while True:
        print("\nYou: ", end="")
        user_input = input().strip()

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        query = user_input
        lang = detect_language(query)

        if user_input.lower() == "voice":
            audio_file = record_audio()
            if audio_file:
                query = transcribe_audio(audio_file, lang)
                if not query:
                    print("Bot: Could not transcribe audio. Please try again.")
                    continue
            else:
                print("Bot: Audio recording failed. Please try again.")
                continue

        response, confidence = process_query(query, lang, user_id)
        log_query(user_id, query, response, confidence)

        print(f"Bot: {response}")

        print("Was this helpful? (1 for yes, 0 for no, or skip): ", end="")
        feedback = input().strip()
        if feedback in ["1", "0"]:
            with sqlite3.connect("agri.db") as conn:
                conn.execute(
                    "UPDATE query_logs SET feedback = ? WHERE id = (SELECT MAX(id) FROM query_logs)",
                    (int(feedback),)
                )
                conn.commit()

if __name__ == "__main__":
    main()