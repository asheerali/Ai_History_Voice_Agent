
import base64
import logging
import os
import random
import shutil
import time
import uuid
import tempfile
import gradio as gr
from pydub import AudioSegment
from gtts import gTTS
import speech_recognition as sr
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set your DeepSeek API Key (for development only, use environment variables in production)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# --- Unified Language Mapping ---
# This maps user-friendly language names to both gTTS and speech recognition codes
LANGUAGE_MAPPING = {
    'English': {'tts': 'en', 'sr': 'en-US'},
    'Spanish': {'tts': 'es', 'sr': 'es-ES'},
    'French': {'tts': 'fr', 'sr': 'fr-FR'},
    'German': {'tts': 'de', 'sr': 'de-DE'},
    'Italian': {'tts': 'it', 'sr': 'it-IT'},
    'Portuguese': {'tts': 'pt', 'sr': 'pt-PT'},
    'Russian': {'tts': 'ru', 'sr': 'ru-RU'},
    'Chinese (Simplified)': {'tts': 'zh-CN', 'sr': 'zh-CN'},
    'Japanese': {'tts': 'ja', 'sr': 'ja-JP'},
    'Korean': {'tts': 'ko', 'sr': 'ko-KR'},
    'Hindi': {'tts': 'hi', 'sr': 'hi-IN'},
    'Arabic': {'tts': 'ar', 'sr': 'ar-EG'},
    'Dutch': {'tts': 'nl', 'sr': 'nl-NL'},
    'Turkish': {'tts': 'tr', 'sr': 'tr-TR'},
    'Polish': {'tts': 'pl', 'sr': 'pl-PL'},
    'Greek': {'tts': 'el', 'sr': 'el-GR'},
    'Thai': {'tts': 'th', 'sr': 'th-TH'},
    'Vietnamese': {'tts': 'vi', 'sr': 'vi-VN'},
    'Czech': {'tts': 'cs', 'sr': 'cs-CZ'},
    'Hungarian': {'tts': 'hu', 'sr': 'hu-HU'},
    'Romanian': {'tts': 'ro', 'sr': 'ro-RO'},
    'Finnish': {'tts': 'fi', 'sr': 'fi-FI'},
    'Danish': {'tts': 'da', 'sr': 'da-DK'},
    'Ukrainian': {'tts': 'uk', 'sr': 'uk-UA'},
    'Bengali': {'tts': 'bn', 'sr': 'bn-IN'},
    'Tamil': {'tts': 'ta', 'sr': 'ta-IN'},
    'Telugu': {'tts': 'te', 'sr': 'te-IN'},
    'Urdu': {'tts': 'ur', 'sr': 'ur-PK'}
}

def get_language_codes(language_name):
    """Get both TTS and speech recognition language codes from a single language name"""
    logger.debug(f"Getting language codes for: {language_name}")
    if language_name in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[language_name]['tts'], LANGUAGE_MAPPING[language_name]['sr']
    # Default to English if language not found
    logger.warning(f"Language '{language_name}' not found in mapping, using English")
    return 'en', 'en-US'

def convert_to_wav(input_path, output_path="converted.wav"):
    logger.debug(f"Converting audio from {input_path} to WAV")
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        logger.debug(f"Audio converted to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        raise RuntimeError(f"Audio conversion failed: {e}")

def transcribe_audio(audio_path, language):
    logger.debug(f"Transcribing audio: {audio_path} with language: {language}")
    recognizer = sr.Recognizer()
    _, sr_lang_code = get_language_codes(language)
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data, language=sr_lang_code)
        logger.debug(f"Transcription successful: {transcription}")
        return transcription
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Speech recognition error: {e}")
        raise RuntimeError(f"Speech recognition failed: {e}")

def generate_facts(transcribed_text, language):
    logger.debug(f"Generating fact for text: {transcribed_text} in language: {language}")
    if not transcribed_text:
        return "Could not understand the audio. Please try again."
    
    # Check if API key is set
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY is not set")
        return "API key is not configured. Please set the DEEPSEEK_API_KEY environment variable."
    
    prompt = f"""
In "{language}" You are an expert Egyptian Museum tour guide and historian.
A visitor just said: "{transcribed_text}" — Based on this, tell a captivating and informative historical fact
related to the mentioned artifact or topic.
Keep your tone enthusiastic and educational, using storytelling to engage tourists.
The fact should be accurate, based on known Egyptian history, and strictly 200–250 words long.
Avoid markdown, lists, or technical jargon. Just a single, concise, vivid paragraph.
"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful and creative assistant in ancient history."},
            {"role": "user", "content": prompt.strip()}
        ],
        "temperature": random.uniform(0.9, 1.0),
        "max_tokens": 1000
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        fact = response.json()["choices"][0]["message"]["content"].strip()
        logger.debug(f"Fact generated: {fact[:50]}...")
        return fact
    except requests.RequestException as e:
        logger.error(f"DeepSeek API error: {e}")
        return f"Error with DeepSeek API: {str(e)}"

def text_to_speech(message, language):
    """
    Convert text to speech using gTTS with improved error handling
    and temporary file management to avoid 'Too Many Requests' errors.
    
    Args:
        message (str): The text to convert to speech
        language (str): Language for the speech
        
    Returns:
        str: Path to the generated audio file or None if failed
    """
    logger.debug(f"Converting text to speech in language: {language}")
    
    try:
        # Get language code
        tts_lang_code = get_language_codes(language)[0]  # Assuming get_language_codes returns a list/tuple
        
        # Use tempfile to manage file creation
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            filename = temp_file.name
            
        # Create and save the audio with exponential backoff for rate limiting
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds delay
        
        for attempt in range(max_retries):
            try:
                tts = gTTS(text=message, lang=tts_lang_code)
                tts.save(filename)
                logger.debug(f"Audio saved to {filename}")
                return filename
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Too Many Requests error - implement backoff
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Either not a rate limit error or we've exhausted retries
                    raise
                    
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        # Clean up the temp file if it exists
        if 'filename' in locals() and os.path.exists(filename):
            try:
                os.unlink(filename)
            except:
                pass
        raise RuntimeError(f"Text-to-speech failed: {e}")

def museum_tour_pipeline(audio_path, language):
    """Process audio file through the museum tour guide pipeline using the same language for input/output"""
    logger.debug(f"Starting museum tour pipeline for audio: {audio_path} in language: {language}")
    
    temp_files = []
    
    try:
        # Convert to WAV for speech recognition
        wav_path = os.path.join("uploads", f"converted_{uuid.uuid4()}.wav")
        convert_to_wav(audio_path, wav_path)
        temp_files.append(wav_path)
        
        # Transcribe the audio
        user_input = transcribe_audio(wav_path, language)
        if not user_input:
            return None, "Could not transcribe the audio. Please try again with clearer audio.", None
        
        # Generate fact based on transcription
        fact = generate_facts(user_input, language)
        
        # Convert fact to speech
        audio_output_path = text_to_speech(fact, language)
        temp_files.append(audio_output_path)
        
        return user_input, fact, audio_output_path
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return None, f"Error: {str(e)}", None
    
    finally:
        # We don't remove temp files here since Gradio needs them for display
        # They'll be removed when the app restarts or by the OS temp file cleaner
        pass

def process_audio_file(audio, language):
    """Main function to process audio for the Gradio interface"""
    logger.debug(f"Processing audio with language: {language}")
    
    if audio is None:
        return None, "No audio detected. Please upload an audio file or record one.", None
    
    # Save temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_path = temp_audio.name
        
        # If audio is tuple from microphone, save it using scipy
        if isinstance(audio, tuple):
            import scipy.io.wavfile as wav
            sample_rate, audio_data = audio
            wav.write(temp_path, sample_rate, audio_data)
        else:
            # Copy uploaded file to temp location
            shutil.copy(audio, temp_path)
    
    # Process the audio
    transcript, fact, audio_output = museum_tour_pipeline(temp_path, language)
    
    # Clean up the temporary input file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return transcript, fact, audio_output

# Create Gradio interface
with gr.Blocks(title="Egyptian Museum Tour Guide") as demo:
    gr.Markdown("# 🏛️ Egyptian Museum Virtual Tour Guide")
    gr.Markdown("""Upload or record audio asking about an Egyptian artifact or history topic,
                and receive an informative response from our virtual tour guide in the same language!""")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Ask about Egyptian history or artifacts"
            )
            
            language = gr.Dropdown(
                choices=sorted(list(LANGUAGE_MAPPING.keys())),
                value="English",
                label="Select Language"
            )
            
            process_btn = gr.Button("🎬 Generate Tour Guide Response", variant="primary")
        
        with gr.Column():
            transcript_output = gr.Textbox(label="Your Question (Transcribed)")
            fact_output = gr.TextArea(label="Historical Information", lines=8)
            audio_response = gr.Audio(label="Audio Response")
    
    process_btn.click(
        fn=process_audio_file,
        inputs=[audio_input, language],
        outputs=[transcript_output, fact_output, audio_response]
    )
    
    gr.Markdown("### How It Works")
    gr.Markdown("""
    1. **Ask a question**: Upload an audio file or record your voice asking about Egyptian history, artifacts, or sites
    2. **Select your language**: Choose the language you're speaking and the guide will respond in the same language
    3. **Get responses**: Receive both text and audio information about the Egyptian topic you asked about
    
    This virtual tour guide uses speech recognition, DeepSeek AI, and text-to-speech technologies to provide an interactive museum experience.
    """)
    
    gr.Markdown("### Examples")
    gr.Markdown("""
    Try asking questions like:
    - "Tell me about the Rosetta Stone"
    - "What is the significance of King Tutankhamun?"
    - "How were the pyramids built?"
    - "Explain the mummification process"
    """)

if __name__ == "__main__":
    demo.launch()