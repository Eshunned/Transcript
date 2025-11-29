import os
import tempfile
import shutil
import json
import base64  # <-- REQUIRED for decoding Base64 audio

# Removed 'requests' and added the official SDK import
try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None
    print("WARNING: The 'sarvamai' SDK is not installed. The transcription function will fail.")

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from pydantic import BaseModel, Field

# --- Configuration and Setup ---

load_dotenv()

SARVAM_AI_API_KEY = "sk_tj00r52m_RY2tHR3bI4snFi2zfrLYv95Pp"

if not SARVAM_AI_API_KEY:
    raise RuntimeError("FATAL ERROR: SARVAM_AI_API_KEY not found. Please set it in your .env file.")

# Initialize FastAPI App
app = FastAPI(
    title="Sarvam AI Transcription Service",
    description="Endpoint to transcribe audio recordings of patient-doctor chat sessions using Sarvam AI.",
    version="1.0.0"
)

# Initialize Sarvam AI Client globally
SARVAM_CLIENT = None
if SarvamAI is not None:
    try:
        # Client initialization is now protected by the 'SarvamAI is not None' check
        SARVAM_CLIENT = SarvamAI(api_subscription_key=SARVAM_AI_API_KEY)
    except Exception as e:
        # If initialization fails even with the class imported
        SARVAM_CLIENT = None
        print(f"ERROR initializing Sarvam AI Client: {e}. Check if key is correct or API service is reachable.")

DEFAULT_TRANSCRIPTION_MODEL = "saarika:v2.5"
DEFAULT_LANGUAGE_CODE = "gu-IN"


# --- Pydantic Models ---

class TranscriptionResponse(BaseModel):
    """Structured response for transcription output."""
    filename: str = Field(description="The name of the uploaded file.")
    transcribed_text: str = Field(description="The full text transcribed by Sarvam AI.")


class Base64AudioRequest(BaseModel):
    """Structured input for Base64 audio submission."""
    audio_data: str = Field(description="Base64 encoded audio bytes.")
    file_extension: str = Field(default=".wav", description="Original file extension (e.g., .wav, .mp3).")
    filename: str = Field(default="voice_note.wav", description="Name of the file.")


# --- Sarvam AI API Integration Function ---

def _call_sarvam_transcription_api(file_path: str) -> str:
    """
    Sends the local audio file to the Sarvam AI SDK for transcription.
    """
    global SARVAM_CLIENT  # Use global client instance

    if SARVAM_CLIENT is None:
        # This should only happen if the SDK failed to import or initialize globally
        raise Exception("Sarvam AI Client failed to initialize. Check SDK installation and API Key.")

    # 1. Open the file in binary mode for the SDK call
    with open(file_path, 'rb') as audio_file:
        try:
            # 2. Make the transcription call using the structure provided by the user
            response = SARVAM_CLIENT.speech_to_text.transcribe(
                file=audio_file,
                model=DEFAULT_TRANSCRIPTION_MODEL,
                language_code=DEFAULT_LANGUAGE_CODE
            )

            # 3. Extract Transcribed Text
            transcribed_text = response.get('text', response.get('transcript',
                                                                 'Transcription result not found in API response.'))

            if transcribed_text == 'Transcription result not found in API response.':
                print(f"Sarvam AI Response: {response}")

            return transcribed_text

        except Exception as e:
            raise Exception(f"Sarvam AI Transcription failed: {e}")


# --- Health Check Endpoint (New) ---
@app.get("/")
async def root():
    """Returns application status and guides user to documentation."""
    return {
        "status": "Service is running",
        "documentation": "Visit /docs for the interactive API interface."
    }


# --- Endpoint 1: File Upload (Original) ---

@app.post("/transcribe_audio_file", response_model=TranscriptionResponse, status_code=200)
async def transcribe_audio_file(
        file: UploadFile = File(..., description="Audio file of the chat session (.wav, .mp3, etc.)")):
    """
    Accepts a file upload via multipart/form-data.
    """

    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")

    temp_file_path = None
    _, ext = os.path.splitext(file.filename or "audio.wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to save uploaded file temporarily.")
        finally:
            file.file.close()

    try:
        print(f"Sending uploaded file {file.filename} to Sarvam AI for transcription...")
        transcribed_text = _call_sarvam_transcription_api(temp_file_path)

        return TranscriptionResponse(
            filename=file.filename or "audio_file",
            transcribed_text=transcribed_text
        )

    except Exception as e:
        print(f"An unexpected error occurred during transcription: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Transcription service failed: {e}"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# --- Endpoint 2: Microphone Input (New) ---

@app.post("/transcribe_base64_audio", response_model=TranscriptionResponse, status_code=200)
async def transcribe_base64_audio(request: Base64AudioRequest):
    """
    Accepts Base64 encoded audio data (suitable for mic recordings from a browser)
    and sends it for transcription via Sarvam AI.
    """
    temp_file_path = None
    try:
        # 1. Decode Base64 string into audio bytes
        audio_bytes = base64.b64decode(request.audio_data)

        # 2. Write bytes to a temporary file
        temp_file_path = os.path.join(tempfile.gettempdir(), f"mic_recording{request.file_extension}")
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)

        # 3. Call the Sarvam AI transcription service
        print(f"Sending Base64 audio ({request.filename}) to Sarvam AI for transcription...")
        transcribed_text = _call_sarvam_transcription_api(temp_file_path)

        return TranscriptionResponse(
            filename=request.filename,
            transcribed_text=transcribed_text
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio data received.")
    except Exception as e:
        print(f"An unexpected error occurred during Base64 transcription: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Transcription service failed: {e}"
        )
    finally:
        # 4. Cleanup: Delete the temporary local file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)