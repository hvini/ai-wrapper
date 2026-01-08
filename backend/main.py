import asyncio
import json
import logging
import queue
import threading
import time
import numpy as np
import websockets
from faster_whisper import WhisperModel
import requests
import sys
import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backend")

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

WHISPER_MODEL_SIZE = "small"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096

AVAILABLE_MODELS = [
    {
        "id": "llama-3-8b-instruct",
        "name": "Llama 3 8B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    }
]

# Try to import soundcard, fallback to sounddevice (though sounddevice failed)
USE_SOUNDCARD = False
try:
    import soundcard as sc
    USE_SOUNDCARD = True
    logger.info("Using SoundCard for audio")
except ImportError:
    try:
        import sounddevice as sd
        logger.info("Using SoundDevice for audio")
    except ImportError:
        logger.error("No audio library found!")
        sys.exit(1)
    except OSError:
        logger.error("SoundDevice failed (PortAudio missing?). Install 'soundcard' or install 'libportaudio2'.")
        # Ensure we don't crash loop
        pass

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = False
        self.device_id = None
        self.record_thread = None

    def list_devices(self):
        devices = []
        if USE_SOUNDCARD:
             # soundcard
            try:
                mics = sc.all_microphones(include_loopback=True)
                for i, mic in enumerate(mics):
                    # Check loopback by name convention because attribute might be missing
                    is_loopback = "monitor" in mic.name.lower() or "loopback" in mic.name.lower()
                    devices.append({
                        "index": mic.id, 
                        "name": mic.name,
                        "is_loopback": is_loopback,
                        "channels": mic.channels
                    })
            except Exception as e:
                logger.error(f"Error listing devices: {e}")
        else:
            pass
        return devices

    def set_device(self, index):
        self.device_id = index

    def _record_loop_sc(self):
        logger.info(f"Starting recording loop with device {self.device_id} (Type: {type(self.device_id)})")
        if self.device_id is None:
            logger.error("Device ID is None, cannot record")
            self.running = False
            return

        try:
            # Find the mic object
            mic = sc.get_microphone(self.device_id, include_loopback=True)
            
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                while self.running:
                    # Record a block
                    data = recorder.record(numframes=BLOCK_SIZE)
                    # data is (frames, channels) float32
                    # flatten to 1D
                    data = data.flatten()
                    self.audio_queue.put(data)
                    
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.running = False

    def start(self):
        if self.running:
            return True
            
        self.running = True
        
        # Clear queue
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        if USE_SOUNDCARD:
            self.record_thread = threading.Thread(target=self._record_loop_sc)
            self.record_thread.start()
            return True
        return False

    def stop(self):
        self.running = False
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
            self.record_thread = None
        logger.info("Stopped recording")

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

class TranscriptionService:
    def __init__(self):
        self.current_model_size = WHISPER_MODEL_SIZE
        self.load_model(self.current_model_size)

    def load_model(self, size):
        logger.info(f"Loading Whisper model ({size})...")
        try:
            # Try GPU first
            self.model = WhisperModel(size, device="cuda", compute_type="float16")
            logger.info(f"Whisper model ({size}) loaded on GPU (CUDA)")
            self.current_model_size = size
            return True
        except Exception as e:
            logger.warning(f"Failed to load on GPU ({e}). Falling back to CPU...")
            try:
                self.model = WhisperModel(size, device="cpu", compute_type="int8")
                logger.info(f"Whisper model ({size}) loaded on CPU")
                self.current_model_size = size
                return True
            except Exception as e2:
                logger.error(f"Failed to load model on CPU: {e2}")
                return False

    def transcribe(self, audio_data):
        try:
            # VAD filter provided by faster-whisper
            # We can also check if signal is silence manually to save compute
            if np.max(np.abs(audio_data)) < 0.002:
                return ""

            segments, info = self.model.transcribe(
                audio_data, 
                beam_size=1,            # Faster (Greedy)
                best_of=1,              # Faster
                vad_filter=True, 
                vad_parameters=dict(min_silence_duration_ms=500), # More aggressive VAD
                language="pt",
                temperature=0.0,
                condition_on_previous_text=False # vital for short independent chunks
            )
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class LlamaService:
    def __init__(self):
        self.context = ""
        self.llm = None
        self.current_model_path = None
        # Try to auto-load first available model
        self.auto_load_model()

    def auto_load_model(self):
        # Find first gguf in models dir
        try:
            files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
            if files:
                self.load_model(files[0])
        except Exception as e:
            logger.error(f"Auto-load failed: {e}")

    def load_model(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
            
        try:
            logger.info(f"Loading Llama model: {path}")
            # n_gpu_layers=-1 attempts to offload all layers to GPU
            self.llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=2048, verbose=True)
            self.current_model_path = filename
            logger.info(f"Llama model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            return False

    def set_context(self, context):
        self.context = context

    def process(self, text):
        if not self.llm:
            return "Erro: Nenhum modelo de IA carregado. Por favor baixe/selecione um modelo."
            
        if not text or len(text) < 5: 
            return None
        
        # Enhanced prompt with context
        context_str = f"Contexto: {self.context}\n" if self.context else ""
        system_prompt = "Você é um assistente útil e conciso. Responda sempre em Português."
        user_message = f"{context_str}Analise a seguinte transcrição de áudio brevemente: \"{text}\""
        
        try:
            output = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=512,
                temperature=0.7
            )
            
            return output['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Llama inference failed: {e}")
            return f"Erro na inferência: {str(e)}"

# Global instances
recorder = AudioRecorder()
transcriber = None
llama_service = LlamaService()

async def ws_handler(websocket):
    logger.info("Client connected")
    global transcriber
    session_transcript = []
    if transcriber is None:
        transcriber = TranscriptionService()

    async def process_audio_loop():
        buffer = np.array([], dtype=np.float32)
        PROCESS_INTERVAL = 3.0 
        
        while True:
            if not recorder.running:
                await asyncio.sleep(0.1)
                buffer = np.array([], dtype=np.float32)
                continue
                
            chunk = recorder.get_audio_chunk()
            if chunk is not None:
                if len(buffer) == 0:
                     logger.info("Receiving audio data...")
                buffer = np.concatenate((buffer, chunk))
                
                if len(buffer) >= SAMPLE_RATE * PROCESS_INTERVAL:
                    logger.info(f"Buffer full ({len(buffer)} samples), transcribing...")
                    current_buffer = buffer.copy()
                    
                    # Implement simple overlap to catch words at boundary
                    # Keep last 0.5s of audio for context in next chunk
                    overlap_samples = int(SAMPLE_RATE * 0.5)
                    if len(buffer) > overlap_samples:
                         buffer = buffer[-overlap_samples:]
                    else:
                         buffer = np.array([], dtype=np.float32)
                    
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(None, transcriber.transcribe, current_buffer)
                    
                    if text:
                        logger.info(f"Transcribed: {text}")
                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": text
                        }))
                        session_transcript.append(text)
                    else:
                        logger.info("Transcription yielded empty text (silence?)")
            else:
                await asyncio.sleep(0.01)

    process_task = asyncio.create_task(process_audio_loop())

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "get_devices":
                devices = recorder.list_devices()
                logger.info(f"Sending {len(devices)} devices to client")
                await websocket.send(json.dumps({
                    "type": "devices",
                    "data": devices
                }))
                
            elif msg_type == "start_record":
                session_transcript.clear()
                device_idx = data.get("device_index")
                logger.info(f"Request start record with device_index: {device_idx} (Type: {type(device_idx)})")
                recorder.set_device(device_idx)
                if recorder.start():
                    await websocket.send(json.dumps({"type": "status", "message": "Recording started"}))
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "Failed to start"}))
                    
            elif msg_type == "stop_record":
                recorder.stop()
                await websocket.send(json.dumps({"type": "status", "message": "Recording stopped"}))

                full_text = " ".join(session_transcript).strip()
                if full_text:
                    logger.info("Analyzing full transcript...")
                    await websocket.send(json.dumps({"type": "status", "message": "Generating AI Insights..."}))
                    loop = asyncio.get_running_loop()
                    analysis = await loop.run_in_executor(None, llama_service.process, full_text)
                    if analysis:
                        await websocket.send(json.dumps({
                            "type": "ollama_response", # Keeping type name for compatibility for now
                            "text": analysis
                        }))
            
            elif msg_type == "update_context":
                ctx = data.get("context", "")
                llama_service.set_context(ctx)
                logger.info(f"Updated context: {ctx}")

            elif msg_type == "update_config":
                new_model = data.get("model")
                if new_model and new_model != transcriber.current_model_size:
                    logger.info(f"Requests model change to: {new_model}")
                    await websocket.send(json.dumps({"type": "status", "message": f"Reloading model to {new_model}..."}))
                    
                    loop = asyncio.get_running_loop()
                    success = await loop.run_in_executor(None, transcriber.load_model, new_model)
                    
                    if success:
                        await websocket.send(json.dumps({"type": "status", "message": f"Model changed to {new_model}"}))
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "Failed to load model"}))
            
            elif msg_type == "get_llm_models":
                # List local models and available to download models
                local_models = []
                if os.path.exists(MODELS_DIR):
                    local_models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
                
                await websocket.send(json.dumps({
                    "type": "llm_models",
                    "local": local_models,
                    "available": AVAILABLE_MODELS,
                    "current": llama_service.current_model_path
                }))

            elif msg_type == "load_llm_model":
                filename = data.get("filename")
                logger.info(f"Request to load LLM model: {filename}")
                loop = asyncio.get_running_loop()
                success = await loop.run_in_executor(None, llama_service.load_model, filename)
                if success:
                     await websocket.send(json.dumps({"type": "status", "message": f"Loaded LLM: {filename}"}))
                     # Refresh list to update 'current'
                     await websocket.send(json.dumps({
                        "type": "llm_models",
                        "local": [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")],
                        "available": AVAILABLE_MODELS,
                        "current": llama_service.current_model_path
                    }))
                else:
                     await websocket.send(json.dumps({"type": "error", "message": f"Failed to load LLM: {filename}"}))

            elif msg_type == "download_model":
                model_id = data.get("model_id")
                # Find model info
                model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
                if model_info:
                    await websocket.send(json.dumps({"type": "status", "message": f"Downloading {model_info['name']}... This may take a while."}))
                    
                    def download_task():
                        try:
                            hf_hub_download(
                                repo_id=model_info["repo_id"],
                                filename=model_info["filename"],
                                local_dir=MODELS_DIR,
                                local_dir_use_symlinks=False
                            )
                            return True
                        except Exception as e:
                            logger.error(f"Download failed: {e}")
                            return False

                    loop = asyncio.get_running_loop()
                    success = await loop.run_in_executor(None, download_task)
                    
                    if success:
                        await websocket.send(json.dumps({"type": "status", "message": f"Downloaded {model_info['name']}"}))
                        # Refresh list
                        await websocket.send(json.dumps({
                            "type": "llm_models",
                            "local": [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")],
                            "available": AVAILABLE_MODELS,
                            "current": llama_service.current_model_path
                        }))
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": f"Failed to download {model_info['name']}"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS Error: {e}")
    finally:
        recorder.stop()
        process_task.cancel()

async def main():
    async with websockets.serve(ws_handler, "localhost", 8765):
        logger.info("WebSocket server started on port 8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        # On linux, we often use 'spawn' or 'fork', but default fork is fine for basics
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
