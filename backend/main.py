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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backend")

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
WHISPER_MODEL_SIZE = "large"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096

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

class OllamaService:
    def __init__(self):
        self.context = ""

    def set_context(self, context):
        self.context = context

    def process(self, text):
        if not text or len(text) < 5: 
            return None
        
        # Enhanced prompt with context
        context_str = f"Contexto: {self.context}\n" if self.context else ""
        prompt = f"{context_str}Analise a seguinte transcrição de áudio brevemente (responda em Português): \"{text}\""
        
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": "llama3", 
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            
            if response.status_code == 200:
                res_json = response.json()
                return res_json.get('response', '')
            elif response.status_code == 404:
                return "Ollama error: Model 'llama3' not found. Please `ollama pull llama3` or change code."
            else:
                return f"Ollama error: {response.text}"
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return None

# Global instances
recorder = AudioRecorder()
transcriber = None
ollama = OllamaService()

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
                    analysis = await loop.run_in_executor(None, ollama.process, full_text)
                    if analysis:
                        await websocket.send(json.dumps({
                            "type": "ollama_response",
                            "text": analysis
                        }))
            
            elif msg_type == "update_context":
                ctx = data.get("context", "")
                ollama.set_context(ctx)
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
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
