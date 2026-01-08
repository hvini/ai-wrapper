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
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backend")

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
    },
    {
        "id": "qwen-2.5-coder-7b-instruct",
        "name": "Qwen 2.5 Coder 7B (Best for Code)",
        "repo_id": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    },
    {
        "id": "phi-3.5-mini-instruct",
        "name": "Phi 3.5 Mini 3.8B (Fastest)",
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    }
]

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
            try:
                mics = sc.all_microphones(include_loopback=True)
                for i, mic in enumerate(mics):
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
            mic = sc.get_microphone(self.device_id, include_loopback=True)
            
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                while self.running:
                    data = recorder.record(numframes=BLOCK_SIZE)
                    data = data.flatten()
                    self.audio_queue.put(data)
                    
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.running = False

    def start(self):
        if self.running:
            return True
            
        self.running = True

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
            if np.max(np.abs(audio_data)) < 0.002:
                return ""

            segments, info = self.model.transcribe(
                audio_data,
                beam_size=1,
                best_of=1,
                vad_filter=False,
                language="pt",
                temperature=0.0,
                condition_on_previous_text=False
            )
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

import subprocess
import atexit

LLAMA_SERVER_BIN = os.path.join(os.path.dirname(__file__), "llama.cpp/build/bin/llama-server")

class LlamaService:
    def __init__(self):
        self.context = ""
        self.server_process = None
        self.current_model_path = None
        self.port = 8081
        atexit.register(self.stop_server)
        self.auto_load_model()

    def stop_server(self):
        if self.server_process:
            logger.info("Stopping Llama Server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

    def auto_load_model(self):
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
            
        if not os.path.exists(LLAMA_SERVER_BIN):
            logger.error(f"Llama Server binary not found at: {LLAMA_SERVER_BIN}")
            return False

        self.stop_server()

        try:
            logger.info(f"Starting Llama Server with model: {filename}")
            # Start process
            # -m model
            # -c context size
            # --port port
            # --n-gpu-layers -1 (offload all)
            cmd = [
                LLAMA_SERVER_BIN,
                "-m", path,
                "-c", "2048",
                "--port", str(self.port),
                "-ngl", "999" # Try to offload all layers
            ]
            
            # Use separate process group or detach if needed, but simple Popen is usually fine for parent-child
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, # Redirect logs to avoid clutter or handle them?
                stderr=subprocess.PIPE     # Maybe capture stderr for errors
            )
            
            # Wait for server to be ready
            self.current_model_path = filename
            return self._wait_for_ready()

        except Exception as e:
            logger.error(f"Failed to start Llama Server: {e}")
            return False

    def _wait_for_ready(self):
        logger.info("Waiting for Llama Server to be ready...")
        url = f"http://127.0.0.1:{self.port}/health" # or just check any endpoint
        # llama-server often has /health
        
        for _ in range(20): # Wait up to 20s
            try:
                # Try simple GET
                requests.get(f"http://127.0.0.1:{self.port}/index.html", timeout=1)
                logger.info("Llama Server is ready!")
                return True
            except:
                time.sleep(1)
                if self.server_process and self.server_process.poll() is not None:
                    # Process died
                    _, err = self.server_process.communicate()
                    logger.error(f"Llama Server exited unexpectedly. Stderr: {err}")
                    return False
        
        logger.error("Timed out waiting for Llama Server")
        self.stop_server()
        return False

    def set_context(self, context):
        self.context = context

    def process(self, text):
        if not self.server_process:
            return "Erro: Nenhum modelo de IA carregado (Server not running)."
            
        if not text or len(text) < 5: 
            return None
        
        context_str = f"Contexto: {self.context}\n" if self.context else ""
        system_prompt = "Você é um assistente direto e objetivo. Vá direto ao ponto. Responda sempre em Português."
        user_message = f"{context_str}Analise a seguinte transcrição de áudio brevemente: \"{text}\""
        
        try:
            # Use OpenAI compatible endpoint
            url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 512,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                res_json = response.json()
                return res_json['choices'][0]['message']['content']
            else:
                return f"Error from Llama Server: {response.text}"

        except Exception as e:
            logger.error(f"Llama inference failed: {e}")
            return f"Erro na inferência: {str(e)}"

recorder = AudioRecorder()
transcriber = None
llama_service = LlamaService()

async def ws_handler(websocket):
    logger.info("Client connected")
    global transcriber
    session_transcript = []
    if transcriber is None:
        transcriber = TranscriptionService()

    
    # Event to signal when audio processing is fully caught up (queue empty + buffer processed)
    processing_done_event = asyncio.Event()
    processing_done_event.set() # Initially true as we are not recording

    async def process_audio_loop():
        buffer = np.array([], dtype=np.float32)
        
        # VAD Parameters
        SILENCE_THRESHOLD = 0.005 
        REQUIRED_SILENCE_S = 0.4
        MAX_PHRASE_S = 15.0
        MIN_PHRASE_S = 0.5 
        
        # State
        STATE_IDLE = "IDLE"
        STATE_RECORDING = "RECORDING"
        state = STATE_IDLE
        
        consecutive_silent_samples = 0
        
        while True:
            # We check queue first regardless of running state to drain it
            chunk = recorder.get_audio_chunk()
            
            if chunk is not None:
                # We have data, so we are definitely not "done" processing
                processing_done_event.clear()
                
                # Check for silence in THIS chunk
                chunk_max = np.max(np.abs(chunk))
                is_silent = chunk_max < SILENCE_THRESHOLD
                
                if state == STATE_IDLE:
                    if not is_silent:
                        state = STATE_RECORDING
                        buffer = chunk.copy()
                        consecutive_silent_samples = 0
                        logger.info("Speech detected, starting capture...")
                
                elif state == STATE_RECORDING:
                    buffer = np.concatenate((buffer, chunk))
                    
                    if is_silent:
                        consecutive_silent_samples += len(chunk)
                    else:
                        consecutive_silent_samples = 0
                    
                    total_duration = len(buffer) / SAMPLE_RATE
                    silence_duration = consecutive_silent_samples / SAMPLE_RATE
                    
                    should_transcribe = False
                    reason = ""
                    
                    if silence_duration >= REQUIRED_SILENCE_S:
                        speech_duration = total_duration - silence_duration
                        if speech_duration > MIN_PHRASE_S:
                            should_transcribe = True
                            reason = "end_of_speech"
                        else:
                            # Too short, discard
                            logger.info(f"Discarding short noise ({speech_duration:.2f}s)")
                            state = STATE_IDLE
                            buffer = np.array([], dtype=np.float32)
                            
                    elif total_duration >= MAX_PHRASE_S:
                        should_transcribe = True
                        reason = "max_duration"
                    
                    if should_transcribe:
                        logger.info(f"Transcribing ({reason})... Duration: {total_duration:.2f}s")
                        
                        current_buffer = buffer.copy()
                        buffer = np.array([], dtype=np.float32)
                        state = STATE_IDLE
                        consecutive_silent_samples = 0
                        
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
                # Queue is empty
                if not recorder.running:
                    # Recorder stopped AND queue is empty
                    
                    # If we have leftover buffer that is significant, flush it
                    if len(buffer) > 0:
                        total_duration = len(buffer) / SAMPLE_RATE
                        if total_duration > MIN_PHRASE_S:
                            logger.info(f"Flushing remaining audio... ({total_duration:.2f}s)")
                            current_buffer = buffer.copy()
                            buffer = np.array([], dtype=np.float32)
                            
                            loop = asyncio.get_running_loop()
                            text = await loop.run_in_executor(None, transcriber.transcribe, current_buffer)
                            if text:
                                logger.info(f"Transcribed (Flush): {text}")
                                await websocket.send(json.dumps({"type": "transcription", "text": text}))
                                session_transcript.append(text)
                        else:
                            # Discard tiny remainder
                            buffer = np.array([], dtype=np.float32)
                    
                    state = STATE_IDLE
                    consecutive_silent_samples = 0
                    
                    # Signal we are done
                    if not processing_done_event.is_set():
                        processing_done_event.set()
                        
                    await asyncio.sleep(0.1)
                else:
                    # Running but no data yet
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
                processing_done_event.clear() # Mark as busy
                device_idx = data.get("device_index")
                logger.info(f"Request start record with device_index: {device_idx} (Type: {type(device_idx)})")
                recorder.set_device(device_idx)
                if recorder.start():
                    await websocket.send(json.dumps({"type": "status", "message": "Recording started"}))
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "Failed to start"}))
                    
            elif msg_type == "stop_record":
                recorder.stop()
                await websocket.send(json.dumps({"type": "status", "message": "Stopping..."}))
                
                # Wait for processing to finish
                logger.info("Waiting for transcription to flush...")
                try:
                    await asyncio.wait_for(processing_done_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for transcription flush")

                await websocket.send(json.dumps({"type": "status", "message": "Recording stopped"}))

                full_text = " ".join(session_transcript).strip()
                if full_text:
                    logger.info("Analyzing full transcript...")
                    await websocket.send(json.dumps({"type": "status", "message": "Generating AI Insights..."}))
                    loop = asyncio.get_running_loop()
                    analysis = await loop.run_in_executor(None, llama_service.process, full_text)
                    if analysis:
                        await websocket.send(json.dumps({
                            "type": "ollama_response",
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
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
