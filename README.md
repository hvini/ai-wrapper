# Audio Intelligence Wrapper

An Electron + Python application that listens to system audio, transcribes it using Whisper, and leverages a local Ollama LLM for analysis.

## Prerequisites

1. **Python 3.12+**
2. **Node.js 20+**
3. **Ollama** installed and running (`ollama serve`).
   - Make sure you have a model pulled, e.g., `ollama pull llama3`.
4. **PortAudio** (Required for sounddevice audio capture)
   - Ubuntu: `sudo apt-get install libportaudio2`

## Installation

1. Install Node dependencies:
   ```bash
   npm install
   ```

2. Setup Python environment (Already done if you see backend/venv):
   ```bash
   python3 -m venv backend/venv
   ./backend/venv/bin/pip install -r backend/requirements.txt
   ```

## Running the App

Start both the Electron app and the Python backend with one command:

```bash
npm start
```

## System Audio Capture (Linux)

To capture **system audio** (what you hear), you need to select the Monitor device.
1. Launch the app.
2. In the dropdown, look for a device named **"Monitor of..."** or **"Loopback"**.
3. Select it and click "Start Listening".

If you don't see a Monitor device, likely you are using PulseAudio. `sounddevice` should list it.
If not, you might need to use `pavucontrol` to redirect audio or ensure your user has audio permissions.

## Troubleshooting

- **Ollama Input/Output Error**: Ensure Ollama is running (`systemctl status ollama` or just `ollama serve`).
- **Audio Error**: Ensure `libportaudio2` is installed.
