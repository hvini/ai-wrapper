import { useState, useEffect, useRef } from 'react'
import './index.css'

function App() {
  const [status, setStatus] = useState("Disconnected");
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [deviceStatus, setDeviceStatus] = useState("Detecting audio...");
  const [context, setContext] = useState("");

  const ws = useRef(null);
  const transcriptionRef = useRef(null);
  const analysisRef = useRef(null);

  useEffect(() => {
    connectWs();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  // Send context update when it changes (debounced)
  useEffect(() => {
    const handler = setTimeout(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: "update_context", context: context }));
      }
    }, 500);
    return () => clearTimeout(handler);
  }, [context, status]);

  // Auto-scroll
  useEffect(() => {
    if (transcriptionRef.current) {
      transcriptionRef.current.scrollTop = transcriptionRef.current.scrollHeight;
    }
  }, [transcripts]);

  useEffect(() => {
    if (analysisRef.current) {
      analysisRef.current.scrollTop = analysisRef.current.scrollHeight;
    }
  }, [analyses]);


  const connectWs = () => {
    setStatus("Connecting...");
    ws.current = new WebSocket("ws://localhost:8765");

    ws.current.onopen = () => {
      setStatus("Connected");
      ws.current.send(JSON.stringify({ type: "get_devices" }));
      if (context) {
        ws.current.send(JSON.stringify({ type: "update_context", context: context }));
      }
    };

    ws.current.onclose = () => {
      setStatus("Disconnected");
      setIsRecording(false);
      setTimeout(connectWs, 2000);
    };

    ws.current.onerror = (err) => {
      console.error("WS Error:", err);
      setStatus("Error");
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "devices":
          if (data.data.length > 0) {
            const monitor = data.data.find(d => {
              const name = d.name.toLowerCase();
              return name.includes('monitor') || name.includes('loopback') || d.is_loopback;
            });

            if (monitor) {
              setSelectedDevice(monitor.index);
              setDeviceStatus(`Ready: ${monitor.name.substring(0, 30)}...`);
            } else {
              setSelectedDevice(data.data[0].index);
              setDeviceStatus(`Ready (Default): ${data.data[0].name.substring(0, 20)}...`);
            }
          } else {
            setDeviceStatus("No Audio Devices Found");
          }
          break;
        case "transcription":
          setTranscripts(prev => [...prev, data.text]);
          break;
        case "ollama_response":
          setAnalyses(prev => [...prev, data.text]);
          break;
        case "status":
          console.log("Status:", data.message);
          break;
        case "error":
          alert("Error: " + data.message);
          setIsRecording(false);
          break;
        default:
          break;
      }
    };
  };

  const toggleRecording = () => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;

    if (isRecording) {
      ws.current.send(JSON.stringify({ type: "stop_record" }));
      setIsRecording(false);
    } else {
      if (selectedDevice === null) {
        alert("Audio device not yet detected. Please wait.");
        return;
      }
      ws.current.send(JSON.stringify({
        type: "start_record",
        device_index: selectedDevice
      }));
      setTranscripts([]);
      setAnalyses([]);
      setIsRecording(true);
    }
  };

  return (
    <div className="glass-container">
      <div className="header">
        <h1>AI Wrapper</h1>
        <div className={`status-check ${status.toLowerCase()}`}>
          {status === "Connected" ? deviceStatus : status}
        </div>
      </div>

      <div className="main-content">
        <textarea
          className="context-input"
          placeholder="System Context (e.g. 'Summarize key points regarding Biology')"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          rows="2"
        />

        <div className="transcription-area">
          <h2>Live Transcription</h2>
          <div className="scroll-box" ref={transcriptionRef}>
            {transcripts.length === 0 ? <p style={{ opacity: 0.5 }}>Waiting for speech...</p> :
              transcripts.map((t, i) => <p key={i}> {t}</p>)
            }
          </div>
        </div>

        <div className="analysis-area">
          <h2>AI Insights</h2>
          <div className="scroll-box" ref={analysisRef}>
            {analyses.length === 0 ? <p style={{ opacity: 0.5 }}>Waiting for context...</p> :
              analyses.map((a, i) => (
                <div key={i} style={{ marginBottom: '15px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '10px' }}>
                  {a}
                </div>
              ))
            }
          </div>
        </div>
      </div>

      <div className="controls-footer">
        <button
          className={`glass-btn ${isRecording ? "recording" : ""}`}
          onClick={toggleRecording}
          disabled={status !== "Connected"}
        >
          {isRecording ? "Stop Listening" : "Start Listening"}
        </button>
      </div>
    </div>
  )
}

export default App
