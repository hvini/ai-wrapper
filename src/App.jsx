import { useState, useEffect, useRef } from 'react'
import './index.css'

function App() {
  const [status, setStatus] = useState("Disconnected");
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [deviceStatus, setDeviceStatus] = useState("Detecting audio...");
  const [context, setContext] = useState("Seja direto e vá direto ao ponto.");
  const [activeTab, setActiveTab] = useState("live");
  const [modelSize, setModelSize] = useState("small");
  const [expandedSection, setExpandedSection] = useState(null); // 'transcription', 'analysis', or null
  const [llmModels, setLlmModels] = useState({ local: [], available: [], current: null });

  const ws = useRef(null);
  const transcriptionRef = useRef(null);
  const analysisRef = useRef(null);

  useEffect(() => {
    connectWs();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  useEffect(() => {
    const handler = setTimeout(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: "update_config", model: modelSize }));
      }
    }, 500);
    return () => clearTimeout(handler);
  }, [modelSize, status]);

  // Send context update when it changes (debounced)
  useEffect(() => {
    const handler = setTimeout(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: "update_context", context: context }));
      }
    }, 500);
    return () => clearTimeout(handler);
  }, [context, status]);

  useEffect(() => {
    if (activeTab === 'context' && ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "get_llm_models" }));
    }
  }, [activeTab, status]);

  // Auto-scroll
  useEffect(() => {
    if (transcriptionRef.current) {
      transcriptionRef.current.scrollTop = transcriptionRef.current.scrollHeight;
    }
  }, [transcripts]);




  const connectWs = () => {
    setStatus("Connecting...");
    ws.current = new WebSocket("ws://localhost:8765");

    ws.current.onopen = () => {
      setStatus("Connected");
      ws.current.send(JSON.stringify({ type: "get_devices" }));
      ws.current.send(JSON.stringify({ type: "get_llm_models" }));
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
          // Optional: Display status toast or something
          break;
        case "llm_models":
          setLlmModels({
            local: data.local || [],
            available: data.available || [],
            current: data.current
          });
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

  const loadLlm = (filename) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    ws.current.send(JSON.stringify({ type: "load_llm_model", filename }));
  };

  const downloadModel = (modelId) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    const confirmDownload = confirm("Downloading a large model (4GB+). This will happen in the background. Continue?");
    if (confirmDownload) {
      ws.current.send(JSON.stringify({ type: "download_model", model_id: modelId }));
    }
  };

  // Toggle expansion helper
  const toggleExpand = (section) => {
    if (expandedSection === section) {
      setExpandedSection(null); // Reset to split view
    } else {
      setExpandedSection(section);
    }
  };

  const getSectionClass = (section) => {
    if (!expandedSection) return ""; // Default split
    return expandedSection === section ? "expanded" : "collapsed";
  };

  return (
    <div className="glass-container">
      <div className="header">
        <h1>AI Wrapper</h1>
        <div className={`status-check ${status.toLowerCase()}`}>
          {status === "Connected" ? deviceStatus : status}
        </div>
      </div>

      <div className="location-tabs">
        <button
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          Live Monitor
        </button>
        <button
          className={`tab-btn ${activeTab === 'context' ? 'active' : ''}`}
          onClick={() => setActiveTab('context')}
        >
          Context & Setup
        </button>
      </div>

      <div className="main-content">
        {activeTab === 'context' ? (
          <div className="context-container">
            <div className="input-group" style={{ flex: '0 0 auto', marginBottom: '20px' }}>
              <label>Whisper Model Size</label>
              <select
                className="context-input"
                value={modelSize}
                onChange={(e) => setModelSize(e.target.value)}
                style={{ padding: '12px', cursor: 'pointer' }}
              >
                <option value="tiny">Tiny (Fastest, Low Accuracy)</option>
                <option value="base">Base (Balanced)</option>
                <option value="small">Small (Good Accuracy)</option>
                <option value="medium">Medium (Better, Slower)</option>
                <option value="large-v3">Large v3 (Best, Slowest)</option>
              </select>
            </div>

            <div className="input-group" style={{ flex: '0 0 auto', marginBottom: '20px' }}>
              <label>LLM Model (GGUF)</label>
              <div className="model-manager">
                {llmModels.available.map((model) => {
                  const isDownloaded = llmModels.local.includes(model.filename);
                  const isActive = llmModels.current === model.filename;

                  return (
                    <div key={model.id} className="model-item">
                      <div className="model-info">
                        <span className="model-name">{model.name}</span>
                        {isActive && <span className="tag active-tag">Active</span>}
                        {isDownloaded && !isActive && <span className="tag downloaded-tag">Downloaded</span>}
                      </div>
                      <div className="model-actions">
                        {isDownloaded ? (
                          isActive ? (
                            <button className="small-glass-btn disabled" disabled>Active</button>
                          ) : (
                            <button className="small-glass-btn" onClick={() => loadLlm(model.filename)}>Load</button>
                          )
                        ) : (
                          <button className="small-glass-btn download-btn" onClick={() => downloadModel(model.id)}>Download</button>
                        )}
                      </div>
                    </div>
                  )
                })}

                {/* Show other local models not in available list (custom models) */
                  llmModels.local
                    .filter(f => !llmModels.available.some(am => am.filename === f))
                    .map(f => (
                      <div key={f} className="model-item">
                        <div className="model-info">
                          <span className="model-name">{f} (Custom)</span>
                          {llmModels.current === f && <span className="tag active-tag">Active</span>}
                        </div>
                        <div className="model-actions">
                          {llmModels.current !== f && (
                            <button className="small-glass-btn" onClick={() => loadLlm(f)}>Load</button>
                          )}
                        </div>
                      </div>
                    ))
                }
              </div>
            </div>

            <div className="input-group">
              <label>System Context / Instructions</label>
              <textarea
                className="context-input full-height"
                placeholder="Ex: 'You are a helpful assistant assisting with a medical diagnosis. Focus on key symptoms mentioned...'"
                value={context}
                onChange={(e) => setContext(e.target.value)}
              />
              <p className="hint-text">
                This context is sent to the local LLM to guide the analysis of the live transcription.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className={`transcription-area ${getSectionClass('transcription')}`}>
              <div className="area-header">
                <div className="header-title">
                  <h2>Transcription</h2>
                  <span className="live-indicator"></span>
                </div>
                <button
                  className="icon-btn"
                  onClick={() => toggleExpand('transcription')}
                  title={expandedSection === 'transcription' ? "Restore Split View" : "Maximize view"}
                >
                  {expandedSection === 'transcription' ? "Exit Fullscreen" : "Fullscreen"}
                </button>
              </div>
              <div className="scroll-box" ref={transcriptionRef}>
                {transcripts.length === 0 ?
                  <div className="empty-state">Waiting for speech...</div> :
                  <div className="transcription-stream">
                    {transcripts.map((t, i) => <span key={i}>{t} </span>)}
                  </div>
                }
              </div>
            </div>

            <div className={`analysis-area ${getSectionClass('analysis')}`}>
              <div className="area-header">
                <div className="header-title">
                  <h2>AI Insights</h2>
                </div>
                <button
                  className="icon-btn"
                  onClick={() => toggleExpand('analysis')}
                  title={expandedSection === 'analysis' ? "Restore Split View" : "Maximize view"}
                >
                  {expandedSection === 'analysis' ? "Exit Fullscreen" : "Fullscreen"}
                </button>
              </div>
              <div className="scroll-box" ref={analysisRef}>
                {analyses.length === 0 ?
                  <div className="empty-state">Waiting for context...</div> :
                  analyses.map((a, i) => (
                    <div key={i} className="analysis-item">
                      {a}
                    </div>
                  ))
                }
              </div>
            </div>
          </>
        )}
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
      <div className="resize-handle" />
    </div>
  )
}

export default App
