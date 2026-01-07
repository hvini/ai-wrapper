const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 450,
        height: 600,
        transparent: true,
        frame: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false, // For simple prototype; specific security usually requires true + preload
        },
    });

    const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, '../dist/index.html')}`;
    mainWindow.loadURL(startUrl);

    mainWindow.on('closed', function () {
        mainWindow = null;
    });
}

function startPythonBackend() {
    // Path to python script
    // In dev: backend/main.py
    // In prod: bundled path
    const scriptPath = path.join(__dirname, '../backend/main.py');
    const pythonExecutable = path.join(__dirname, '../backend/venv/bin/python'); // Assuming venv is created

    console.log('Starting Python backend using:', pythonExecutable, scriptPath);

    pythonProcess = spawn(pythonExecutable, [scriptPath]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}

app.on('ready', () => {
    startPythonBackend();
    createWindow();
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

app.on('activate', function () {
    if (mainWindow === null) {
        createWindow();
    }
});
