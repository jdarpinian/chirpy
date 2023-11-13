import { app, BrowserWindow } from 'electron';

let createWindow = async () => {
    let win = new BrowserWindow({autoHideMenuBar: true, show: false, x: 100, y: 100, webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
        devTools: true,
    }});
    win.loadFile('index.html');
    win.once('ready-to-show', ()=>win.show());
}

(async ()=>{
    await app.whenReady();
    if (process.platform !== 'darwin') app.on('window-all-closed', ()=>app.quit());
    app.on('web-contents-created', (e, contents)=>{
        // Disable page navigation and new windows for security
        contents.openDevTools();
        contents.on('will-navigate', (e, url)=>e.preventDefault());
        contents.setWindowOpenHandler(({url})=>{
            return { action: 'deny' };
        });
    });
    app.on('activate', ()=>{
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
    createWindow();
    createWindow();
})();

import WebSocket, { WebSocketServer } from 'ws';

const wss = new WebSocketServer({ port: 8089 });

wss.on('connection', ws => {
    ws.on('message', (message, isBinary) => {
        console.log('received: %s', message);
        wss.clients.forEach(client => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
                client.send(message, { binary: isBinary });
            }
        });
    });
});

