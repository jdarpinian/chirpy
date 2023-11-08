import { app, BrowserWindow } from 'electron';

let createWindow = async () => {
    let win = new BrowserWindow({autoHideMenuBar: true, show: false, webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
    }});
    win.loadFile('index.html');
    win.once('ready-to-show', ()=>win.show());
}

(async ()=>{
    await app.whenReady();
    if (process.platform !== 'darwin') app.on('window-all-closed', ()=>app.quit());
    app.on('web-contents-created', (e, contents)=>{
        // Disable page navigation and new windows for security
        contents.on('will-navigate', (e, url)=>e.preventDefault());
        contents.setWindowOpenHandler(({url})=>{
            return { action: 'deny' };
        });
    });
    app.on('activate', ()=>{
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
    createWindow();
})();

