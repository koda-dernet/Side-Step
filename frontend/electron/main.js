const { app, BrowserWindow, Menu, Notification, ipcMain } = require("electron");
const path = require("path");

function argVal(name) {
  const prefix = `--${name}=`;
  const arg = process.argv.find((a) => a.startsWith(prefix));
  return arg ? arg.slice(prefix.length) : null;
}

const serverUrl = argVal("url") || "http://127.0.0.1:8770";
const authToken = argVal("token") || "";

app.setName("Side-Step");
if (process.platform === "linux") app.setDesktopName("Side-Step");

// GPU flags
app.commandLine.appendSwitch("ignore-gpu-blocklist");
app.commandLine.appendSwitch("enable-gpu-rasterization");

if (process.platform === "linux") {
  app.commandLine.appendSwitch("ozone-platform", "x11");
  app.commandLine.appendSwitch("disable-gpu-sandbox");
}

Menu.setApplicationMenu(null);

ipcMain.handle("get-token", () => authToken);
ipcMain.handle("notify", (_e, payload) => {
  const title = String(payload?.title || "Side-Step");
  const body = String(payload?.body || "");
  if (!Notification.isSupported()) return { ok: false, error: "unsupported" };
  const icon = resolveIcon();
  const notification = new Notification({ title, body, icon, silent: false });
  notification.show();
  return { ok: true };
});
ipcMain.on("boot-error", (_e, msg) => {
  console.error("[Side-Step] Boot error:", msg);
});

function resolveIcon() {
  const assetsDir = path.join(__dirname, "..", "assets");
  const fs = require("fs");
  if (process.platform === "win32") {
    const ico = path.join(assetsDir, "icon.ico");
    if (fs.existsSync(ico)) return ico;
  }
  const png = path.join(assetsDir, "icon.png");
  if (fs.existsSync(png)) return png;
  return undefined;
}

app.whenReady().then(() => {
  const iconPath = resolveIcon();

  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: "#16161E",
    transparent: false,
    hasShadow: true,
    autoHideMenuBar: true,
    title: "Side-Step",
    icon: iconPath,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const sep = serverUrl.includes("?") ? "&" : "?";
  const appURL = `${serverUrl}${sep}token=${authToken}`;
  win.loadURL(appURL);
  win.maximize();

  win.webContents.on("did-finish-load", () => {
    win.focus();
    win.webContents.focus();
  });

  win.webContents.on("did-start-navigation", (_e, url, isInPlace, isMainFrame) => {
    if (isMainFrame) {
      console.error("[Side-Step] did-start-navigation:", { url, isInPlace, isMainFrame });
    }
  });
  win.webContents.on("did-navigate", (_e, url) => {
    console.error("[Side-Step] did-navigate:", url);
  });
  win.webContents.on("did-navigate-in-page", (_e, url, isMainFrame) => {
    if (isMainFrame) {
      console.error("[Side-Step] did-navigate-in-page:", url);
    }
  });

  // ── Prevent the renderer from navigating away (blank-page guard) ──
  const _allowedPaths = new Set(["/", "/theme-editor"]);
  win.webContents.on("will-navigate", (e, url) => {
    const dest = new URL(url);
    const home = new URL(appURL);
    const allowed =
      dest.origin === home.origin && _allowedPaths.has(dest.pathname);
    if (!allowed) {
      console.error("[Side-Step] blocked will-navigate:", url);
      e.preventDefault();
    }
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const dest = new URL(url);
      const home = new URL(appURL);
      if (dest.origin === home.origin && _allowedPaths.has(dest.pathname)) {
        return { action: "allow" };
      }
    } catch (_) { /* malformed URL */ }
    console.error("[Side-Step] blocked window.open:", url);
    return { action: "deny" };
  });

  // ── DevTools shortcut (F12 or Ctrl+Shift+I) ──
  win.webContents.on("before-input-event", (_e, input) => {
    if (
      input.type === "keyDown" &&
      (input.key === "F12" ||
        (input.key === "I" && input.control && input.shift))
    ) {
      win.webContents.toggleDevTools();
    }
  });

  // ── Renderer crash detection ──
  win.webContents.on("render-process-gone", (_e, details) => {
    console.error("[Side-Step] Renderer crashed:", details.reason, details.exitCode);
  });

  // Custom close flow: ask the renderer to show an in-app modal instead of
  // the stock OS dialog.  The renderer replies via "confirm-close" IPC.
  let forceClose = false;

  win.on("close", (e) => {
    if (!forceClose) {
      e.preventDefault();
      win.webContents.send("close-requested");
    }
  });

  // Renderer confirmed the user wants to leave.
  ipcMain.on("confirm-close", () => {
    forceClose = true;
    win.close();
  });

  // Renderer says the user cancelled — allow beforeunload to pass through
  // harmlessly if it fires on a future close.
  win.webContents.on("will-prevent-unload", (e) => {
    if (forceClose) e.preventDefault();
  });

  win.on("closed", () => app.quit());
});

app.on("window-all-closed", () => app.quit());
