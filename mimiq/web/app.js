(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const token = new URLSearchParams(location.search).get("t") || "";
  const store = (() => { try { return JSON.parse(localStorage.getItem("mimiq") || "{}"); } catch { return {}; } })();
  const cfg = Object.assign({ facing: "user", res: "1280x720", fps: 30, quality: 0.82 }, store);
  const save = () => localStorage.setItem("mimiq", JSON.stringify(cfg));

  let ws = null, stream = null, running = false, inflight = 0, seq = 0, lastSent = 0;
  let wakeLock = null, retry = 0, sentBytes = 0, sentFrames = 0, rtt = 0, statT = performance.now();
  // Transport: WebSocket first; if Safari refuses WSS (self-signed certificate), fall back to HTTPS POST.
  let mode = "ws", wsFails = 0, httpOk = false;
  const MAX_INFLIGHT = 2;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });

  const toast = (msg, ms = 2600) => {
    const t = $("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), ms);
  };
  const pill = (state, text) => { const p = $("pill"); p.className = "pill " + state; p.querySelector("span").textContent = text; };
  const show = (id) => ["intro", "blocked", "live"].forEach((s) => $(s).classList.toggle("hidden", s !== id));

  // ---------------------------------------------------------------- guards
  if (!window.MIMIQ_AUTH || !token) {
    show("blocked");
    $("blockedText").textContent = "Откройте Mimiq на компьютере → «Подключить iPhone» и отсканируйте QR-код камерой iPhone.";
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    show("blocked");
    $("blockedText").textContent = "Safari не разрешает камеру на этой странице. Откройте ссылку из QR-кода в Safari и подтвердите переход на сайт, либо установите сертификат Mimiq.";
    return;
  }

  // ---------------------------------------------------------------- socket
  function connect() {
    if (mode === "http") return;
    if (ws && (ws.readyState === 0 || ws.readyState === 1)) return;
    pill("wait", "Подключение…");
    let opened = false;
    try { ws = new WebSocket(`wss://${location.host}/ws?t=${encodeURIComponent(token)}`); }
    catch { useHttp(); return; }
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { opened = true; wsFails = 0; retry = 0; inflight = 0; pill("ok", "Подключено к ПК"); hello(); };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      handle(m);
    };
    ws.onclose = () => {
      inflight = 0;
      if (!opened && ++wsFails >= 2) { useHttp(); return; }
      pill("", "Нет связи");
      const delay = Math.min(4000, 500 * 2 ** retry++);
      setTimeout(connect, delay);
    };
    ws.onerror = () => {};
  }
  function useHttp() {
    mode = "http"; ws = null; inflight = 0;
    pill("wait", "Подключение (HTTPS)…");
    hello();
  }
  const linkUp = () => (mode === "http" ? httpOk : !!ws && ws.readyState === 1);
  function post(path, body, type) {
    return fetch(`/${path}?t=${encodeURIComponent(token)}`, { method: "POST", body, cache: "no-store",
      headers: { "Content-Type": type } }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))));
  }
  function onReply(r) {
    if (!httpOk) { httpOk = true; pill("ok", "Подключено к ПК"); }
    if (r && r.t === "ack") handle(r);
    if (r && r.msgs) r.msgs.forEach(handle);
  }
  function onHttpFail() { if (httpOk) { httpOk = false; pill("", "Нет связи"); } }
  function sendJson(obj) {
    if (mode === "http") post(obj.t, JSON.stringify(obj), "application/json").then(onReply, onHttpFail);
    else if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
  }
  function sendFrame(data) {
    if (mode === "http") {
      post("frame", data, "application/octet-stream").then(onReply, () => { inflight = Math.max(0, inflight - 1); onHttpFail(); });
    } else if (ws && ws.readyState === 1) ws.send(data);
    else inflight = Math.max(0, inflight - 1);
  }
  function handle(m) {
    if (m.t === "ack") {
      inflight = Math.max(0, inflight - 1);
      const now = performance.now() >>> 0;
      const d = (now - m.c) >>> 0; if (d < 5000) rtt = rtt ? rtt * 0.8 + d * 0.2 : d;
    } else if (m.t === "config") {
      let restart = false;
      if (m.facing && m.facing !== cfg.facing) { cfg.facing = m.facing; restart = true; }
      if (m.res && m.res !== cfg.res) { cfg.res = m.res; restart = true; }
      if (m.fps && +m.fps !== cfg.fps) { cfg.fps = +m.fps; restart = true; }
      if (m.quality) cfg.quality = +m.quality;
      save(); syncUi(); if (restart && running) startCamera();
    } else if (m.t === "toast") toast(m.text || "");
  }
  function hello() {
    if (mode !== "http" && (!ws || ws.readyState !== 1)) return;
    const s = stream ? stream.getVideoTracks()[0].getSettings() : {};
    sendJson({ t: "hello", device: deviceName(), camera: cfg.facing, width: s.width || 0,
      height: s.height || 0, fps: s.frameRate || cfg.fps, ua: navigator.userAgent });
  }
  function deviceName() {
    const ua = navigator.userAgent;
    if (/iPhone/.test(ua)) return "iPhone"; if (/iPad/.test(ua)) return "iPad"; if (/Android/.test(ua)) return "Android";
    return "Телефон";
  }

  // ---------------------------------------------------------------- camera
  async function startCamera() {
    const [w, h] = cfg.res.split("x").map(Number);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: {
        facingMode: { ideal: cfg.facing }, width: { ideal: w }, height: { ideal: h },
        frameRate: { ideal: cfg.fps, max: cfg.fps } } });
    } catch (e) {
      $("introHint").textContent = e && e.name === "NotAllowedError"
        ? "Доступ к камере запрещён. Разрешите его: Настройки → Safari → Камера."
        : "Не удалось включить камеру: " + (e && e.message ? e.message : e);
      show("intro"); return;
    }
    video.srcObject = stream;
    video.classList.toggle("mirror", cfg.facing === "user");
    try { await video.play(); } catch {}
    const track = stream.getVideoTracks()[0];
    track.onended = () => { running = false; show("intro"); $("introHint").textContent = "Камера остановлена — нажмите, чтобы продолжить."; };
    running = true; show("live"); syncUi(); hello(); keepAwake(); loop();
  }

  function loop() {
    const next = () => (video.requestVideoFrameCallback ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick));
    function tick() {
      if (!running) return;
      next();
      if (!linkUp() || inflight >= MAX_INFLIGHT || document.hidden) return;
      const now = performance.now();
      if (now - lastSent < 1000 / cfg.fps - 3) return;
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh) return;
      lastSent = now;
      if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
      ctx.drawImage(video, 0, 0, vw, vh);
      inflight++;
      const id = ++seq >>> 0, stamp = now >>> 0;
      canvas.toBlob((blob) => {
        if (!blob || !linkUp()) { inflight = Math.max(0, inflight - 1); return; }
        const hdr = new DataView(new ArrayBuffer(12));
        hdr.setUint8(0, 77); hdr.setUint8(1, 81); hdr.setUint8(2, 1); hdr.setUint8(3, 0);
        hdr.setUint32(4, id, true); hdr.setUint32(8, stamp, true);
        sendFrame(new Blob([hdr.buffer, blob]));
        sentBytes += blob.size + 12; sentFrames++;
      }, "image/jpeg", cfg.quality);
    }
    next();
  }

  setInterval(() => {
    const now = performance.now(), dt = (now - statT) / 1000; statT = now;
    const fps = sentFrames / dt, mbps = (sentBytes * 8) / 1e6 / dt; sentFrames = 0; sentBytes = 0;
    const s = stream ? stream.getVideoTracks()[0].getSettings() : {};
    const text = `${fps.toFixed(0)} к/с · ${mbps.toFixed(1)} Мбит/с · ${rtt ? rtt.toFixed(0) : "—"} мс` +
      (s.width ? ` · ${s.width}×${s.height}` : "");
    $("stats").textContent = text; $("dimStats").textContent = text;
    // in HTTPS mode stats double as a heartbeat, so the PC knows the phone is still here
    if ((running && linkUp()) || mode === "http") sendJson({ t: "stats", fps, mbps, rtt, width: s.width, height: s.height, camera: cfg.facing });
  }, 1000);

  async function keepAwake() {
    try { if ("wakeLock" in navigator && !wakeLock) { wakeLock = await navigator.wakeLock.request("screen"); wakeLock.addEventListener("release", () => (wakeLock = null)); } } catch {}
  }
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { keepAwake(); connect(); if (running && stream && stream.getVideoTracks()[0].readyState !== "live") startCamera(); }
  });

  // ---------------------------------------------------------------- ui
  function syncUi() {
    $("flipLabel").textContent = cfg.facing === "user" ? "Фронтальная камера" : "Основная камера";
    document.querySelectorAll("#resSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === cfg.res));
    document.querySelectorAll("#fpsSeg button").forEach((b) => b.classList.toggle("on", +b.dataset.v === cfg.fps));
    $("quality").value = Math.round(cfg.quality * 100); $("qVal").textContent = Math.round(cfg.quality * 100);
  }
  $("startBtn").onclick = () => startCamera();
  $("flipBtn").onclick = () => { cfg.facing = cfg.facing === "user" ? "environment" : "user"; save(); toast(cfg.facing === "user" ? "Фронтальная камера" : "Основная камера — лучше качество"); startCamera(); };
  $("resSeg").onclick = (e) => { const v = e.target.dataset && e.target.dataset.v; if (v && v !== cfg.res) { cfg.res = v; save(); startCamera(); } };
  $("fpsSeg").onclick = (e) => { const v = e.target.dataset && +e.target.dataset.v; if (v && v !== cfg.fps) { cfg.fps = v; save(); startCamera(); } };
  $("quality").oninput = (e) => { cfg.quality = +e.target.value / 100; $("qVal").textContent = e.target.value; save(); };
  $("dimBtn").onclick = () => $("dim").classList.remove("hidden");
  $("dim").onclick = () => $("dim").classList.add("hidden");

  syncUi(); show("intro"); connect();
})();
