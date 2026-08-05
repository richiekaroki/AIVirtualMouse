/* ── Hand Motion — Client-Side Detection ── */
/* MediaPipe Hands runs in the browser via WASM. No server CPU needed. */

let socket, localStream, videoEl, canvasEl, ctx;
let mpHands = null, mpCamera = null;
let handResults = null;
let lastFpsTime = Date.now(), fpsCounter = 0;
let frameCount = 0;
let selectedRec = null;
let uniqueGestures = new Set();
let isRecording = false, recordChunks = [];
let sessionStart = 0, sessionTimer = null, uniqueGestureCount = 0;
let onboardStep = -1, onboardActive = false;
let lastDetectedGesture = null;
let soundEnabled = false, audioCtx = null;

/* ── Virtual Mouse / Cursor Control ── */
let cursorMode = false;
let cursorX = 0, cursorY = 0;
let cursorSmoothX = 0, cursorSmoothY = 0;
let cursorClicking = false;
let lastClickTime = 0;
let cursorTrail = [];
const SMOOTHING = 0.28;
const CLICK_COOLDOWN = 350;

/* ── Gesture Detection Rules ── */
const TIP_IDS = [4, 8, 12, 16, 20];
const MCP_IDS = [2, 5, 9, 13, 17];
const PIP_IDS = [3, 6, 10, 14, 18];

function countFingersUp(landmarks) {
    const fingers = [];
    if (!landmarks || landmarks.length < 21) return fingers;
    const lm = landmarks;
    fingers.push(lm[4].x < lm[3].x ? 1 : 0);
    for (let i = 1; i < 5; i++) {
        fingers.push(lm[TIP_IDS[i]].y < lm[PIP_IDS[i]].y ? 1 : 0);
    }
    return fingers;
}

function classifyGesture(fingers, landmarks) {
    if (!fingers || fingers.length < 5) return null;
    const [t, i, m, r, p] = fingers;
    const count = t + i + m + r + p;
    const allUp = t && i && m && r && p;
    const allDown = !t && !i && !m && !r && !p;
    const lm = landmarks;
    if (allUp) return 'OPEN_HAND';
    if (allDown) return 'FIST';
    if (i && !m && !r && !p) return 'POINT';
    if (i && m && !r && !p) return 'PEACE_V';
    if (t && !i && !m && !r && !p) return 'THUMBS_UP';
    if (t && i && !m && !r && !p) {
        const d = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y);
        if (d < 0.06) return 'OK_SIGN';
    }
    if (!t && i && !m && !r && p) return 'SHAKA';
    if (t && !i && !m && !r && p) return 'PINKY';
    if (t && i && !m && !r && !p) return 'GUN';
    if (i && m && r && !p && !t) return 'FOUR';
    if (i && m && r && p) return 'FIVE';
    if (!i && m && r && p && !t) return 'THREE';
    if (!t && i && m && !r && !p) return 'LOVE';
    if (t && i && !m && r && p) return 'SPIDER';
    if (!t && i && m && r && p) return 'ROCK';
    if (count === 2 && i && p && !t && !m && !r) return 'HANG_LOOSE';
    if (t && !i && !m && !r && !p) {
        const thumbUp = lm[4].y < lm[3].y;
        if (thumbUp) return 'THUMBS_UP';
    }
    return null;
}

function computeFeatures(landmarks) {
    if (!landmarks || landmarks.length < 21) return null;
    const lm = landmarks;
    let openCount = 0;
    for (let i = 1; i < 5; i++) {
        if (lm[TIP_IDS[i]].y < lm[MCP_IDS[i]].y) openCount++;
    }
    const pinch = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) * 640;
    const span = Math.hypot(lm[4].x - lm[20].x, lm[4].y - lm[20].y) * 640;
    return {
        hand_openness: openCount / 4,
        pinch_distance: Math.round(pinch),
        hand_span: Math.round(span)
    };
}

/* ── Sound ── */
function toggleSound() {
    soundEnabled = !soundEnabled;
    const btn = document.getElementById('sound-btn');
    btn.classList.toggle('on', soundEnabled);
    document.getElementById('sound-icon-on').style.display = soundEnabled ? '' : 'none';
    document.getElementById('sound-icon-off').style.display = soundEnabled ? 'none' : '';
    if (soundEnabled && !audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (soundEnabled) playGestureSound(880, 0.08);
}

function playGestureSound(freq, dur) {
    if (!soundEnabled || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + dur);
    } catch(e) {}
}

/* ── Virtual Mouse Functions ── */
function toggleCursorMode() {
    cursorMode = !cursorMode;
    const btn = document.getElementById('btn-cursor');
    const indicator = document.getElementById('cursor-indicator');
    btn.classList.toggle('on', cursorMode);
    if (cursorMode) {
        indicator.style.display = '';
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="M13 13l6 6"/></svg> Cursor ON';
    } else {
        indicator.style.display = 'none';
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="M13 13l6 6"/></svg> Cursor';
    }
}

function updateCursor(lm) {
    if (!cursorMode || !lm || lm.length < 21) return;
    const indexTip = lm[8];
    const screenW = window.innerWidth;
    const screenH = window.innerHeight;
    const targetX = (1 - indexTip.x) * screenW;
    const targetY = indexTip.y * screenH;
    cursorSmoothX += (targetX - cursorSmoothX) * SMOOTHING;
    cursorSmoothY += (targetY - cursorSmoothY) * SMOOTHING;
    cursorX = cursorSmoothX;
    cursorY = cursorSmoothY;
    moveCursorIndicator(cursorX, cursorY);
    cursorTrail.push({ x: cursorX, y: cursorY, t: Date.now() });
    if (cursorTrail.length > 8) cursorTrail.shift();
    drawCursorTrail();
    checkClickGesture(lm);
}

function moveCursorIndicator(x, y) {
    const el = document.getElementById('cursor-indicator');
    if (!el) return;
    el.style.left = x + 'px';
    el.style.top = y + 'px';
}

function drawCursorTrail() {
    if (cursorTrail.length < 2) return;
    const now = Date.now();
    for (let i = 1; i < cursorTrail.length; i++) {
        const p = cursorTrail[i];
        const age = (now - p.t) / 300;
        if (age > 1) continue;
        const alpha = (1 - age) * 0.4;
        const size = (1 - age) * 3;
        const el = document.createElement('div');
        el.style.cssText = `position:fixed;left:${p.x}px;top:${p.y}px;width:${size}px;height:${size}px;border-radius:50%;background:rgba(245,158,11,${alpha});pointer-events:none;z-index:9998;transform:translate(-50%,-50%)`;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 300);
    }
}

function checkClickGesture(lm) {
    const dist = Math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y);
    const now = Date.now();
    if (dist < 0.05 && !cursorClicking && now - lastClickTime > CLICK_COOLDOWN) {
        cursorClicking = true;
        lastClickTime = now;
        fireClickAt(cursorX, cursorY);
    } else if (dist > 0.08) {
        cursorClicking = false;
    }
}

function fireClickAt(x, y) {
    const target = document.elementFromPoint(x, y);
    if (target) {
        target.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: x, clientY: y }));
        const ring = document.getElementById('cursor-click-ring');
        if (ring) {
            ring.style.left = x + 'px';
            ring.style.top = y + 'px';
            ring.style.display = '';
            ring.classList.remove('animate');
            void ring.offsetWidth;
            ring.classList.add('animate');
            setTimeout(() => { ring.style.display = 'none'; ring.classList.remove('animate'); }, 500);
        }
        playGestureSound(1200, 0.05);
    }
}

/* ── Onboarding ── */
const ONBOARD_STEPS = [
    { gesture: 'OPEN_HAND', emoji: '\u{1F590}\uFE0F', name: 'Open Hand', hint: 'Spread all five fingers and hold your hand open' },
    { gesture: 'FIST', emoji: '\u270A', name: 'Fist', hint: 'Close all five fingers into a tight fist' },
    { gesture: 'POINT', emoji: '\u261D\uFE0F', name: 'Point', hint: 'Extend only your index finger' },
];

function startOnboarding() { onboardStep = 0; onboardActive = true; showOnboardStep(0); document.getElementById('onboard-toast').classList.add('visible'); }
function showOnboardStep(i) { const s = ONBOARD_STEPS[i]; document.getElementById('onboard-step').textContent = 'Step ' + (i+1) + ' of ' + ONBOARD_STEPS.length; document.getElementById('onboard-gesture').textContent = s.emoji; document.getElementById('onboard-name').textContent = s.name; document.getElementById('onboard-hint').textContent = s.hint; for (let j = 0; j < ONBOARD_STEPS.length; j++) document.getElementById('ob-dot-' + j).className = 'onboard-dot' + (j < i ? ' done' : j === i ? ' active' : ''); }
function advanceOnboarding() { onboardStep++; if (onboardStep >= ONBOARD_STEPS.length) { finishOnboarding(); return; } showOnboardStep(onboardStep); }
function finishOnboarding() { onboardActive = false; document.getElementById('onboard-toast').classList.remove('visible'); try { sessionStorage.setItem('onboard_done', '1'); } catch(e) {} }
function skipOnboarding() { finishOnboarding(); }
function trackGesture(name) { if (!name) return; if (!uniqueGestures.has(name)) { uniqueGestures.add(name); uniqueGestureCount++; document.getElementById('stat-gestures').textContent = uniqueGestureCount; } if (onboardActive && onboardStep >= 0 && name === ONBOARD_STEPS[onboardStep].gesture) advanceOnboarding(); }

/* ── Session ── */
function startSession() { sessionStart = Date.now(); uniqueGestureCount = 0; document.getElementById('stat-gestures').textContent = '0'; document.getElementById('stat-uptime').textContent = '0:00'; if (sessionTimer) clearInterval(sessionTimer); sessionTimer = setInterval(updateSessionUptime, 1000); }
function stopSession() { if (sessionTimer) clearInterval(sessionTimer); sessionTimer = null; sessionStart = 0; }
function updateSessionUptime() { if (!sessionStart) return; const s = Math.floor((Date.now() - sessionStart) / 1000); const m = Math.floor(s / 60), sec = s % 60; document.getElementById('stat-uptime').textContent = m + ':' + String(sec).padStart(2, '0'); }

/* ── WebSocket ── */
function initSocket() {
    socket = io({ transports: ['polling', 'websocket'] });
    socket.on('connect', () => { setWsStatus(true); loadRecordings(); });
    socket.on('disconnect', () => setWsStatus(false));
    socket.on('reconnect_attempt', () => setWsConnecting());
    socket.on('reconnect_failed', () => setWsStatus(false));
    socket.on('playback_frame', d => {
        if (d.landmarks && d.landmarks.length >= 21) {
            drawSkeleton(d.landmarks, 0);
            updateGesture(d.primitive);
            updateFingers(d.finger_states);
            if (d.features) updateFeatures(d.features, d.velocity);
            updateHandedness(d.handedness);
        }
    });
    socket.on('playback_finished', () => { clearOverlay(); updateGesture(null); updateHandedness(null); });
}

function setWsStatus(on) { document.getElementById('ws-badge').className = 'ws-pill ' + (on ? 'on' : 'off'); document.getElementById('ws-dot').className = 'ws-dot ' + (on ? 'on' : 'off'); document.getElementById('ws-text').textContent = on ? 'Live' : 'Offline'; }
function setWsConnecting() { document.getElementById('ws-badge').className = 'ws-pill connecting'; document.getElementById('ws-dot').className = 'ws-dot connecting'; document.getElementById('ws-text').textContent = 'Connecting...'; }

/* ── Skeleton Drawing ── */
function drawSkeleton(landmarks, handIdx = 0) {
    if (!ctx || !canvasEl || !landmarks || landmarks.length < 21) return;
    const w = canvasEl.width, h = canvasEl.height;
    const conns = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
    const palettes = [
        { line: '#F59E0B', glow: 'rgba(245,158,11,0.35)', tip: '#FBBF24', joint: '#E8E8EC' },
        { line: '#A78BFA', glow: 'rgba(167,139,250,0.35)', tip: '#C4B5FD', joint: '#DDD6FE' },
    ];
    const c = palettes[handIdx % palettes.length];
    ctx.save();
    ctx.shadowColor = c.glow;
    ctx.shadowBlur = 12;
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const [a, b] of conns) {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
        ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
        ctx.stroke();
    }
    ctx.shadowBlur = 0;
    for (let i = 0; i < 21; i++) {
        const x = landmarks[i].x * w, y = landmarks[i].y * h;
        const tip = TIP_IDS.includes(i);
        ctx.fillStyle = tip ? c.tip : c.joint;
        ctx.beginPath();
        ctx.arc(x, y, tip ? 5.5 : 3, 0, Math.PI * 2);
        ctx.fill();
        if (tip) {
            ctx.strokeStyle = c.line;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.stroke();
        }
    }
    ctx.restore();
}

function clearOverlay() { if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height); }

/* ── Particles ── */
const particles = [];
function spawnParticles(lm, color) {
    if (!ctx || !canvasEl || !lm || lm.length < 21) return;
    const w = canvasEl.width, h = canvasEl.height;
    for (const i of TIP_IDS) {
        const x = lm[i].x * w, y = lm[i].y * h;
        for (let j = 0; j < 4; j++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 1.5 + Math.random() * 3;
            particles.push({ x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 1, decay: 0.015 + Math.random() * 0.02, size: 2 + Math.random() * 2.5, color });
        }
    }
}
function updateParticles() {
    if (!ctx || !particles.length) return;
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx; p.y += p.vy; p.vy += 0.05; p.life -= p.decay;
        if (p.life <= 0) { particles.splice(i, 1); continue; }
        ctx.globalAlpha = p.life; ctx.fillStyle = p.color;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
}

/* ── Trail ── */
const trailPoints = [];
const TRAIL_MAX = 20;
function addTrailPoint(lm, handIdx) {
    if (!lm || lm.length < 21) return;
    const pts = TIP_IDS.map(i => ({ x: lm[i].x, y: lm[i].y }));
    trailPoints.push({ pts, time: Date.now(), handIdx });
    if (trailPoints.length > TRAIL_MAX) trailPoints.shift();
}
function drawTrail() {
    if (!ctx || trailPoints.length < 2) return;
    const now = Date.now(), w = canvasEl.width, h = canvasEl.height;
    for (const t of trailPoints) {
        const age = (now - t.time) / 400;
        if (age > 1) continue;
        ctx.strokeStyle = t.handIdx === 0 ? `rgba(245,158,11,${(1-age)*0.3})` : `rgba(167,139,250,${(1-age)*0.3})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        t.pts.forEach((p, j) => { j === 0 ? ctx.moveTo(p.x * w, p.y * h) : ctx.lineTo(p.x * w, p.y * h); });
        ctx.stroke();
    }
}

/* ── UI Updates ── */
function updateGesture(name) {
    const el = document.getElementById('gesture-name'), cf = document.getElementById('gesture-conf'), ring = document.getElementById('gesture-ring'), sec = el.closest('.gesture-section');
    if (!name) { el.textContent = '--'; el.classList.remove('active','confident'); el.classList.add('idle'); cf.textContent = ''; ring.style.borderColor = 'transparent'; return; }
    const isNew = el.textContent !== name.replace(/_/g, ' ');
    el.textContent = name.replace(/_/g, ' ');
    el.classList.add('active'); el.classList.remove('idle');
    if (isNew) { el.classList.remove('confident'); void el.offsetWidth; el.classList.add('confident'); sec.classList.remove('flash'); void sec.offsetWidth; sec.classList.add('flash'); }
    cf.textContent = 'rule-based';
    ring.style.borderColor = 'rgba(245,158,11,0.15)';
}
function updateMLGesture(name, conf) {
    const el = document.getElementById('ml-gesture-name'), cf = document.getElementById('ml-gesture-conf'), row = document.getElementById('ml-row'), ring = document.getElementById('gesture-ring');
    if (!name) { el.textContent = '--'; cf.textContent = ''; row.classList.add('idle'); return; }
    el.textContent = name.replace(/_/g, ' ');
    const pct = conf != null ? Math.round(conf * 100) : 0;
    cf.textContent = pct ? pct + '%' : '';
    row.classList.remove('idle');
    if (conf != null) { const intensity = Math.min(conf, 1); ring.style.borderColor = `rgba(${Math.round(245*intensity+100*(1-intensity))},${Math.round(158*intensity+100*(1-intensity))},${Math.round(11*intensity+100*(1-intensity))},${0.1+intensity*0.25})`; }
}
function updateFingers(f) {
    const el = document.getElementById('fingers-display'), L = ['T','I','M','R','P'], N = ['Thumb','Index','Middle','Ring','Pinky'];
    if (!f || !f.length) { el.innerHTML = L.map((l,i) => `<div class="f-dot idle" title="${N[i]}">${l}</div>`).join(''); return; }
    el.innerHTML = f.map((v, i) => `<div class="f-dot ${v ? 'on' : ''}" title="${N[i]}: ${v ? 'extended' : 'retracted'}">${L[i]}</div>`).join('');
}
function updateFeatures(f) {
    if (!f) return;
    const open = document.getElementById('f-open'), pinch = document.getElementById('f-pinch'), span = document.getElementById('f-span'), vel = document.getElementById('f-vel');
    open.textContent = f.hand_openness != null ? f.hand_openness.toFixed(2) : '--';
    pinch.textContent = f.pinch_distance != null ? Math.round(f.pinch_distance) : '--';
    span.textContent = f.hand_span != null ? Math.round(f.hand_span) : '--';
    vel.textContent = '--';
    ['f-open','f-pinch','f-span','f-vel'].forEach(id => document.getElementById(id).classList.remove('idle'));
}
function updateHandedness(h) {
    const el = document.getElementById('handedness-badge');
    if (!h || h === 'Unknown') { el.textContent = '--'; el.className = 'hand-badge idle'; return; }
    el.textContent = h; el.className = 'hand-badge ' + h.toLowerCase();
}
function updateSecondHand(hand) {
    const el = document.getElementById('second-hand');
    if (!hand) { el.style.display = 'none'; return; }
    el.style.display = '';
    document.getElementById('handedness-badge-2').textContent = hand.handedness || '--';
    document.getElementById('gesture-name-2').textContent = hand.gesture ? hand.gesture.replace(/_/g, ' ') : '--';
}
function updateFace(face) {
    const section = document.getElementById('face-section');
    if (!face) { section.style.display = 'none'; return; }
    section.style.display = '';
    document.getElementById('face-expr').textContent = face.expression || 'neutral';
    document.getElementById('face-eyes').textContent = face.eyes_open ? 'open' : 'closed';
    document.getElementById('face-mouth').textContent = face.mouth_open ? 'open' : 'closed';
    document.getElementById('face-brow').textContent = face.eyebrows_raised ? 'raised' : 'neutral';
}

/* ── Recording ── */
const GESTURE_ICONS = {
    POINT:'\u261D', OPEN_HAND:'\u270B', FIST:'\u270A', PEACE_V:'\u270C', THUMBS_UP:'\u{1F44D}',
    PINCH:'\u{1F90F}', OK_SIGN:'\u{1F44C}', WAVE:'\u{1F44B}', CIRCLE:'\u{2B55}', GRAB:'\u{1F91F}',
    PUSH:'\u{1F446}', SWIPE_LEFT:'\u{2B05}', SWIPE_RIGHT:'\u{27A1}', THREE:'\u{1F446}', FOUR:'\u{1F44F}',
    PINKY:'\u{1F91E}', GUN:'\u{1F52B}', SPIDER:'\u{1F577}', ROCK:'\u{1F918}', SHAKA:'\u{1F44F}',
    LOVE:'\u{1F48C}', HANG_LOOSE:'\u{1F596}', FIVE:'\u270B', THREE_FINGERS:'\u2620',
};
function toggleRecord() {
    isRecording = !isRecording;
    const btn = document.getElementById('btn-record');
    if (isRecording) { btn.classList.add('btn-record'); btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop'; recordChunks = []; }
    else { btn.classList.remove('btn-record'); btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="currentColor"/></svg> Rec'; }
}
function takeSnapshot() {
    if (!videoEl || !localStream) return;
    const c = document.createElement('canvas'); c.width = videoEl.videoWidth; c.height = videoEl.videoHeight;
    c.getContext('2d').drawImage(videoEl, 0, 0);
    const a = document.createElement('a'); a.href = c.toDataURL('image/png'); a.download = 'capture_' + Date.now() + '.png'; a.click();
}
function selectRec(el, f) { document.querySelectorAll('.rec-card').forEach(e => e.classList.remove('selected')); el.classList.add('selected'); selectedRec = f; }
function playSelected() { if (!selectedRec || !socket || !socket.connected) return; socket.emit('play_recording', { filename: selectedRec, speed: 1.0 }); }
function stopPlayback() { if (socket) socket.emit('stop_playback'); }
async function loadRecordings() {
    try {
        const r = await fetch('/api/recordings'), d = await r.json(), el = document.getElementById('rec-list');
        document.getElementById('rec-count').textContent = d.count;
        if (!d.recordings.length) { el.innerHTML = '<div class="rec-empty idle">Start camera and record gestures to build your library</div>'; return; }
        el.innerHTML = d.recordings.map(r => {
            const icon = GESTURE_ICONS[r.gesture.toUpperCase()] || '\u{1F590}';
            return `<div class="rec-card" onclick="selectRec(this,'${r.filename}')" data-file="${r.filename}"><div class="rec-card-gesture">${icon} ${r.gesture}</div><div class="rec-card-meta">${r.frames}f \u00B7 ${r.duration.toFixed(1)}s</div></div>`;
        }).join('');
    } catch(e) {}
}

/* ── MediaPipe Hands Detection ── */
async function initMediaPipe() {
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.6
    });
    hands.onResults(onHandResults);
    mpHands = hands;
    return hands;
}

function onHandResults(results) {
    handResults = results;
    if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    drawTrail();
    updateParticles();

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (let hi = 0; hi < results.multiHandLandmarks.length; hi++) {
            const lm = results.multiHandLandmarks[hi];
            drawSkeleton(lm, hi);
            addTrailPoint(lm, hi);
        }

        const primary = results.multiHandLandmarks[0];
        const fingers = countFingersUp(primary);
        const gesture = classifyGesture(fingers, primary);
        const features = computeFeatures(primary);

        const prevGesture = lastDetectedGesture;
        updateGesture(gesture);
        trackGesture(gesture);
        if (gesture && gesture !== prevGesture) {
            spawnParticles(primary, '#F59E0B');
            playGestureSound(880, 0.08);
        }
        lastDetectedGesture = gesture;
        updateFingers(fingers);
        updateFeatures(features);
        updateHandedness(results.multiHandedness && results.multiHandedness[0] ? results.multiHandedness[0].label : 'Right');

        if (cursorMode) updateCursor(primary);

        if (results.multiHandLandmarks.length > 1) {
            updateSecondHand({ handedness: results.multiHandedness[1] ? results.multiHandedness[1].label : 'Left', gesture: classifyGesture(countFingersUp(results.multiHandLandmarks[1]), results.multiHandLandmarks[1]) });
        } else {
            updateSecondHand(null);
        }

        if (socket && socket.connected) {
            const flat = [];
            primary.forEach(lm => { flat.push(lm.x * 640, lm.y * 480); });
            socket.emit('process_frame', {
                landmarks: flat,
                fingers,
                gesture,
                confidence: 1.0,
                width: 640,
                height: 480,
                client_ts: Date.now()
            });
        }
    } else {
        clearOverlay();
        updateGesture(null);
        updateMLGesture(null);
        updateFingers([]);
        updateHandedness(null);
        updateSecondHand(null);
        lastDetectedGesture = null;
    }

    fpsCounter++;
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
        const fpsText = fpsCounter + ' FPS';
        document.getElementById('fps-display').textContent = fpsText;
        document.getElementById('cam-fps-overlay').textContent = fpsText;
        fpsCounter = 0; lastFpsTime = now;
    }
}

/* ── Camera Start/Stop ── */
async function startCamera() {
    const errEl = document.getElementById('cam-error');
    errEl.classList.remove('visible');
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } });
        videoEl = document.getElementById('local-video');
        videoEl.srcObject = localStream;
        await videoEl.play();

        canvasEl = document.getElementById('overlay-canvas');
        canvasEl.width = videoEl.videoWidth || 640;
        canvasEl.height = videoEl.videoHeight || 480;
        ctx = canvasEl.getContext('2d');

        document.getElementById('placeholder').style.display = 'none';
        document.getElementById('intro-banner').classList.add('hidden');
        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-stop').disabled = false;
        document.getElementById('btn-snapshot').disabled = false;
        document.getElementById('btn-record').disabled = false;
        document.getElementById('btn-cursor').disabled = false;
        document.getElementById('live-badge').style.display = '';
        document.getElementById('cam-fps-overlay').classList.add('visible');
        startSession();
        try { if (!sessionStorage.getItem('onboard_done')) startOnboarding(); } catch(e) {}

        document.getElementById('detection-loading').style.display = 'none';

        if (!mpHands) {
            document.getElementById('detection-loading').style.display = '';
            document.getElementById('detection-loading').textContent = 'Loading hand detection model...';
            try {
                await initMediaPipe();
                document.getElementById('detection-loading').style.display = 'none';
            } catch(e) {
                console.error('MediaPipe init failed:', e);
                document.getElementById('detection-loading').textContent = 'Detection model failed to load. Try refreshing.';
                document.getElementById('detection-loading').style.color = '#ef4444';
                return;
            }
        }

        mpCamera = new Camera(videoEl, {
            onFrame: async () => {
                if (mpHands) await mpHands.send({ image: videoEl });
            },
            width: 640,
            height: 480
        });
        mpCamera.start();
    } catch (e) {
        const msg = document.getElementById('cam-error-msg');
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') msg.textContent = 'Camera permission was denied. Allow access in your browser settings, then retry.';
        else if (e.name === 'NotFoundError') msg.textContent = 'No camera found. Connect a camera and try again.';
        else msg.textContent = 'Camera unavailable: ' + e.message;
        document.getElementById('placeholder').style.display = 'none';
        errEl.classList.add('visible');
    }
}

function retryCamera() { document.getElementById('cam-error').classList.remove('visible'); document.getElementById('placeholder').style.display = ''; startCamera(); }

function stopCamera() {
    if (mpCamera) { mpCamera.stop(); mpCamera = null; }
    if (localStream) localStream.getTracks().forEach(t => t.stop()); localStream = null;
    if (isRecording) toggleRecord();
    clearOverlay();
    particles.length = 0;
    trailPoints.length = 0;
    lastDetectedGesture = null;
    cursorMode = false;
    document.getElementById('cursor-indicator').style.display = 'none';
    document.getElementById('cursor-click-ring').style.display = 'none';
    document.getElementById('btn-cursor').classList.remove('on');
    document.getElementById('btn-cursor').innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="M13 13l6 6"/></svg> Cursor';
    document.getElementById('cam-error').classList.remove('visible');
    document.getElementById('placeholder').style.display = '';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-snapshot').disabled = true;
    document.getElementById('btn-record').disabled = true;
    document.getElementById('btn-cursor').disabled = true;
    document.getElementById('live-badge').style.display = 'none';
    document.getElementById('cam-fps-overlay').classList.remove('visible');
    document.getElementById('fps-display').textContent = '-- FPS';
    stopSession();
    onboardActive = false;
    document.getElementById('onboard-toast').classList.remove('visible');
    updateGesture(null); updateMLGesture(null); updateFingers([]); updateHandedness(null); updateSecondHand(null); updateFace(null);
    ['f-open','f-pinch','f-span','f-vel'].forEach(id => document.getElementById(id).classList.add('idle'));
}

window.addEventListener('load', () => { initSocket(); loadRecordings(); updateFingers([]); });
