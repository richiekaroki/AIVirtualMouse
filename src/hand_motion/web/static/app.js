const VIDEO_FPS = 10, JPEG_QUALITY = 0.5, FRAME_INTERVAL = 1000 / VIDEO_FPS;
let socket, localStream, videoEl, canvasEl, ctx, sendTimer, selectedRec = null;
let frameCount = 0, lastFpsTime = Date.now(), fpsCounter = 0;
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
const SMOOTHING = 0.35;
const CLICK_COOLDOWN = 400;

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
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + dur * 0.3);
    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + dur);
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

function updateCursor(landmarks) {
    if (!cursorMode || !landmarks || landmarks.length < 42) return;
    const indexTipX = landmarks[8 * 2];
    const indexTipY = landmarks[8 * 2 + 1];
    const screenW = window.innerWidth;
    const screenH = window.innerHeight;
    const targetX = (1 - indexTipX / 640) * screenW;
    const targetY = (indexTipY / 480) * screenH;
    cursorSmoothX += (targetX - cursorSmoothX) * SMOOTHING;
    cursorSmoothY += (targetY - cursorSmoothY) * SMOOTHING;
    cursorX = cursorSmoothX;
    cursorY = cursorSmoothY;
    moveCursorIndicator(cursorX, cursorY);
    checkClickGesture(landmarks);
}

function moveCursorIndicator(x, y) {
    const el = document.getElementById('cursor-indicator');
    if (!el) return;
    el.style.left = x + 'px';
    el.style.top = y + 'px';
}

function checkClickGesture(lm) {
    const indexTip = [lm[8 * 2], lm[8 * 2 + 1]];
    const middleTip = [lm[12 * 2], lm[12 * 2 + 1]];
    const dist = Math.hypot(indexTip[0] - middleTip[0], indexTip[1] - middleTip[1]);
    const now = Date.now();
    if (dist < 30 && !cursorClicking && now - lastClickTime > CLICK_COOLDOWN) {
        cursorClicking = true;
        lastClickTime = now;
        fireClickAt(cursorX, cursorY);
    } else if (dist > 50) {
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
const ONBOARD_STEPS = [
    { gesture: 'OPEN_HAND', emoji: '\u{1F590}\uFE0F', name: 'Open Hand', hint: 'Spread all five fingers and hold your hand open' },
    { gesture: 'FIST', emoji: '\u270A', name: 'Fist', hint: 'Close all five fingers into a tight fist' },
    { gesture: 'POINT', emoji: '\u261D\uFE0F', name: 'Point', hint: 'Extend only your index finger' },
];

function initSocket() {
    socket = io({ transports: ['polling', 'websocket'] });
    socket.on('connect', () => { setWsStatus(true); loadRecordings(); });
    socket.on('disconnect', () => setWsStatus(false));
    socket.on('reconnect_attempt', () => setWsConnecting());
    socket.on('reconnect_failed', () => setWsStatus(false));

    socket.on('frame_result', d => {
        if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        drawTrail();
        updateParticles();
        if (d.error) {
            console.warn('Server error:', d.error);
        }
        if (d.hands && d.hands.length > 0) {
            // Multi-hand rendering
            d.hands.forEach(hand => {
                if (hand.landmarks && hand.landmarks.length >= 42) {
                    drawSkeleton(hand.landmarks, hand.hand_index);
                    addTrailPoint(hand.landmarks, hand.hand_index);
                }
            });
            // Primary hand displays in main UI
            const primary = d.hands[0];
            const prevGesture = lastDetectedGesture;
            updateGesture(primary.gesture, primary.confidence);
            trackGesture(primary.gesture);
            if (primary.gesture && primary.gesture !== prevGesture && primary.landmarks && primary.landmarks.length >= 42) {
                spawnParticles(primary.landmarks, '#F59E0B');
                playGestureSound(880, 0.08);
            }
            lastDetectedGesture = primary.gesture;
            updateMLGesture(primary.ml_gesture, primary.ml_confidence);
            updateFingers(primary.fingers);
            updateFeatures(primary.features, primary.velocity);
            updateHandedness(primary.handedness);
            // Virtual mouse cursor control
            if (cursorMode && primary.landmarks) {
                updateCursor(primary.landmarks);
            }
            // Second hand display
            updateSecondHand(d.hands.length > 1 ? d.hands[1] : null);
        } else if (d.landmarks && d.landmarks.length >= 42) {
            // Fallback: single-hand format (playback or backward-compat)
            drawSkeleton(d.landmarks);
            addTrailPoint(d.landmarks, 0);
            const prevGesture = lastDetectedGesture;
            updateGesture(d.gesture, d.confidence);
            trackGesture(d.gesture);
            if (d.gesture && d.gesture !== prevGesture) {
                spawnParticles(d.landmarks, '#F59E0B');
                playGestureSound(880, 0.08);
            }
            lastDetectedGesture = d.gesture;
            updateMLGesture(d.ml_gesture, d.ml_confidence);
            updateFingers(d.fingers);
            updateFeatures(d.features, d.velocity);
            updateHandedness(d.handedness);
            updateSecondHand(null);
        } else {
            clearOverlay(); updateGesture(null); updateMLGesture(null); updateFingers([]); updateHandedness(null); updateSecondHand(null);
            lastDetectedGesture = null;
        }
        // Non-manual markers (face)
        updateFace(d.face);
        fpsCounter++;
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
            const fpsText = fpsCounter + ' FPS';
            document.getElementById('fps-display').textContent = fpsText;
            document.getElementById('cam-fps-overlay').textContent = fpsText;
            fpsCounter = 0; lastFpsTime = now;
        }
    });

    socket.on('playback_frame', d => {
        if (d.landmarks && d.landmarks.length >= 42) { drawSkeleton(d.landmarks); updateGesture(d.primitive); updateMLGesture(null); updateFingers(d.finger_states); if (d.features) updateFeatures(d.features, d.velocity); updateHandedness(d.handedness); }
    });
    socket.on('playback_finished', () => { clearOverlay(); trailPoints.length = 0; particles.length = 0; updateGesture(null); updateHandedness(null); });
}

function setWsStatus(on) {
    const state = on ? 'on' : 'off';
    document.getElementById('ws-badge').className = 'ws-pill ' + state;
    document.getElementById('ws-dot').className = 'ws-dot ' + state;
    document.getElementById('ws-text').textContent = on ? 'Live' : 'Offline';
}

function setWsConnecting() {
    document.getElementById('ws-badge').className = 'ws-pill connecting';
    document.getElementById('ws-dot').className = 'ws-dot connecting';
    document.getElementById('ws-text').textContent = 'Connecting...';
}

function startSession() {
    sessionStart = Date.now();
    uniqueGestureCount = 0;
    document.getElementById('stat-gestures').textContent = '0';
    document.getElementById('stat-uptime').textContent = '0:00';
    if (sessionTimer) clearInterval(sessionTimer);
    sessionTimer = setInterval(updateSessionUptime, 1000);
}

function stopSession() {
    if (sessionTimer) clearInterval(sessionTimer);
    sessionTimer = null;
    sessionStart = 0;
}

function updateSessionUptime() {
    if (!sessionStart) return;
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    const m = Math.floor(s / 60), sec = s % 60;
    document.getElementById('stat-uptime').textContent = m + ':' + String(sec).padStart(2, '0');
}

function trackGesture(name) {
    if (!name) return;
    if (!uniqueGestures.has(name)) { uniqueGestures.add(name); uniqueGestureCount++; document.getElementById('stat-gestures').textContent = uniqueGestureCount; }
    if (onboardActive && onboardStep >= 0 && name === ONBOARD_STEPS[onboardStep].gesture) { advanceOnboarding(); }
}

function startOnboarding() {
    onboardStep = 0;
    onboardActive = true;
    showOnboardStep(0);
    document.getElementById('onboard-toast').classList.add('visible');
}

function showOnboardStep(i) {
    const s = ONBOARD_STEPS[i];
    document.getElementById('onboard-step').textContent = 'Step ' + (i+1) + ' of ' + ONBOARD_STEPS.length;
    document.getElementById('onboard-gesture').textContent = s.emoji;
    document.getElementById('onboard-name').textContent = s.name;
    document.getElementById('onboard-hint').textContent = s.hint;
    for (let j = 0; j < ONBOARD_STEPS.length; j++) {
        const dot = document.getElementById('ob-dot-' + j);
        dot.className = 'onboard-dot' + (j < i ? ' done' : j === i ? ' active' : '');
    }
}

function advanceOnboarding() {
    onboardStep++;
    if (onboardStep >= ONBOARD_STEPS.length) { finishOnboarding(); return; }
    showOnboardStep(onboardStep);
}

function finishOnboarding() {
    onboardActive = false;
    document.getElementById('onboard-toast').classList.remove('visible');
    try { sessionStorage.setItem('onboard_done', '1'); } catch(e) {}
}

function skipOnboarding() { finishOnboarding(); }

async function startCamera() {
    const errEl = document.getElementById('cam-error');
    errEl.classList.remove('visible');
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } });
        videoEl = document.getElementById('local-video');
        videoEl.srcObject = localStream; await videoEl.play();
        canvasEl = document.getElementById('overlay-canvas');
        canvasEl.width = videoEl.videoWidth || 640; canvasEl.height = videoEl.videoHeight || 480;
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
        sendTimer = setInterval(captureAndSend, FRAME_INTERVAL);
    } catch (e) {
        const msg = document.getElementById('cam-error-msg');
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
            msg.textContent = 'Camera permission was denied. Allow access in your browser settings, then retry.';
        } else if (e.name === 'NotFoundError') {
            msg.textContent = 'No camera found. Connect a camera and try again.';
        } else {
            msg.textContent = 'Camera unavailable: ' + e.message;
        }
        document.getElementById('placeholder').style.display = 'none';
        errEl.classList.add('visible');
    }
}

function retryCamera() {
    document.getElementById('cam-error').classList.remove('visible');
    document.getElementById('placeholder').style.display = '';
    startCamera();
}

function stopCamera() {
    if (sendTimer) clearInterval(sendTimer); sendTimer = null;
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
    document.getElementById('live-badge').style.display = 'none';
    document.getElementById('cam-fps-overlay').classList.remove('visible');
    document.getElementById('fps-display').textContent = '-- FPS';
    stopSession();
    onboardActive = false;
    document.getElementById('onboard-toast').classList.remove('visible');
    updateGesture(null);
    updateMLGesture(null);
    updateFingers([]);
    updateHandedness(null);
    updateSecondHand(null);
    updateFace(null);
    document.getElementById('f-open').classList.add('idle');
    document.getElementById('f-pinch').classList.add('idle');
    document.getElementById('f-span').classList.add('idle');
    document.getElementById('f-vel').classList.add('idle');
}

function captureAndSend() {
    if (!videoEl || !localStream || !socket || !socket.connected || videoEl.readyState < 2) return;
    const w = 480, h = 360;
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    c.getContext('2d').drawImage(videoEl, 0, 0, w, h);
    frameCount++;
    socket.emit('process_frame', { image: c.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1], width: w, height: h, client_ts: Date.now(), frame_num: frameCount });
}

function drawSkeleton(lm, handIdx = 0) {
    if (!ctx || !canvasEl) return;
    const sx = canvasEl.width / 640, sy = canvasEl.height / 480;
    const conns = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
    const colors = [
        { line: 'rgba(245,158,11,0.6)', tip: '#F59E0B', joint: '#E8E8EC' },
        { line: 'rgba(167,139,250,0.6)', tip: '#A78BFA', joint: '#C4B5FD' },
    ];
    const c = colors[handIdx % colors.length];
    ctx.strokeStyle = c.line; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    for (const [a, b] of conns) { ctx.beginPath(); ctx.moveTo(lm[a*2]*sx, lm[a*2+1]*sy); ctx.lineTo(lm[b*2]*sx, lm[b*2+1]*sy); ctx.stroke(); }
    for (let i = 0; i < 21; i++) {
        const x = lm[i*2]*sx, y = lm[i*2+1]*sy, tip = [4,8,12,16,20].includes(i);
        ctx.fillStyle = tip ? c.tip : c.joint;
        ctx.beginPath(); ctx.arc(x, y, tip ? 4 : 2.5, 0, Math.PI*2); ctx.fill();
    }
}

function clearOverlay() { if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height); }

/* ── Particle System ── */
const particles = [];
const PARTICLE_TIPS = [4, 8, 12, 16, 20];

function spawnParticles(lm, color) {
    if (!ctx || !canvasEl || !lm || lm.length < 42) return;
    const sx = canvasEl.width / 640, sy = canvasEl.height / 480;
    for (const i of PARTICLE_TIPS) {
        const x = lm[i*2]*sx, y = lm[i*2+1]*sy;
        for (let j = 0; j < 4; j++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 1.5 + Math.random() * 3;
            particles.push({
                x, y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 1,
                decay: 0.015 + Math.random() * 0.02,
                size: 2 + Math.random() * 2.5,
                color: color || '#F59E0B'
            });
        }
    }
}

function updateParticles() {
    if (!ctx || !particles.length) return;
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx; p.y += p.vy;
        p.vy += 0.05;
        p.life -= p.decay;
        if (p.life <= 0) { particles.splice(i, 1); continue; }
        ctx.globalAlpha = p.life;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

/* ── Gesture Trail ── */
const trailPoints = [];
const TRAIL_MAX = 20;

function addTrailPoint(lm, handIdx) {
    if (!lm || lm.length < 42) return;
    const sx = canvasEl.width / 640, sy = canvasEl.height / 480;
    const tips = [4, 8, 12, 16, 20];
    const pts = tips.map(i => ({ x: lm[i*2]*sx, y: lm[i*2+1]*sy }));
    trailPoints.push({ pts, time: Date.now(), handIdx });
    if (trailPoints.length > TRAIL_MAX) trailPoints.shift();
}

function drawTrail() {
    if (!ctx || trailPoints.length < 2) return;
    const now = Date.now();
    for (let i = 0; i < trailPoints.length; i++) {
        const t = trailPoints[i];
        const age = (now - t.time) / 400;
        if (age > 1) continue;
        const alpha = (1 - age) * 0.3;
        const color = t.handIdx === 0 ? `rgba(245,158,11,${alpha})` : `rgba(167,139,250,${alpha})`;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let j = 0; j < t.pts.length; j++) {
            if (j === 0) ctx.moveTo(t.pts[j].x, t.pts[j].y);
            else ctx.lineTo(t.pts[j].x, t.pts[j].y);
        }
        ctx.stroke();
    }
}

function updateGesture(name) {
    const el = document.getElementById('gesture-name'), cf = document.getElementById('gesture-conf'), ring = document.getElementById('gesture-ring'), sec = el.closest('.gesture-section');
    if (!name) { el.textContent = '--'; el.classList.remove('active','confident'); el.classList.add('idle'); cf.textContent = ''; ring.style.borderColor = 'transparent'; return; }
    const isNew = el.textContent !== name.replace(/_/g, ' ');
    el.textContent = name.replace(/_/g, ' ');
    el.classList.add('active'); el.classList.remove('idle');
    if (isNew) {
        el.classList.remove('confident'); void el.offsetWidth; el.classList.add('confident');
        sec.classList.remove('flash'); void sec.offsetWidth; sec.classList.add('flash');
    }
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
    if (conf != null) {
        const intensity = Math.min(conf, 1);
        const r = Math.round(245 * intensity + 100 * (1 - intensity));
        const g = Math.round(158 * intensity + 100 * (1 - intensity));
        const b = Math.round(11 * intensity + 100 * (1 - intensity));
        ring.style.borderColor = `rgba(${r},${g},${b},${0.1 + intensity * 0.25})`;
    }
}

function updateFingers(f) {
    const el = document.getElementById('fingers-display'), L = ['T','I','M','R','P'], N = ['Thumb','Index','Middle','Ring','Pinky'];
    if (!f || !f.length) { el.innerHTML = L.map((l,i) => `<div class="f-dot idle" title="${N[i]}">${l}</div>`).join(''); return; }
    el.innerHTML = f.map((v, i) => `<div class="f-dot ${v ? 'on' : ''}" title="${N[i]}: ${v ? 'extended' : 'retracted'}">${L[i]}</div>`).join('');
}

function updateFeatures(f, v) {
    if (!f) return;
    const open = document.getElementById('f-open'), pinch = document.getElementById('f-pinch'), span = document.getElementById('f-span'), vel = document.getElementById('f-vel');
    open.textContent = f.hand_openness != null ? f.hand_openness.toFixed(2) : '--';
    pinch.textContent = f.pinch_distance != null ? Math.round(f.pinch_distance) : '--';
    span.textContent = f.hand_span != null ? Math.round(f.hand_span) : '--';
    if (v) vel.textContent = v.magnitude != null ? Math.round(v.magnitude) : '--';
    open.classList.remove('idle'); pinch.classList.remove('idle'); span.classList.remove('idle'); vel.classList.remove('idle');
}

function updateHandedness(h) {
    const el = document.getElementById('handedness-badge');
    if (!h || h === 'Unknown') { el.textContent = '--'; el.className = 'hand-badge idle'; return; }
    el.textContent = h;
    el.className = 'hand-badge ' + h.toLowerCase();
}

function updateSecondHand(hand) {
    const el = document.getElementById('second-hand');
    if (!hand) { el.style.display = 'none'; return; }
    el.style.display = '';
    const badge = document.getElementById('handedness-badge-2');
    const name = document.getElementById('gesture-name-2');
    const conf = document.getElementById('ml-gesture-conf-2');
    if (hand.handedness && hand.handedness !== 'Unknown') {
        badge.textContent = hand.handedness;
        badge.className = 'hand-badge ' + hand.handedness.toLowerCase();
    } else { badge.textContent = '--'; badge.className = 'hand-badge'; }
    name.textContent = hand.ml_gesture ? hand.ml_gesture.replace(/_/g, ' ') : (hand.gesture ? hand.gesture.replace(/_/g, ' ') : '--');
    conf.textContent = hand.ml_confidence ? Math.round(hand.ml_confidence * 100) + '%' : '';
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

function toggleRecord() {
    isRecording = !isRecording;
    const btn = document.getElementById('btn-record');
    if (isRecording) { btn.classList.add('btn-record'); btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop'; }
    else { btn.classList.remove('btn-record'); btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="currentColor"/></svg> Rec'; }
}

function takeSnapshot() {
    if (!videoEl || !localStream) return;
    const c = document.createElement('canvas'); c.width = videoEl.videoWidth; c.height = videoEl.videoHeight;
    c.getContext('2d').drawImage(videoEl, 0, 0);
    const a = document.createElement('a'); a.href = c.toDataURL('image/png'); a.download = 'capture_' + Date.now() + '.png'; a.click();
}

const GESTURE_ICONS = {
    POINT:'\u261D', OPEN_HAND:'\u270B', FIST:'\u270A', PEACE_V:'\u270C', THUMBS_UP:'\u{1F44D}',
    PINCH:'\u{1F90F}', OK_SIGN:'\u{1F44C}', WAVE:'\u{1F44B}', CIRCLE:'\u{2B55}', GRAB:'\u{1F91F}',
    PUSH:'\u{1F446}', SWIPE_LEFT:'\u{2B05}', SWIPE_RIGHT:'\u{27A1}', THREE:'\u{1F446}', FOUR:'\u{1F44F}',
    PINKY:'\u{1F91E}', GUN:'\u{1F52B}', SPIDER:'\u{1F577}', ROCK:'\u{1F918}', SHAKA:'\u{1F44F}',
};

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

window.addEventListener('load', () => { initSocket(); loadRecordings(); updateFingers([]); });
