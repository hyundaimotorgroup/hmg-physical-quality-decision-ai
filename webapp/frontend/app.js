const loginSection = document.getElementById('loginSection');
const appSection = document.getElementById('appSection');
const userSelect = document.getElementById('userSelect');
const loginBtn = document.getElementById('loginBtn');
const loginStatus = document.getElementById('loginStatus');
const userInfo = document.getElementById('userInfo');
const logoutLink = document.getElementById('logoutLink');

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileNameLabel = document.getElementById('fileNameLabel');
const pendingList = document.getElementById('pendingList');
const analyzeBtn = document.getElementById('analyzeBtn');

const uploadSection = document.getElementById('uploadSection');
const workspaceSection = document.getElementById('workspaceSection');
const queueNote = document.getElementById('queueNote');
const jobCards = document.getElementById('jobCards');

const resultPlaceholder = document.getElementById('resultPlaceholder');
const videoResult = document.getElementById('videoResult');
const imageResult = document.getElementById('imageResult');
const resultVideo = document.getElementById('resultVideo');
const dedupFrame = document.getElementById('dedupFrame');
const resultImage = document.getElementById('resultImage');
const imageOverall = document.getElementById('imageOverall');
const weldTableBody = document.getElementById('weldTableBody');

const PHASE_LABELS = {
  starting: 'Starting...',
  loading_models: 'Loading models...',
  processing: 'Judging frames...',
  encoding_video: 'Building result video / report...',
  done: 'Done',
};

// Visual pipeline steps shown as a stepper on each job card -- lets the
// user see WHICH stage is running, not just a generic spinner. Keys must
// line up with the backend's actual progress.phase values (main.py) plus
// a client-only "queued" lead-in step for videos waiting in line.
const VIDEO_STEPS = [
  { key: 'queued', label: 'Queued' },
  { key: 'starting', label: 'Prepare' },
  { key: 'loading_models', label: 'Load models' },
  { key: 'processing', label: 'Analyze frames' },
  { key: 'encoding_video', label: 'Build result' },
  { key: 'done', label: 'Done' },
];

// Image analysis is one synchronous request/response with no granular
// server-side progress, so this is a client-only best-effort split: switch
// to "analyzing" right after the request is sent (upload of one image is
// near-instant next to the detection+classification that actually takes
// the time), then "done"/error on response.
const IMAGE_STEPS = [
  { key: 'uploading', label: 'Upload' },
  { key: 'analyzing', label: 'Detect + classify' },
  { key: 'done', label: 'Done' },
];

function renderStepper(steps, currentKey, { error = false } = {}) {
  const currentIndex = steps.findIndex(s => s.key === currentKey);
  const isDone = currentKey === steps[steps.length - 1].key && !error;
  return '<div class="stepper">' + steps.map((s, i) => {
    let cls = '';
    if (error && i === currentIndex) cls = 'error';
    else if (isDone || i < currentIndex) cls = 'completed';
    else if (i === currentIndex) cls = 'active';
    const dotContent = cls === 'completed' ? '&#10003;' : (cls === 'error' ? '&#10005;' : '');
    const stepHtml = `<div class="stepper-step ${cls}"><div class="dot">${dotContent}</div><div class="label">${s.label}</div></div>`;
    if (i === steps.length - 1) return stepHtml;
    const connectorFilled = (isDone || i < currentIndex) ? 'filled' : '';
    return stepHtml + `<div class="stepper-connector ${connectorFilled}"></div>`;
  }).join('') + '</div>';
}

function formatEta(sec) {
  if (sec == null || !isFinite(sec)) return '';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `about ${m} min ${s} s left` : `about ${s} s left`;
}

// User-controlled strings (uploaded file names, user display names) get
// interpolated into HTML template strings below that are assigned via
// innerHTML -- without escaping, a file named e.g. `<img src=x
// onerror=alert()>.jpg` would execute as markup the moment its card
// renders (Stored XSS), not just when the raw filename is viewed.
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ── Auth ──────────────────────────────────────────────────────────────

async function init() {
  const meRes = await fetch('/api/me');
  if (meRes.ok) {
    const { user } = await meRes.json();
    showApp(user);
  } else {
    await showLogin();
  }
}

async function showLogin() {
  loginSection.style.display = 'block';
  appSection.style.display = 'none';
  const res = await fetch('/api/users');
  const { users } = await res.json();
  userSelect.innerHTML = users
    .map(u => `<option value="${escapeHtml(u.empNo)}">${escapeHtml(u.name)}${u.department ? ` (${escapeHtml(u.department)}${u.team ? ' · ' + escapeHtml(u.team) : ''})` : ''}</option>`)
    .join('');
}

loginBtn.addEventListener('click', async () => {
  const empNo = userSelect.value;
  if (!empNo) return;
  loginStatus.textContent = 'Signing in...';
  try {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ empNo }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { user } = await res.json();
    showApp(user);
  } catch (err) {
    loginStatus.textContent = `Sign-in failed: ${err}`;
  }
});

logoutLink.addEventListener('click', async (e) => {
  e.preventDefault();
  await fetch('/api/logout', { method: 'POST' });
  location.href = location.pathname;
});

function showApp(user) {
  loginSection.style.display = 'none';
  appSection.style.display = 'block';
  userInfo.innerHTML = `Signed in as <b>${escapeHtml(user.name)}</b>`;

  const jobId = new URLSearchParams(location.search).get('job_id');
  if (jobId) {
    loadExistingJob(jobId);
  }
}

async function loadExistingJob(jobId) {
  // Opened directly (e.g. the "View result" link opens in a new tab) -- there's
  // no batch/job-list context here, so show just the result, full width,
  // with the left job panel hidden rather than empty.
  uploadSection.style.display = 'none';
  workspaceSection.style.display = 'block';
  document.getElementById('jobsPanel').style.display = 'none';
  resultPlaceholder.style.display = 'none';

  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) {
    document.querySelector('.result-panel').insertAdjacentHTML('afterbegin', '<div class="summary">Record not found.</div>');
    return;
  }
  const { record, result } = await res.json();
  if (record.kind === 'video') {
    showVideoResult(result);
  } else {
    showImageResultFromRecord(result);
  }
}

// ── File selection (multi) ──────────────────────────────────────────────

let selectedFiles = [];

function setFiles(files) {
  selectedFiles = Array.from(files || []);
  if (!selectedFiles.length) {
    fileNameLabel.textContent = 'Click or drop video / image files here (mp4/mov, jpg/png, ...; multiple files allowed)';
    pendingList.style.display = 'none';
    pendingList.innerHTML = '';
    return;
  }
  fileNameLabel.textContent = `${selectedFiles.length} file(s) selected`;
  pendingList.style.display = 'block';
  pendingList.innerHTML = selectedFiles
    .map(f => `<div>${f.type.startsWith('image/') ? '[image]' : '[video]'} ${escapeHtml(f.name)} (${(f.size / 1e6).toFixed(1)}MB)</div>`)
    .join('');
}

fileInput.addEventListener('change', () => setFiles(fileInput.files));

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    setFiles(e.dataTransfer.files);
  }
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFiles.length) {
    alert('Please select a video or image file first.');
    return;
  }
  analyzeBtn.disabled = true;
  uploadSection.style.display = 'none';
  workspaceSection.style.display = 'block';

  const videoFiles = selectedFiles.filter(f => !f.type.startsWith('image/'));
  const imageFiles = selectedFiles.filter(f => f.type.startsWith('image/'));
  queueNote.style.display = videoFiles.length > 1 ? 'block' : 'none';

  jobCards.innerHTML = '';
  const cardStates = new Map(); // localId -> { file, kind, cardEl }

  for (const file of selectedFiles) {
    const localId = `local_${cardStates.size}_${Date.now()}`;
    const kind = file.type.startsWith('image/') ? 'image' : 'video';
    const cardEl = document.createElement('div');
    cardEl.className = 'history-item';
    const initialStepKey = kind === 'image' ? 'uploading' : 'queued';
    cardEl.innerHTML = renderCardBody({ filename: file.name, kind, stepKey: initialStepKey, statusLabel: 'Queued...', pct: null });
    jobCards.appendChild(cardEl);
    cardStates.set(localId, { file, kind, cardEl });
  }

  // Videos go through the server's single-worker queue anyway, but firing
  // all uploads at once still means N simultaneous upload POSTs competing
  // for bandwidth/disk for no benefit -- upload+kick off one at a time,
  // same as the server-side processing order.
  for (const [, state] of cardStates) {
    if (state.kind === 'video') {
      await runVideoJob(state);
    } else {
      await runImageJob(state);
    }
  }

  analyzeBtn.disabled = false;
});

function renderCardBody({ filename, kind, stepKey, error, statusLabel, pct, meta, liveSrc, done }) {
  const kindLabel = kind === 'image' ? 'Image' : 'Video';
  const steps = kind === 'image' ? IMAGE_STEPS : VIDEO_STEPS;
  const stepperHtml = renderStepper(steps, stepKey, { error: !!error });
  // At 100% stop the progress bar's move/shimmer animation (.done); the stepper above
  // already shows "Done", so the numeric % would be redundant and is omitted.
  let progressHtml;
  if (done) {
    progressHtml = `<div class="progress-track"><div class="progress-fill done"></div></div>`;
  } else if (pct == null) {
    progressHtml = `<div class="progress-track"><div class="progress-fill indeterminate"></div></div>`;
  } else {
    progressHtml = `<div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div><div class="progress-pct">${pct}%</div>`;
  }
  const liveHtml = liveSrc
    ? `<div class="live-preview-wrap"><img src="${liveSrc}"><div class="live-preview-caption">Live frame preview (raw per-frame verdict - before final aggregation)</div></div>`
    : '';
  return `
    <div class="hi-top">
      <span class="hi-filename">[${kindLabel}] ${escapeHtml(filename)}</span>
    </div>
    ${stepperHtml}
    <div class="status-line" style="margin:10px 0 4px">${escapeHtml(statusLabel)}</div>
    ${progressHtml}
    ${meta ? `<div class="hi-exports">${meta}</div>` : ''}
    ${liveHtml}
    <div class="hi-links" id="cardLinks"></div>
  `;
}

function updateCard(state, opts) {
  state.cardEl.innerHTML = renderCardBody({ filename: state.file.name, kind: state.kind, ...opts });
}

function addResultLink(state, jobId) {
  const linksEl = state.cardEl.querySelector('#cardLinks');
  if (linksEl) linksEl.innerHTML = `<a class="view-result-btn" href="?job_id=${jobId}" target="_blank">View result &rarr;</a>`;
}

// ── Video job (upload -> poll -> result) ────────────────────────────────

async function runVideoJob(state) {
  updateCard(state, { stepKey: 'queued', statusLabel: 'Uploading...', pct: null });

  const formData = new FormData();
  formData.append('video', state.file);

  let jobId;
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(await res.text());
    jobId = (await res.json()).job_id;
  } catch (err) {
    updateCard(state, { stepKey: 'queued', error: true, statusLabel: `Upload failed: ${err}`, pct: null });
    return;
  }

  await pollVideo(state, jobId);
}

function pollVideo(state, jobId) {
  return new Promise((resolve) => {
    const timer = setInterval(async () => {
      let data;
      try {
        const res = await fetch(`/api/status/${jobId}`);
        data = await res.json();
      } catch (err) {
        return; // transient network error, keep polling
      }

      if (data.status === 'queued') {
        const pos = data.queue_position;
        updateCard(state, { stepKey: 'queued', statusLabel: pos ? `Queued (${pos - 1} ahead)` : 'Queued...', pct: null });
      } else if (data.status === 'running') {
        const p = data.progress;
        const label = PHASE_LABELS[p.phase] || p.phase;
        let pct = null;
        let meta = '';
        if (p.phase === 'processing' && p.total) {
          pct = Math.min(100, Math.round((p.current / p.total) * 100));
          meta = `${p.current} / ${p.total} frames${p.rate_fps ? ` · ${p.rate_fps.toFixed(2)} frame/s` : ''} · ${formatEta(p.eta_sec)}`;
        } else if (p.phase === 'encoding_video') {
          pct = 100;
        }
        const liveSrc = p.live_preview_url ? `${p.live_preview_url}?frame=${p.current || 0}` : null;
        updateCard(state, { stepKey: p.phase, statusLabel: label, pct, meta, liveSrc });
      } else if (data.status === 'done') {
        clearInterval(timer);
        updateCard(state, { stepKey: 'done', statusLabel: 'Done', done: true });
        addResultLink(state, jobId);
        resolve();
      } else if (data.status === 'error') {
        clearInterval(timer);
        updateCard(state, { stepKey: data.progress.phase, error: true, statusLabel: `Analysis failed: ${data.error}`, pct: null });
        resolve();
      }
    }, 2000);
  });
}

function showVideoResult(result) {
  if (result.overlay_video) resultVideo.src = result.overlay_video;
  if (result.dedup_audit) dedupFrame.src = result.dedup_audit;
  resultPlaceholder.style.display = 'none';
  videoResult.style.display = 'block';
  imageResult.style.display = 'none';
}

// ── Image job (upload -> synchronous result) ────────────────────────────

async function runImageJob(state) {
  updateCard(state, { stepKey: 'uploading', statusLabel: 'Uploading...', pct: null });

  const formData = new FormData();
  formData.append('image', state.file);

  try {
    updateCard(state, { stepKey: 'analyzing', statusLabel: 'Analyzing image... (takes a few seconds)', pct: null });
    const res = await fetch('/api/analyze-image', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const badClass = data.overall_class === 'BAD' ? 'bad' : 'good';
    updateCard(state, {
      stepKey: 'done',
      statusLabel: 'Done',
      done: true,
      meta: `Overall verdict: <span class="badge ${badClass}">${data.overall_class}</span> · ${data.objects.length} object(s)`,
    });
    addResultLink(state, data.job_id);
  } catch (err) {
    updateCard(state, { stepKey: 'analyzing', error: true, statusLabel: `Analysis failed: ${err}`, pct: null });
  }
}

function renderWeldTable(welds) {
  weldTableBody.innerHTML = welds.map((w, i) => {
    const badgeClass = w.class_name === 'BAD' ? 'bad' : 'good';
    return `<tr><td>${i + 1}</td><td><span class="badge ${badgeClass}">${w.class_name}</span></td>` +
      `<td>${(w.confidence * 100).toFixed(1)}%</td><td>${w.used_retrieval ? 'Reference-bank check' : ''}</td></tr>`;
  }).join('');
}

function showImageResultFromRecord(result) {
  resultImage.src = result.overlay_image;
  imageOverall.innerHTML = '';
  weldTableBody.innerHTML = '';
  resultPlaceholder.style.display = 'none';
  videoResult.style.display = 'none';
  imageResult.style.display = 'block';
}

init();
