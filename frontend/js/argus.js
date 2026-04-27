/**
 * Argus Platform — Frontend JavaScript
 * ======================================
 * Sections:
 *   1. Config & Utilities
 *   2. Navigation (tabs + mobile drawer + scroll reveal)
 *   3. API Health Check
 *   4. Risk Scorer
 *   5. Policy Assistant (RAG Chat)
 *   6. Claims Agent
 *   7. Research Report (sidebar scroll tracking)
 */

/* ============================================================
   1. CONFIG & UTILITIES
   ============================================================ */

/** API base URL — localhost in dev, relative path in production */
const API = ['localhost', '127.0.0.1'].includes(location.hostname)
  ? 'http://localhost:8000'
  : 'https://Ramesh79-argus.hf.space';

/** Shorthand element selector by ID */
const $ = id => document.getElementById(id);

/** Escape HTML to prevent XSS */
const esc = s =>
  String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

/** Loader HTML snippet */
const loaderHtml = () => `
  <div class="loading-row">
    <div class="loader"><span></span><span></span><span></span></div>
    <span>Processing…</span>
  </div>`;

/** Error box HTML snippet */
const errHtml = msg => `<div class="error-box">⚠ ${esc(msg)}</div>`;

/** Map risk label to CSS color variable */
const riskColor = label =>
  ({ LOW: 'var(--teal)', MEDIUM: 'var(--gold)', HIGH: 'var(--red)', CRITICAL: 'var(--red)' }[label] || 'var(--tx)');


/* ============================================================
   2. NAVIGATION
   ============================================================ */

/**
 * Switch the visible page and sync all nav/drawer tab states.
 * @param {string}  pageId  - key: 'overview' | 'scorer' | 'policy' | 'agent' | 'report' | 'results'
 * @param {boolean} isMobile - whether called from the mobile drawer
 */
function goTab(pageId, isMobile = false) {
  // Hide all pages, deactivate all tabs
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab, .drawer-tab').forEach(t => t.classList.remove('active'));

  // Show the target page
  $('p-' + pageId)?.classList.add('active');

  // Activate matching desktop nav tab
  document.querySelector(`.nav-tab[data-page="${pageId}"]`)?.classList.add('active');
  // Activate matching drawer tab
  document.querySelector(`.drawer-tab[data-page="${pageId}"]`)?.classList.add('active');

  if (isMobile) closeDrawer();

  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Trigger scroll-reveal for newly visible page
  setTimeout(() => observeReveal('p-' + pageId), 80);
}

/** Toggle the mobile hamburger drawer */
function toggleDrawer() {
  const drawer = $('nav-drawer');
  const btn    = $('hamburger-btn');
  const open   = drawer.classList.toggle('open');
  btn.classList.toggle('open', open);
  document.body.style.overflow = open ? 'hidden' : '';
}

function closeDrawer() {
  $('nav-drawer')?.classList.remove('open');
  $('hamburger-btn')?.classList.remove('open');
  document.body.style.overflow = '';
}

// Close drawer on outside click
document.addEventListener('click', e => {
  const drawer = $('nav-drawer');
  if (
    drawer?.classList.contains('open') &&
    !drawer.contains(e.target) &&
    !$('hamburger-btn')?.contains(e.target)
  ) {
    closeDrawer();
  }
});

/* ── Scroll Reveal ── */
const revealObserver = new IntersectionObserver(
  entries => entries.forEach(e => {
    if (e.isIntersecting) e.target.classList.add('visible');
  }),
  { threshold: 0.07 }
);

function observeReveal(containerId) {
  document.querySelectorAll(`#${containerId} .reveal`).forEach(el =>
    revealObserver.observe(el)
  );
}

// Observe overview page on load
observeReveal('p-overview');


/* ============================================================
   3. API HEALTH CHECK
   ============================================================ */
async function checkHealth() {
  try {
    const res = await fetch(API + '/api/health', {
      signal: AbortSignal.timeout(5000),
    });
    // We don't show status in header per design requirements
    // Just keep checking so it's available for diagnostics
    if (!res.ok) console.warn('API health check failed:', res.status);
  } catch (err) {
    console.warn('API offline:', err.message);
  }
}

checkHealth();
setInterval(checkHealth, 30_000);


/* ============================================================
   4. RISK SCORER
   ============================================================ */

let addrMatchVal = false;

/** Toggle the address-match toggle button */
function toggleAddr() {
  addrMatchVal = !addrMatchVal;
  $('f-addr').classList.toggle('on', addrMatchVal);
}

/** Sample claim presets */
const PRESETS = {
  high: { amt: 12800, card: 'prepaid', dev: 'mobile',  hr: 3,  vel: 8, age: 22,   em: 0.91, dist: 1240, pcl: 2, addr: false },
  low:  { amt:   620, card: 'credit',  dev: 'desktop', hr: 14, vel: 1, age: 1580, em: 0.08, dist:    3, pcl: 0, addr: true  },
};

/** Load a preset into the scorer form */
function loadSample(type) {
  const s = PRESETS[type];
  $('f-amt').value  = s.amt;
  $('f-card').value = s.card;
  $('f-dev').value  = s.dev;
  $('f-hr').value   = s.hr;  $('f-hrv').textContent  = s.hr + ':00';
  $('f-vel').value  = s.vel;
  $('f-age').value  = s.age;
  $('f-em').value   = s.em;  $('f-emv').textContent  = s.em.toFixed(2);
  $('f-dist').value = s.dist;
  $('f-pcl').value  = s.pcl;
  addrMatchVal = s.addr;
  $('f-addr').classList.toggle('on', addrMatchVal);
}

/** Submit the scorer form to the API */
async function scoreIt() {
  const btn = $('sc-btn');
  btn.disabled = true;
  btn.textContent = 'Scoring…';
  $('sc-out').innerHTML = loaderHtml();

  const payload = {
    transaction_amt:       parseFloat($('f-amt').value)  || 100,
    card_type:             $('f-card').value,
    device_type:           $('f-dev').value,
    hour_of_day:           parseInt($('f-hr').value),
    transaction_velocity:  parseFloat($('f-vel').value)  || 0,
    account_age_days:      parseInt($('f-age').value)    || 0,
    address_match:         addrMatchVal,
    email_risk_score:      parseFloat($('f-em').value),
    distance_from_home_km: parseFloat($('f-dist').value) || 0,
    prior_claims_count:    parseInt($('f-pcl').value)    || 0,
  };

  try {
    const res = await fetch(API + '/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    $('sc-claim-id').textContent = data.claim_id;
    renderScore(data);
  } catch (err) {
    $('sc-out').innerHTML = errHtml(err.message || 'Connection failed — is the API running?');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Score Claim';
  }
}

/**
 * Render the risk gauge and SHAP attribution chart.
 * @param {Object} data - /api/score response
 */
function renderScore(data) {
  const pct   = Math.round(data.fraud_probability * 100);
  const color = riskColor(data.risk_label);

  const shapRows = data.shap_features.map(f => {
    const barW = Math.min(Math.abs(f.shap_value) * 200, 48);
    const isPos = f.direction === 'increases_risk';
    return `
      <div class="shap-row">
        <div class="shap-feat">${esc(f.feature)}</div>
        <div class="shap-bars">
          <div class="shap-zero"></div>
          <div class="shap-bar ${isPos ? 'pos' : 'neg'}" style="width:${barW}px"></div>
        </div>
        <div class="shap-val ${isPos ? 'pos' : 'neg'}">${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(3)}</div>
      </div>`;
  }).join('');

  $('sc-out').innerHTML = `
    <div class="gauge-center">
      <div class="gauge-score" style="color:${color}">${data.fraud_probability.toFixed(3)}</div>
      <div class="gauge-ends"><span>Low risk</span><span>High risk</span></div>
      <div class="gauge-track"><div class="gauge-fill" style="width:${pct}%"></div></div>
      <div class="risk-pill risk-${data.risk_label}">${data.risk_label} RISK · ${pct}/100</div>
    </div>

    <div style="margin:14px 0 7px;font-size:10.5px;color:var(--tx-3);text-transform:uppercase;
                letter-spacing:.07em;font-family:var(--font-mono)">SHAP Feature Attribution</div>
    <div class="shap-wrap">${shapRows}</div>

    <div style="margin-top:13px;background:var(--bg-overlay);border:1px solid var(--border);
                border-radius:var(--r-md);padding:11px 14px;font-size:13px;
                color:var(--tx-2);line-height:1.65">
      <span style="color:var(--tx);font-weight:500">Recommendation: </span>${esc(data.recommendation)}
    </div>

    <div style="margin-top:8px;font-size:10.5px;color:var(--tx-3);text-align:right;
                font-family:var(--font-mono)">
      Model v${esc(data.model_version)} · Confidence ${(data.confidence * 100).toFixed(0)}%
    </div>`;
}


/* ============================================================
   5. POLICY ASSISTANT (RAG CHAT)
   ============================================================ */

let chatBusy = false;
let msgIdx   = 0;

/** Fill the chat input with a preset question */
function setQ(question) {
  $('chat-input').value = question;
}

/** Send a chat message to the RAG endpoint */
async function sendChat() {
  if (chatBusy) return;
  const input = $('chat-input');
  const q = input.value.trim();
  if (!q) return;

  chatBusy = true;
  $('chat-send-btn').disabled = true;
  $('chat-send-btn').textContent = '…';
  input.value = '';

  appendMsg('user', q);
  const loaderId = appendMsg('loading', '');

  try {
    const res = await fetch(API + '/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: 4 }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    removeMsg(loaderId);
    appendMsg('ai', data.answer, data.sources, data.confidence);
    renderSources(data.sources);
  } catch (err) {
    removeMsg(loaderId);
    appendMsg('ai', 'Error: ' + (err.message || 'Request failed'));
  } finally {
    chatBusy = false;
    $('chat-send-btn').disabled = false;
    $('chat-send-btn').textContent = 'Send';
  }
}

/**
 * Append a message bubble to the chat history.
 * @param {'user'|'ai'|'loading'} type
 * @param {string} text
 * @param {Array}  sources - source citations
 * @param {string} conf    - confidence level
 * @returns {string} element ID (for later removal)
 */
function appendMsg(type, text, sources = [], conf = '') {
  const id   = 'msg-' + (++msgIdx);
  const hist = $('chat-history');
  let html;

  if (type === 'user') {
    html = `<div class="msg-user" id="${id}">${esc(text)}</div>`;

  } else if (type === 'loading') {
    html = `<div class="msg-ai" id="${id}">
              <div class="loader"><span></span><span></span><span></span></div>
            </div>`;

  } else {
    const confColor = conf === 'HIGH' ? 'var(--teal)' : conf === 'MEDIUM' ? 'var(--gold)' : 'var(--tx-3)';
    const confHtml  = conf
      ? `<div style="margin-top:5px;font-size:10.5px;color:${confColor};
                    font-family:var(--font-mono)">${conf} confidence</div>`
      : '';
    const srcHtml = sources.length
      ? `<div class="source-chips">
           ${sources.map(s => `<span class="source-chip">📄 ${esc(s.source)}${s.page != null ? ' p.' + s.page : ''}</span>`).join('')}
         </div>`
      : '';
    html = `<div class="msg-ai" id="${id}">${esc(text)}${srcHtml}${confHtml}</div>`;
  }

  hist.insertAdjacentHTML('beforeend', html);
  hist.scrollTop = hist.scrollHeight;
  return id;
}

/** Remove a message bubble by ID */
function removeMsg(id) {
  document.getElementById(id)?.remove();
}

/** Render retrieved source chunks in the right panel */
function renderSources(sources) {
  const pane = $('source-pane');
  if (!sources?.length) {
    pane.innerHTML = '<div class="empty-state" style="padding:16px 0">No sources retrieved</div>';
    return;
  }
  pane.innerHTML = sources.map((s, i) => `
    <div style="background:var(--bg-base);border:1px solid var(--border);border-radius:var(--r-md);
                padding:11px 13px;margin-bottom:8px">
      <div style="font-size:10.5px;color:var(--gold);font-family:var(--font-mono);margin-bottom:4px">
        📄 ${esc(s.source)}${s.page != null ? ' · Page ' + s.page : ''} · Chunk ${i + 1}
      </div>
      <div style="font-size:12px;color:var(--tx-2);line-height:1.6">${esc(s.excerpt)}</div>
    </div>`).join('');
}


/* ============================================================
   6. CLAIMS AGENT
   ============================================================ */

const AGENT_SAMPLES = {
  hail:  "Policyholder reports vehicle sustained significant hail damage during last night's storm. Vehicle was parked in the driveway. Claiming $3,200 for panel repairs across bonnet and roof. Third claim in 14 months.",
  theft: "Customer reports laptop and electronics stolen from home. Rear window broken for entry. Total claim $8,400. No security system installed. Property vacant for 3 weeks while owner was overseas.",
  flood: "Claimant states ground floor flooded following heavy rainfall. Water entered through front door. Claiming $45,000 for flooring, furniture and structural repairs. Property is in a designated flood zone.",
  fraud: "Claimant reports vehicle total loss from fire. Occurred at 3am in remote location. Vehicle financed, payments 6 months in arrears. Second fire claim in 18 months. Prepaid mobile used for lodgement.",
};

/** Load an agent sample description */
function setAgent(key) {
  $('agent-textarea').value = AGENT_SAMPLES[key];
}

/** Submit a claim description to the agent endpoint */
async function runAgent() {
  const btn  = $('ag-btn');
  const desc = $('agent-textarea').value.trim();
  if (!desc) return;

  btn.disabled = true;
  btn.textContent = 'Analysing…';
  $('ag-out').innerHTML = loaderHtml();
  $('ag-tools-card').style.display = 'none';
  $('ag-tools').innerHTML = '';

  try {
    const res = await fetch(API + '/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim_description: desc }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderAgent(data);
  } catch (err) {
    $('ag-out').innerHTML = errHtml(err.message || 'Agent failed');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Agent';
  }
}

/**
 * Render the full agent output: tool calls, risk assessment, policy context, recommendation.
 * @param {Object} data - /api/agent response
 */
function renderAgent(data) {
  $('ag-claim-id').textContent = data.claim_id;

  // Render tool calls panel
  if (data.tool_calls?.length) {
    $('ag-tools-card').style.display = '';
    $('ag-tools').innerHTML = data.tool_calls.map(t => `
      <div class="tool-call">
        <div class="tool-call-name">⚡ ${esc(t.tool_name)}</div>
        <div class="tool-call-io">Input: <span>${esc(t.input)}</span></div>
        <div class="tool-call-io" style="margin-top:2px">Output: <span>${esc(t.output)}</span></div>
      </div>`).join('');
  }

  let html = '';
  const ra = data.risk_assessment;
  const pc = data.policy_context;

  if (ra) {
    const pct = Math.round(ra.fraud_probability * 100);
    html += `
      <div style="margin-bottom:13px">
        <div style="font-size:10px;color:var(--tx-3);font-family:var(--font-mono);
                    text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Risk Assessment</div>
        <div style="display:flex;align-items:center;gap:12px;background:var(--bg-base);
                    border:1px solid var(--border);border-radius:var(--r-md);padding:12px">
          <div style="font-family:var(--font-disp);font-size:30px;font-weight:800;
                      color:${riskColor(ra.risk_label)};letter-spacing:-.03em">
            ${ra.fraud_probability.toFixed(3)}
          </div>
          <div>
            <div class="risk-pill risk-${ra.risk_label}" style="margin-bottom:4px">${ra.risk_label}</div>
            <div style="font-size:11.5px;color:var(--tx-3)">Risk score: ${pct}/100</div>
          </div>
        </div>
      </div>`;
  }

  if (pc) {
    const confColor = { HIGH: 'var(--teal)', MEDIUM: 'var(--gold)', LOW: 'var(--tx-3)' }[pc.confidence] || 'var(--tx-3)';
    const ans = pc.answer.substring(0, 420) + (pc.answer.length > 420 ? '…' : '');
    html += `
      <div style="margin-bottom:12px">
        <div style="font-size:10px;color:var(--tx-3);font-family:var(--font-mono);
                    text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">
          Policy Coverage · <span style="color:${confColor}">${pc.confidence} confidence</span>
        </div>
        <div style="background:var(--bg-base);border:1px solid var(--border);border-radius:var(--r-md);
                    padding:11px 14px;font-size:13px;color:var(--tx-2);line-height:1.7">${esc(ans)}</div>
      </div>`;
  }

  html += `
    <div class="rec-box">
      <div class="rec-label">Final Recommendation</div>
      <div class="rec-text">${esc(data.final_recommendation)}</div>
    </div>
    <div style="margin-top:7px;font-size:10.5px;color:var(--tx-3);text-align:right;font-family:var(--font-mono)">
      Processed in ${data.processing_time_ms}ms · ${data.tool_calls.length} tools called
    </div>`;

  $('ag-out').innerHTML = html;
}


/* ============================================================
   7. RESEARCH REPORT — SIDEBAR SCROLL TRACKING
   ============================================================ */

const REPORT_SECTIONS = ['r-exec', 'r-company', 'r-problems', 'r-aistack', 'r-opp', 'r-rec'];

/** Smooth-scroll to a report section */
function rScroll(sectionId) {
  document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/** Update the active sidebar link based on scroll position */
window.addEventListener('scroll', () => {
  let current = REPORT_SECTIONS[0];
  REPORT_SECTIONS.forEach(id => {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top < 160) current = id;
  });
  document.querySelectorAll('.sidebar-link').forEach((link, i) => {
    link.classList.toggle('active', REPORT_SECTIONS[i] === current);
  });
}, { passive: true });
