"use strict";
// STS2026 TTS Visualiser front end: no build step, no external libraries.

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const PAUSE_MS = 250;      // between sentences during playback
const GAP_SECONDS = 0.3;   // silence between sentences in a downloaded WAV
const MEL_HEIGHT = 150;    // CSS pixels
const MATRIX_HEIGHT = 110;
const STRIP_HEIGHT = 22;
const AXIS_HEIGHT = 16;
const MAGMA = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
  [250, 193, 39], [252, 255, 164]];

const state = {
  status: null,
  models: new Map(),   // id -> registry row, merged with load info
  segments: [],        // from /api/prepare
  tracks: [],          // [{model, results: [], errors: []}], A then optional B
  active: 0,           // index of the track being heard
  current: -1,         // index of the current segment
  loadedAudio: null,   // "track:segment" currently in the <audio> element
  playing: false,      // playback intended (may be waiting for synthesis)
  advanceTimer: null,
  run: 0,              // incremented to abandon an in-flight synthesis loop
  figures: [],         // per track: {mel, align, pxPerSec, result}
};

const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (MAGMA.length - 1);
    const k = Math.min(Math.floor(x), MAGMA.length - 2);
    const f = x - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = MAGMA[k][c] + f * (MAGMA[k + 1][c] - MAGMA[k][c]);
  }
  return lut;
})();

// ---------------------------------------------------------------------------------- helpers

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function setProgress(message) {
  $("progress").textContent = message;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function base64Bytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function fmt(seconds, digits = 2) {
  return `${seconds.toFixed(digits)} s`;
}

// --------------------------------------------------------------------------- status / models

async function init() {
  bindControls();
  try {
    state.status = await api("/api/status");
  } catch (error) {
    showError(`Cannot reach the server: ${error.message}`);
    $("device").textContent = "offline";
    return;
  }
  const s = state.status;
  $("device").textContent = s.gpu ? `${s.device} · ${s.gpu}` : s.device;
  for (const row of s.models) state.models.set(row.id, row);
  fillModelSelects();
  for (const example of s.examples) $("examples").append(el("option", {value: example.text, text: example.label}));
  if (s.examples.length && !$("text").value) $("text").value = s.examples[0].text;
  const evaluation = s.evaluation || {};
  $("eval-note").textContent = evaluation.note || "";
  $("limitations").replaceChildren(...s.limitations.map((item) => el("li", {text: item})));
  renderCards();
  modelChanged();
}

function fillModelSelects() {
  const a = $("model-a");
  const b = $("model-b");
  a.replaceChildren();
  b.replaceChildren(el("option", {value: "", text: "None"}));
  for (const row of state.models.values()) {
    const label = row.available ? row.label : `${row.label} (checkpoint not found)`;
    for (const select of [a, b]) {
      const option = el("option", {value: row.id, text: label});
      option.disabled = !row.available;
      select.append(option);
    }
  }
  const available = [...state.models.values()].filter((row) => row.available);
  const preferred = state.models.get(state.status.default_model);
  a.value = preferred && preferred.available ? preferred.id : (available[0] || {}).id || "";
}

async function ensureLoaded(id) {
  const row = state.models.get(id);
  if (row.loaded) return row;
  const info = await api("/api/load", {model: id});
  Object.assign(row, info, {loaded: true});
  return row;
}

async function modelChanged() {
  renderCards();
  const ids = [$("model-a").value, $("model-b").value].filter(Boolean);
  try {
    for (const id of ids) {
      if (!state.models.get(id).loaded) setProgress(`Loading ${state.models.get(id).label}…`);
      await ensureLoaded(id);
    }
    setProgress("");
  } catch (error) {
    showError(error.message);
    setProgress("");
  }
  renderCards();
  renderSpeakers();
}

function renderSpeakers() {
  const row = state.models.get($("model-a").value);
  const speakers = (row && row.speakers) || [];
  $("speaker-field").hidden = speakers.length === 0;
  const select = $("speaker");
  const previous = select.value;
  select.replaceChildren(...speakers.map((name) => el("option", {value: name, text: name})));
  if (speakers.includes(previous)) select.value = previous;
}

function renderCards() {
  const reference = ((state.status || {}).evaluation || {}).reference;
  const a = $("model-a").value;
  const b = $("model-b").value;
  const cards = [...state.models.values()].map((row) => {
    const card = el("article", {class: "card" + (row.id === a ? " sel-a" : row.id === b ? " sel-b" : "")});
    const role = row.id === a ? " (A)" : row.id === b ? " (B)" : "";
    card.append(el("h4", {text: row.label + role}));
    if (row.description) card.append(el("p", {text: row.description}));
    if (typeof row.wer === "number") {
      const wer = el("div", {class: "wer"});
      wer.append(el("div", {class: "label"}, el("span", {text: "Whisper WER"}),
        el("strong", {text: `${row.wer.toFixed(1)}%`})));
      const bar = el("div", {class: "bar"});
      const fill = el("div", {class: "fill"});
      fill.style.width = `${Math.min(row.wer, 100)}%`;
      bar.append(fill);
      if (reference) {
        const ref = el("div", {class: "ref", title: `${reference.label}: ${reference.wer}%`});
        ref.style.left = `${reference.wer}%`;
        bar.append(ref);
      }
      wer.append(bar);
      card.append(wer);
    }
    const details = [];
    if (row.loaded) {
      details.push(`step ${row.step ?? "?"} · ${(row.parameters / 1e6).toFixed(1)} M parameters · ${row.sample_rate} Hz`);
      if (row.switches) {
        const parts = ["text_encoder", "posterior_encoder", "decoder"].map((k) => `${k} ${row.switches[k]}`);
        details.push(parts.join(" · "));
      }
      if (row.speakers && row.speakers.length) details.push(`speakers: ${row.speakers.join(", ")}`);
    }
    for (const line of details) card.append(el("p", {class: "path", text: line}));
    card.append(row.available
      ? el("p", {class: "path", text: row.path})
      : el("p", {class: "path missing", text: `Not found: ${row.pattern}`}));
    return card;
  });
  if (reference) {
    cards.push(el("article", {class: "card"}, el("h4", {text: reference.label}),
      el("p", {text: "Whisper transcribing the real recordings of the same sentences, for scale (green mark on each bar)."}),
      el("div", {class: "wer"}, el("div", {class: "label"}, el("span", {text: "Whisper WER"}),
        el("strong", {text: `${reference.wer.toFixed(1)}%`})))));
  }
  $("cards").replaceChildren(...cards);
}

// ----------------------------------------------------------------------------------- synthesis

function readSettings() {
  const fixed = $("fixed-seed").checked;
  const seed = Math.max(0, Math.min(2147483647, Math.floor(Number($("seed").value) || 0)));
  return {
    noise_scale: Number($("noise-scale").value),
    length_scale: Number($("length-scale").value),
    seed: fixed ? seed : null,
  };
}

function speakerFor(id) {
  const speakers = state.models.get(id).speakers || [];
  if (!speakers.length) return null;
  const chosen = $("speaker").value;
  return speakers.includes(chosen) ? chosen : speakers[0];
}

async function speak() {
  const text = $("text").value;
  if (!text.trim()) {
    showError("Type or open some text first.");
    return;
  }
  showError("");
  stopPlayback();
  const run = ++state.run;
  const ids = [$("model-a").value, $("model-b").value].filter(Boolean);
  if (!ids.length) {
    showError("No checkpoint is available. Check webui/config.json.");
    return;
  }
  try {
    for (const id of ids) {
      if (!state.models.get(id).loaded) setProgress(`Loading ${state.models.get(id).label}…`);
      await ensureLoaded(id);
    }
    renderCards();
    const normalizer = state.models.get(ids[0]).normalizer || "english";
    const {segments} = await api("/api/prepare", {text, normalizer});
    if (run !== state.run) return;
    for (const track of state.tracks) for (const r of track.results) if (r) URL.revokeObjectURL(r.url);
    state.segments = segments;
    state.tracks = ids.map((model) => ({model, results: [], errors: []}));
    state.active = 0;
    document.querySelector('input[name="track"][value="0"]').checked = true;
    $("track-toggle").hidden = state.tracks.length < 2;
    $("download-b").hidden = state.tracks.length < 2;
    $("download-a").disabled = $("download-b").disabled = true;
    renderSegments();

    const playable = segments.filter((s) => !s.skip);
    if (!playable.length) {
      setProgress("Nothing to read: no letters left after text preparation.");
      return;
    }
    playSegment(playable[0].index);
    const settings = readSettings();
    let done = 0, compute = 0, seconds = 0;
    for (const segment of playable) {
      for (let t = 0; t < state.tracks.length; t++) {
        const track = state.tracks[t];
        try {
          const result = await api("/api/synthesize", {
            model: track.model, text: segment.normalized, ...settings, speaker: speakerFor(track.model),
          });
          if (run !== state.run) return;
          result.url = URL.createObjectURL(new Blob([base64Bytes(result.wav)], {type: "audio/wav"}));
          result.settings = settings;
          track.results[segment.index] = result;
          compute += result.infer_seconds;
          if (t === 0) seconds += result.seconds;
        } catch (error) {
          if (run !== state.run) return;
          track.errors[segment.index] = error.message;
        }
        updateSegmentMeta(segment.index);
        if (segment.index === state.current) {
          renderFigures();
          if (state.playing && t === state.active && state.loadedAudio === null) playSegment(segment.index);
        }
      }
      done++;
      setProgress(`Synthesised ${done}/${playable.length} · ${compute.toFixed(2)} s GPU time` +
        ` for ${seconds.toFixed(1)} s of audio (A)`);
    }
    $("download-a").disabled = false;
    $("download-b").disabled = state.tracks.length < 2;
  } catch (error) {
    if (run === state.run) {
      showError(error.message);
      setProgress("");
    }
  }
}

// ------------------------------------------------------------------------------------ reader

function wordSpans(normalized) {
  const node = el("div", {class: "norm"});
  const words = [];
  const re = /\S+/g;
  let match, last = 0;
  while ((match = re.exec(normalized))) {
    node.append(normalized.slice(last, match.index));
    const span = el("span", {class: "w", text: match[0]});
    node.append(span);
    words.push({start: match.index, end: match.index + match[0].length, span});
    last = match.index + match[0].length;
  }
  node.append(normalized.slice(last));
  return {node, words};
}

function renderSegments() {
  const list = $("segments");
  list.replaceChildren();
  if (!state.segments.length) {
    list.append(el("li", {class: "empty", text: "Sentences appear here. Click one to play it."}));
    return;
  }
  for (const segment of state.segments) {
    const item = el("li", {class: segment.skip ? "skip" : "", "data-index": segment.index});
    // Words are highlighted in the text the model reads; it replaces the original when identical.
    const {node, words} = wordSpans(segment.normalized);
    segment.words = words;
    if (segment.normalized === segment.original) {
      node.className = "orig";
      item.append(node);
    } else {
      item.append(el("div", {class: "orig", text: segment.original}), node);
    }
    if (segment.dropped.length) {
      item.append(el("div", {class: "dropped", text: `Not read: ${segment.dropped.join(" ")}`}));
    }
    item.append(el("div", {class: "meta"}));
    if (!segment.skip) {
      item.tabIndex = 0;
      item.addEventListener("click", () => playSegment(segment.index));
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter") playSegment(segment.index);
      });
    }
    segment.item = item;
    list.append(item);
    updateSegmentMeta(segment.index);
  }
}

function updateSegmentMeta(index) {
  const segment = state.segments[index];
  if (!segment || !segment.item) return;
  const meta = segment.item.querySelector(".meta");
  meta.replaceChildren();
  if (segment.skip) {
    meta.append(el("span", {class: "tag", text: "skipped: nothing to read"}));
    return;
  }
  state.tracks.forEach((track, t) => {
    const name = t === 0 ? "A" : "B";
    const result = track.results[index];
    const error = track.errors[index];
    if (result) {
      meta.append(el("span", {class: `tag ${name.toLowerCase()}`,
        text: `${name} ${fmt(result.seconds, 1)} audio · ${fmt(result.infer_seconds, 3)} · RTF ${result.rtf.toFixed(3)}`}));
    } else if (error) {
      meta.append(el("span", {class: "tag err", text: `${name} failed: ${error}`}));
    } else {
      meta.append(el("span", {class: "tag", text: `${name} …`}));
    }
  });
}

function highlightCurrent() {
  for (const segment of state.segments) {
    if (!segment.item) continue;
    const on = segment.index === state.current;
    segment.item.classList.toggle("current", on);
    for (const word of segment.words || []) word.span.classList.remove("on");
    if (on) {
      segment.item.setAttribute("aria-current", "true");
      segment.item.scrollIntoView({block: "nearest"});
    } else {
      segment.item.removeAttribute("aria-current");
    }
  }
}

function playableFrom(index, step) {
  for (let i = index; i >= 0 && i < state.segments.length; i += step) {
    if (!state.segments[i].skip) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------------- playback

function playSegment(index) {
  if (index < 0 || index >= state.segments.length) return;
  clearTimeout(state.advanceTimer);
  const changed = index !== state.current;
  state.current = index;
  state.playing = true;
  highlightCurrent();
  if (changed || !state.figures.length) renderFigures();
  const track = state.tracks[state.active];
  const result = track && track.results[index];
  if (track && track.errors[index]) {
    advance();
    return;
  }
  if (!result) {
    // Not synthesised yet: the synthesis loop starts playback when it arrives.
    audio.pause();
    state.loadedAudio = null;
    updatePlayButton();
    return;
  }
  state.loadedAudio = `${state.active}:${index}`;
  audio.src = result.url;
  audio.currentTime = 0;
  audio.play().catch((error) => {
    state.playing = false;
    updatePlayButton();
    if (error.name !== "AbortError") showError(`Playback blocked by the browser: ${error.message}. Press Play.`);
  });
  updatePlayButton();
}

function advance() {
  const next = playableFrom(state.current + 1, 1);
  if (next < 0) {
    state.playing = false;
    updatePlayButton();
    return;
  }
  playSegment(next);
}

function togglePlay() {
  if (state.playing) {
    state.playing = false;
    clearTimeout(state.advanceTimer);
    audio.pause();
    updatePlayButton();
    return;
  }
  if (!state.segments.length) return;
  const index = state.current >= 0 ? state.current : playableFrom(0, 1);
  if (state.loadedAudio === `${state.active}:${index}` && audio.src && !audio.ended) {
    state.playing = true;
    audio.play();
    updatePlayButton();
  } else {
    playSegment(index);
  }
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.advanceTimer);
  audio.pause();
  state.loadedAudio = null;
  updatePlayButton();
}

function step(direction) {
  if (!state.segments.length) return;
  const target = playableFrom(state.current + direction, direction);
  if (target < 0) return;
  if (state.playing) {
    playSegment(target);
  } else {
    state.current = target;
    state.loadedAudio = null;
    highlightCurrent();
    renderFigures();
  }
}

function updatePlayButton() {
  const button = $("play");
  button.textContent = state.playing ? "Pause" : "Play";
  button.setAttribute("aria-label", state.playing ? "Pause" : "Play");
}

function tick() {
  const result = currentResult();
  if (result) {
    $("clock").textContent = `${audio.currentTime.toFixed(1)} / ${result.seconds.toFixed(1)} s`;
    highlightWord(result, audio.currentTime);
  }
  drawPlayheads();
  if (!audio.paused) requestAnimationFrame(tick);
}

function currentResult() {
  if (state.loadedAudio !== `${state.active}:${state.current}`) return null;
  return state.tracks[state.active].results[state.current] || null;
}

function tokenAt(result, seconds) {
  const frame = (seconds * result.sample_rate) / result.alignment.hop_length;
  let end = 0;
  const durations = result.alignment.durations;
  for (let i = 0; i < durations.length; i++) {
    end += durations[i];
    if (frame < end) return i;
  }
  return durations.length - 1;
}

function highlightWord(result, seconds) {
  const segment = state.segments[state.current];
  if (!segment || !segment.words) return;
  const tokens = result.alignment.tokens.join("");
  const token = tokens === segment.normalized ? tokenAt(result, seconds) : -1;
  for (const word of segment.words) word.span.classList.toggle("on", token >= word.start && token < word.end);
}

// ------------------------------------------------------------------------------------ figures

function renderFigures() {
  const box = $("figures");
  const segment = state.segments[state.current];
  state.figures = [];
  if (!segment || !state.tracks.length) {
    box.replaceChildren(el("p", {class: "muted empty",
      text: "The mel spectrogram and the text-to-frame alignment of the current sentence appear here."}));
    $("visual-caption").textContent = "";
    return;
  }
  $("visual-caption").textContent = `Sentence ${state.current + 1} of ${state.segments.length}`;
  const results = state.tracks.map((track) => track.results[state.current]);
  const maxSeconds = Math.max(0.5, ...results.filter(Boolean).map((r) => r.seconds));
  const width = Math.max(200, box.clientWidth);
  box.replaceChildren(...state.tracks.map((track, t) => buildFigure(track, t, results[t], maxSeconds, width)));
  drawPlayheads();
}

function buildFigure(track, t, result, maxSeconds, width) {
  const row = state.models.get(track.model);
  const name = t === 0 ? "A" : "B";
  const figure = el("figure", {class: "figure"});
  const head = el("div", {class: "figure-head"},
    el("strong", {}, el("span", {class: `swatch ${name.toLowerCase()}`}), `${name}: ${row.label}`));
  figure.append(head);
  const error = track.errors[state.current];
  if (!result) {
    head.append(el("span", {class: "muted", text: error ? `failed: ${error}` : "synthesising…"}));
    state.figures[t] = null;
    return figure;
  }
  head.append(el("span", {class: "muted",
    text: `seed ${result.seed} · ${fmt(result.seconds)} audio · ${fmt(result.infer_seconds, 3)} on ${result.device}` +
      ` · RTF ${result.rtf.toFixed(4)} · noise ${result.settings.noise_scale} · length ${result.settings.length_scale}`}));
  const pxPerSec = width / maxSeconds;
  const mel = makeCanvas(width, MEL_HEIGHT, "mel", "Mel spectrogram");
  const align = makeCanvas(width, MATRIX_HEIGHT + STRIP_HEIGHT + AXIS_HEIGHT, "align", "Text-to-frame alignment");
  drawMel(mel, result, pxPerSec);
  drawAlignment(align, result, pxPerSec, t);
  const figureState = {mel, align, pxPerSec, result, track: t};
  state.figures[t] = figureState;
  for (const canvas of [mel, align]) {
    canvas.addEventListener("mousemove", (event) => hover(event, figureState, canvas === mel));
    canvas.addEventListener("mouseleave", () => ($("tooltip").hidden = true));
    canvas.addEventListener("click", (event) => seek(event, figureState));
  }
  const ramp = el("span", {class: "ramp"});
  ramp.style.background = `linear-gradient(90deg, ${MAGMA.map((c) => `rgb(${c})`).join(",")})`;
  const [low, high] = result.mel.range;
  figure.append(mel, align, el("div", {class: "legend"}, el("span", {text: `log mel ${low}`}), ramp,
    el("span", {text: `${high}`}), el("span", {text: `· peak ${result.mel.peak.toFixed(2)} · hover for values, click to seek`})));
  return figure;
}

function makeCanvas(width, height, className, label) {
  const ratio = window.devicePixelRatio || 1;
  const canvas = el("canvas", {class: className, role: "img", "aria-label": label});
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  canvas.cssWidth = width;
  canvas.cssHeight = height;
  return canvas;
}

function drawMel(canvas, result, pxPerSec) {
  const {n_mels: bands, frames} = result.mel;
  const data = base64Bytes(result.mel.data);
  result.melBytes = data;
  const image = new ImageData(frames, bands);
  for (let m = 0; m < bands; m++) {
    const y = bands - 1 - m;  // low frequencies at the bottom
    for (let f = 0; f < frames; f++) {
      const v = data[m * frames + f];
      const o = (y * frames + f) * 4;
      image.data[o] = LUT[v * 3];
      image.data[o + 1] = LUT[v * 3 + 1];
      image.data[o + 2] = LUT[v * 3 + 2];
      image.data[o + 3] = 255;
    }
  }
  const source = document.createElement("canvas");
  source.width = frames;
  source.height = bands;
  source.getContext("2d").putImageData(image, 0, 0);
  const ratio = canvas.width / canvas.cssWidth;
  const base = document.createElement("canvas");
  base.width = canvas.width;
  base.height = canvas.height;
  const ctx = base.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, result.seconds * pxPerSec, canvas.cssHeight);
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "10px ui-monospace, Consolas, monospace";
  ctx.fillText(`${bands} mel bands`, 4, 12);
  canvas.base = base;
}

function drawAlignment(canvas, result, pxPerSec, t) {
  const {tokens, durations, frames} = result.alignment;
  const ratio = canvas.width / canvas.cssWidth;
  const base = document.createElement("canvas");
  base.width = canvas.width;
  base.height = canvas.height;
  const ctx = base.getContext("2d");
  ctx.scale(ratio, ratio);
  const color = cssVar(t === 0 ? "--track-a" : "--track-b");
  const ink = cssVar("--ink-2");
  const rule = cssVar("--rule");
  const framePx = (pxPerSec * result.alignment.hop_length) / result.sample_rate;
  const rowH = MATRIX_HEIGHT / tokens.length;
  // Matrix: y = character index (first at the top), x = output frame.
  ctx.fillStyle = color;
  let start = 0;
  const boxes = [];
  for (let i = 0; i < tokens.length; i++) {
    const x = start * framePx;
    const w = durations[i] * framePx;
    if (durations[i] > 0) ctx.fillRect(x, i * rowH, Math.max(w, 1), Math.max(rowH, 1));
    boxes.push({x, w, start, end: start + durations[i]});
    start += durations[i];
  }
  // Character strip under the matrix.
  const y0 = MATRIX_HEIGHT;
  ctx.font = "11px ui-monospace, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  boxes.forEach((box, i) => {
    if (box.w <= 0) return;
    ctx.fillStyle = i % 2 ? rule : "transparent";
    ctx.fillRect(box.x, y0, box.w, STRIP_HEIGHT);
    if (box.w >= 6 && tokens[i] !== " ") {
      ctx.fillStyle = ink;
      ctx.fillText(tokens[i], box.x + box.w / 2, y0 + STRIP_HEIGHT / 2);
    }
  });
  // Time axis.
  const total = frames * result.alignment.hop_length / result.sample_rate;
  const stepSeconds = pxPerSec > 160 ? 0.5 : 1;
  ctx.fillStyle = cssVar("--ink-3");
  ctx.textBaseline = "top";
  for (let s = 0; s <= total + 1e-9; s += stepSeconds) {
    const x = s * pxPerSec;
    ctx.fillRect(x, y0 + STRIP_HEIGHT, 1, 4);
    ctx.textAlign = s === 0 ? "left" : "center";
    ctx.fillText(`${s.toFixed(stepSeconds < 1 ? 1 : 0)}s`, x, y0 + STRIP_HEIGHT + 4);
  }
  canvas.base = base;
  canvas.boxes = boxes;
  canvas.rowH = rowH;
}

function drawPlayheads() {
  const activeResult = currentResult();
  state.figures.forEach((figure, t) => {
    if (!figure) return;
    const playing = t === state.active && activeResult === figure.result;
    const x = playing ? audio.currentTime * figure.pxPerSec : null;
    for (const canvas of [figure.mel, figure.align]) {
      const ctx = canvas.getContext("2d");
      const ratio = canvas.width / canvas.cssWidth;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(canvas.base, 0, 0);
      if (x === null) continue;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      if (canvas === figure.align) {
        const i = tokenAt(figure.result, audio.currentTime);
        const box = canvas.boxes[i];
        if (box) {
          ctx.strokeStyle = cssVar("--ink");
          ctx.lineWidth = 1.5;
          ctx.strokeRect(box.x, MATRIX_HEIGHT + 0.75, Math.max(box.w, 2), STRIP_HEIGHT - 1.5);
          ctx.fillStyle = "rgba(255, 212, 59, 0.25)";
          ctx.fillRect(0, i * canvas.rowH, canvas.cssWidth, Math.max(canvas.rowH, 1));
        }
      }
      ctx.fillStyle = canvas === figure.mel ? "#ffffff" : cssVar("--ink");
      ctx.fillRect(x - 0.75, 0, 1.5, canvas.cssHeight);
    }
  });
}

function hover(event, figure, isMel) {
  const rect = event.target.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const result = figure.result;
  const seconds = x / figure.pxPerSec;
  const tooltip = $("tooltip");
  if (seconds > result.seconds) {
    tooltip.hidden = true;
    return;
  }
  let text;
  if (isMel) {
    const {n_mels: bands, frames, range} = result.mel;
    const f = Math.min(frames - 1, Math.floor((seconds * result.sample_rate) / result.mel.hop_length));
    const m = Math.min(bands - 1, Math.max(0, bands - 1 - Math.floor((y / figure.mel.cssHeight) * bands)));
    const v = range[0] + (result.melBytes[m * frames + f] / 255) * (range[1] - range[0]);
    text = `${seconds.toFixed(2)} s · frame ${f} · mel band ${m} · log mel ${v.toFixed(2)}`;
  } else {
    const i = tokenAt(result, seconds);
    const box = figure.align.boxes[i];
    const ms = ((box.end - box.start) * result.alignment.hop_length / result.sample_rate) * 1000;
    const ch = result.alignment.tokens[i] === " " ? "space" : `"${result.alignment.tokens[i]}"`;
    text = `${ch} · character ${i + 1}/${result.alignment.tokens.length} · frames ${box.start}–${box.end - 1} · ${ms.toFixed(0)} ms`;
  }
  tooltip.textContent = text;
  tooltip.hidden = false;
  const left = Math.min(event.clientX + 12, window.innerWidth - tooltip.offsetWidth - 8);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
}

function seek(event, figure) {
  const rect = event.target.getBoundingClientRect();
  const seconds = (event.clientX - rect.left) / figure.pxPerSec;
  if (seconds > figure.result.seconds) return;
  if (state.active !== figure.track) setTrack(figure.track, false);
  if (state.loadedAudio !== `${state.active}:${state.current}`) playSegment(state.current);
  if (state.loadedAudio !== `${state.active}:${state.current}`) return;
  audio.currentTime = seconds;
  if (!state.playing) {
    state.playing = true;
    audio.play();
    updatePlayButton();
  }
}

function setTrack(t, restart = true) {
  state.active = t;
  document.querySelector(`input[name="track"][value="${t}"]`).checked = true;
  if (restart && state.playing) {
    playSegment(state.current);
    return;
  }
  audio.pause();
  state.loadedAudio = null;
  drawPlayheads();
}

// ----------------------------------------------------------------------------------- download

function pcmFromWav(bytes) {
  const view = new DataView(bytes.buffer);
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const id = String.fromCharCode(...bytes.slice(offset, offset + 4));
    const size = view.getUint32(offset + 4, true);
    if (id === "data") return bytes.slice(offset + 8, offset + 8 + size);
    offset += 8 + size + (size % 2);
  }
  throw new Error("No data chunk in WAV.");
}

function download(t) {
  const track = state.tracks[t];
  if (!track) return;
  const results = state.segments.map((s) => track.results[s.index]).filter(Boolean);
  if (!results.length) return;
  const rate = results[0].sample_rate;
  const gap = new Uint8Array(Math.round(GAP_SECONDS * rate) * 2);
  const parts = [];
  results.forEach((result, i) => {
    if (i) parts.push(gap);
    parts.push(pcmFromWav(base64Bytes(result.wav)));
  });
  const size = parts.reduce((n, p) => n + p.length, 0);
  const header = new DataView(new ArrayBuffer(44));
  const write = (o, s) => [...s].forEach((c, i) => header.setUint8(o + i, c.charCodeAt(0)));
  write(0, "RIFF"); header.setUint32(4, 36 + size, true); write(8, "WAVE");
  write(12, "fmt "); header.setUint32(16, 16, true); header.setUint16(20, 1, true); header.setUint16(22, 1, true);
  header.setUint32(24, rate, true); header.setUint32(28, rate * 2, true); header.setUint16(32, 2, true);
  header.setUint16(34, 16, true); write(36, "data"); header.setUint32(40, size, true);
  const url = URL.createObjectURL(new Blob([header.buffer, ...parts], {type: "audio/wav"}));
  const link = el("a", {href: url, download: `sts2026_${track.model}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`});
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ----------------------------------------------------------------------------------- controls

function bindControls() {
  $("speak").addEventListener("click", speak);
  $("stop").addEventListener("click", () => {
    state.run++;
    stopPlayback();
    setProgress("Stopped.");
  });
  $("play").addEventListener("click", togglePlay);
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("download-a").addEventListener("click", () => download(0));
  $("download-b").addEventListener("click", () => download(1));
  $("model-a").addEventListener("change", modelChanged);
  $("model-b").addEventListener("change", modelChanged);
  for (const input of document.querySelectorAll('input[name="track"]')) {
    input.addEventListener("change", () => setTrack(Number(input.value)));
  }
  $("length-scale").addEventListener("input", (e) => ($("length-out").textContent = Number(e.target.value).toFixed(2)));
  $("noise-scale").addEventListener("input", (e) => ($("noise-out").textContent = Number(e.target.value).toFixed(3)));
  $("fixed-seed").addEventListener("change", (e) => ($("seed").disabled = !e.target.checked));
  $("show-norm").addEventListener("change", (e) => document.body.classList.toggle("hide-norm", !e.target.checked));
  $("examples").addEventListener("change", (e) => {
    if (e.target.value) $("text").value = e.target.value;
    e.target.value = "";
  });
  $("file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $("text").value = await file.text();
    e.target.value = "";
  });
  $("text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      speak();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("textarea, input, select, button") || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === " ") {
      e.preventDefault();
      togglePlay();
    } else if (e.key === "ArrowLeft") {
      step(-1);
    } else if (e.key === "ArrowRight") {
      step(1);
    }
  });
  audio.addEventListener("play", () => requestAnimationFrame(tick));
  audio.addEventListener("seeked", tick);
  audio.addEventListener("ended", () => {
    tick();
    if (state.playing) state.advanceTimer = setTimeout(advance, PAUSE_MS);
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderFigures, 150);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderFigures);
}

init();
