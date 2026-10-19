"use strict";
// Agent mode: a simulated LLM streams a reply word by word and up to three TTS strategies speak it
// with real chunk-synthesis calls. Uses api(), el(), showError(), base64Bytes(), pcmFromWav(),
// ensureLoaded(), speakerFor(), cssVar() and state from app.js.

(() => {
  const $a = (id) => document.getElementById(id);
  const LANES = [{name: "A", color: "--track-a"}, {name: "B", color: "--track-b"}, {name: "C", color: "--lane-c"}];
  const SIZES = [1, 2, 4, 8];
  const LOOKAHEADS = [0, 2, 4];
  const RATE_MIN = 2;
  const RATE_MAX = 60;
  const LEAD = 0.03;            // s from a chunk's audio arriving to the earliest time it can play
  const AB_GAP = 0.7;           // s of silence between A and B in an A/B comparison
  const SECONDS_PER_WORD = 0.34; // only for the timeline's time scale while a run is in progress
  const GUTTER = 118;
  const LLM_ROW = 30;
  const LANE_ROW = 58;
  const AXIS_ROW = 22;

  const ui = {status: null, study: null, studyFallback: false, lanes: [], token: 0, run: null, ctx: null,
    sources: [], playback: null, raf: 0, hits: [], bound: false};

  // ------------------------------------------------------------------------------- strategies

  function codeOf(s) {
    if (s.kind === "full") return "full";
    if (s.kind === "naive") return `naive_${s.n}`;
    if (s.kind === "ctx") return `ctx_${s.n}_k${s.k}`;
    return "off";
  }

  function parseCode(code) {
    let m;
    if (code === "full") return {kind: "full", n: 2, k: 2};
    if ((m = /^naive_(\d+)$/.exec(code))) return {kind: "naive", n: Number(m[1]), k: 2};
    if ((m = /^ctx_(\d+)_k(\d+)$/.exec(code))) return {kind: "ctx", n: Number(m[1]), k: Number(m[2])};
    return {kind: "off", n: 2, k: 2};
  }

  function nameOf(s) {
    if (s.kind === "full") return "Whole sentence";
    if (s.kind === "naive") return `Independent ${s.n}-word chunks`;
    if (s.kind === "ctx") return `Context-aware ${s.n}-word chunks, lookahead ${s.k}`;
    return "Off";
  }

  /** Words of a sentence that must arrive before its first audio can be synthesised. */
  function wordsNeeded(s, length) {
    if (s.kind === "full") return length;
    return Math.min(s.n + (s.kind === "ctx" ? s.k : 0), length);
  }

  /** Chunks of one sentence: word range, words needed before the call, and the request mode. */
  function planSentence(s, length) {
    if (s.kind === "full") return [{start: 0, end: length, need: length, mode: "context", lookahead: 0}];
    const k = s.kind === "ctx" ? s.k : 0;
    const chunks = [];
    for (let a = 0; a < length; a += s.n) {
      const b = Math.min(a + s.n, length);
      chunks.push({start: a, end: b, need: Math.min(b + k, length),
        mode: s.kind === "ctx" ? "context" : "independent", lookahead: k});
    }
    return chunks;
  }

  function activeLanes() {
    return ui.lanes.map((strategy, index) => ({strategy, index})).filter((lane) => lane.strategy.kind !== "off");
  }

  function color(index) {
    return cssVar(LANES[index].color);
  }

  // ---------------------------------------------------------------------------------- set-up

  function rateValue() {
    const v = Number($a("a-rate").value) / 1000;
    const rate = RATE_MIN * Math.pow(RATE_MAX / RATE_MIN, v);
    return rate < 10 ? Math.round(rate * 2) / 2 : Math.round(rate);
  }

  function rateToSlider(rate) {
    return Math.round(1000 * Math.log(rate / RATE_MIN) / Math.log(RATE_MAX / RATE_MIN));
  }

  function updateSliderLabels() {
    $a("a-ttft-out").textContent = `${$a("a-ttft").value} ms`;
    $a("a-rate-out").textContent = `${rateValue()} words/s`;
  }

  function setup(status) {
    ui.status = status;
    const config = status.agent || {};
    const preset = $a("a-preset");
    preset.replaceChildren();
    for (const reply of config.replies || []) preset.append(el("option", {value: reply.text, text: reply.label}));
    preset.append(el("option", {value: "", text: "Custom text…"}));
    $a("a-text").value = (config.replies && config.replies[0] && config.replies[0].text) || "";
    $a("a-ttft").value = config.ttft_ms ?? 300;
    $a("a-rate").value = rateToSlider(config.words_per_second ?? 10);
    updateSliderLabels();

    const select = $a("a-model");
    select.replaceChildren();
    for (const row of status.models) {
      const option = el("option", {value: row.id, text: row.available ? row.label : `${row.label} (checkpoint not found)`});
      option.disabled = !row.available;
      select.append(option);
    }
    const preferred = status.models.find((row) => row.id === config.default_model && row.available)
      || status.models.find((row) => row.available);
    if (preferred) select.value = preferred.id;

    ui.lanes = (config.lanes || ["full", "naive_2", "ctx_1_k2"]).slice(0, 3).map(parseCode);
    while (ui.lanes.length < 3) ui.lanes.push({kind: "off", n: 2, k: 2});
    bind();
    modelSelected();
  }

  function bind() {
    if (ui.bound) return;
    ui.bound = true;
    $a("a-preset").addEventListener("change", (e) => {
      if (e.target.value) $a("a-text").value = e.target.value;
      else $a("a-text").focus();
    });
    $a("a-text").addEventListener("input", () => {
      const preset = $a("a-preset");
      if (preset.value && preset.value !== $a("a-text").value) preset.value = "";
    });
    $a("a-text").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        send();
      }
    });
    $a("a-ttft").addEventListener("input", updateSliderLabels);
    $a("a-rate").addEventListener("input", () => {
      updateSliderLabels();
      renderSpeed();
    });
    $a("a-length").addEventListener("input", renderSpeed);
    $a("a-model").addEventListener("change", modelSelected);
    $a("a-fixed-seed").addEventListener("change", (e) => ($a("a-seed").disabled = !e.target.checked));
    $a("a-run").addEventListener("click", send);
    $a("a-stop").addEventListener("click", stop);
    $a("a-ab-play").addEventListener("click", () => {
      play([Number($a("a-ab-a").value), Number($a("a-ab-b").value)], false);
    });
    const canvas = $a("a-canvas");
    canvas.addEventListener("mousemove", hover);
    canvas.addEventListener("mouseleave", () => ($("tooltip").hidden = true));
    let timer;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(draw, 100);
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      draw();
      renderTradeoff();
      renderSpeed();
    });
  }

  async function modelSelected() {
    const id = $a("a-model").value;
    const studies = ui.status.chunk_studies || {};
    ui.study = studies[id] || null;
    ui.studyFallback = false;
    if (!ui.study) {
      const first = Object.keys(studies)[0];
      if (first) {
        ui.study = studies[first];
        ui.studyFallback = first;
      }
    }
    lanesChanged();
    if (!id) return;
    try {
      if (!state.models.get(id).loaded) setProgress(`Loading ${state.models.get(id).label}…`);
      await ensureLoaded(id);
      setProgress("");
    } catch (error) {
      showError(error.message);
      setProgress("");
    }
  }

  function setProgress(text) {
    $a("a-progress").textContent = text;
  }

  function lanesChanged() {
    renderLaneEditors();
    renderListen();
    renderAbSelects();
    renderCards();
    renderTradeoff();
    renderSpeed();
    draw();
  }

  function studyCondition(s) {
    return ui.study && ui.study.conditions[codeOf(s)];
  }

  function pct(x, digits = 1) {
    return `${(100 * x).toFixed(digits)}%`;
  }

  function choice(options, value, label) {
    const select = el("select", {"aria-label": label});
    for (const [v, text] of options) select.append(el("option", {value: String(v), text}));
    select.value = String(value);
    return select;
  }

  function renderLaneEditors() {
    const box = $a("a-lanes");
    box.replaceChildren();
    ui.lanes.forEach((s, i) => {
      const lane = el("div", {class: "lane"});
      lane.style.setProperty("--lane", `var(${LANES[i].color})`);
      const swatch = el("span", {class: "swatch"});
      swatch.style.background = `var(${LANES[i].color})`;
      lane.append(el("div", {class: "lane-head"}, swatch, `Strategy ${LANES[i].name}`, el("code", {text: codeOf(s)})));
      const kind = choice([["full", "Whole sentence"], ["naive", "Independent chunks"], ["ctx", "Context-aware chunks"],
        ["off", "Off"]], s.kind, `Strategy ${LANES[i].name} type`);
      const size = choice(SIZES.map((n) => [n, `${n} word${n > 1 ? "s" : ""}`]), s.n, `Strategy ${LANES[i].name} chunk size`);
      const look = choice(LOOKAHEADS.map((k) => [k, `lookahead ${k}`]), s.k, `Strategy ${LANES[i].name} lookahead`);
      size.hidden = s.kind === "full" || s.kind === "off";
      look.hidden = s.kind !== "ctx";
      const update = () => {
        ui.lanes[i] = {kind: kind.value, n: Number(size.value), k: Number(look.value)};
        lanesChanged();
      };
      for (const select of [kind, size, look]) select.addEventListener("change", update);
      kind.disabled = size.disabled = look.disabled = isRunning();
      lane.append(el("div", {class: "lane-row"}, kind, size, look));
      const condition = studyCondition(s);
      if (condition) {
        lane.append(el("div", {class: "study",
          text: `Study: WER ${pct(condition.wer)} · waits ${condition.wait_words.toFixed(1)} words on average`}));
      } else if (s.kind !== "off") {
        lane.append(el("div", {class: "study", text: "Not a condition of the chunk study."}));
      }
      box.append(lane);
    });
  }

  function renderListen() {
    const box = $a("a-listen");
    const previous = box.querySelector("input:checked");
    const lanes = activeLanes();
    box.replaceChildren(el("span", {text: "Listen to"}));
    for (const {index} of lanes) {
      const input = el("input", {type: "radio", name: "a-listen", value: String(index),
        "aria-label": `Listen to strategy ${LANES[index].name}`});
      input.disabled = isRunning();
      const swatch = el("span", {class: "swatch"});
      swatch.style.background = `var(${LANES[index].color})`;
      box.append(el("label", {}, input, swatch, LANES[index].name));
    }
    const inputs = [...box.querySelectorAll("input")];
    const keep = previous && inputs.find((input) => input.value === previous.value);
    const fallback = inputs.find((input) => ui.lanes[Number(input.value)].kind === "ctx") || inputs[0];
    if (keep || fallback) (keep || fallback).checked = true;
  }

  function listenValue() {
    const input = $a("a-listen").querySelector("input:checked");
    return input ? Number(input.value) : -1;
  }

  function renderAbSelects() {
    const lanes = activeLanes();
    for (const [id, preferred] of [["a-ab-a", "naive"], ["a-ab-b", "ctx"]]) {
      const select = $a(id);
      const previous = select.value;
      select.replaceChildren(...lanes.map(({strategy, index}) =>
        el("option", {value: String(index), text: `${LANES[index].name}: ${codeOf(strategy)}`})));
      const match = lanes.find((lane) => String(lane.index) === previous)
        || lanes.find((lane) => lane.strategy.kind === preferred) || lanes[0];
      if (match) select.value = String(match.index);
    }
    $a("a-ab-play").disabled = !(ui.run && ui.run.finished) || lanes.length === 0;
  }

  function isRunning() {
    return Boolean(ui.run && !ui.run.finished);
  }

  // ------------------------------------------------------------------------------------- run

  function elapsed(run) {
    return (performance.now() - run.t0) / 1000;
  }

  async function send() {
    stopAudio();
    const token = ++ui.token;
    const model = $a("a-model").value;
    const text = $a("a-text").value.trim();
    const lanes = activeLanes();
    if (!text || !model || !lanes.length) {
      showError(!text ? "Type a reply first." : !model ? "No checkpoint is available." : "Turn on at least one strategy.");
      return;
    }
    showError("");
    try {
      ui.ctx = ui.ctx || new (window.AudioContext || window.webkitAudioContext)();
      await ui.ctx.resume();
      if (!state.models.get(model).loaded) setProgress("Loading the model…");
      await ensureLoaded(model);
      const normalizer = state.models.get(model).normalizer || "english";
      const {segments} = await api("/api/prepare", {text, normalizer});
      if (token !== ui.token) return;
      const sentences = segments.filter((s) => !s.skip).map((s) => s.normalized.split(" ").filter(Boolean));
      if (!sentences.length) {
        showError("Nothing to say: no letters left after text preparation.");
        return;
      }
      const ttft = Number($a("a-ttft").value) / 1000;
      const rate = rateValue();
      const words = [];
      const offsets = [];
      sentences.forEach((list, si) => {
        offsets.push(words.length);
        list.forEach((word, wi) => words.push({text: word, sentence: si, index: wi, at: ttft + words.length / rate}));
      });
      const fixed = $a("a-fixed-seed").checked;
      const seed = fixed ? Math.max(0, Math.min(2147000000, Math.floor(Number($a("a-seed").value) || 0)))
        : Math.floor(Math.random() * 2147000000);
      const run = {
        token, model, ttft, rate, sentences, offsets, words, seed, speaker: speakerFor(model),
        listen: listenValue(), arrived: 0, finished: false, scheduled: false, end: 0, timer: 0,
        lanes: lanes.map(({strategy, index}) => ({
          index, strategy: {...strategy}, code: codeOf(strategy), queue: [], busy: false, next: 0,
          playEnd: 0, firstAudio: null, stalls: [], errors: 0,
          chunks: sentences.flatMap((list, si) => planSentence(strategy, list.length).map((c) => ({
            ...c, sentence: si, gStart: offsets[si] + c.start, gEnd: offsets[si] + c.end,
            gNeed: offsets[si] + c.need, state: "waiting"}))),
        })),
      };
      ui.run = run;
      renderBubble(run);
      lanesChanged();
      run.t0 = performance.now();
      run.ctx0 = ui.ctx.currentTime;
      run.timer = setInterval(() => step(run), 8);
      step(run);
      loop();
    } catch (error) {
      if (token === ui.token) {
        showError(error.message);
        setProgress("");
      }
    }
  }

  function step(run) {
    if (run.token !== ui.token || run.finished) {
      clearInterval(run.timer);
      return;
    }
    const t = elapsed(run);
    while (run.arrived < run.words.length && run.words[run.arrived].at <= t) {
      revealWord(run.arrived);
      run.arrived++;
    }
    for (const lane of run.lanes) {
      while (lane.next < lane.chunks.length && lane.chunks[lane.next].gNeed <= run.arrived) {
        const chunk = lane.chunks[lane.next++];
        chunk.state = "queued";
        chunk.textAt = run.words[chunk.gNeed - 1].at;
        lane.queue.push(chunk);
      }
      if (!lane.busy && lane.queue.length) worker(run, lane);
    }
    if (!run.scheduled && run.arrived === run.words.length
        && run.lanes.every((lane) => lane.next === lane.chunks.length && !lane.busy && !lane.queue.length)) {
      run.scheduled = true;
      run.end = Math.max(...run.lanes.map((lane) => lane.playEnd));
      renderCards();
    }
    if (run.scheduled && t > run.end + 0.15) finish(run);
    else setProgress(progressText(run, t));
  }

  async function worker(run, lane) {
    lane.busy = true;
    while (lane.queue.length && run.token === ui.token) {
      const chunk = lane.queue.shift();
      const sentence = run.sentences[chunk.sentence];
      const received = Math.min(sentence.length, run.arrived - run.offsets[chunk.sentence]);
      chunk.requestAt = elapsed(run);
      chunk.state = "synthesising";
      try {
        const result = await api("/api/synthesize_chunk", {
          model: run.model, text: sentence.slice(0, received).join(" "), start: chunk.start, end: chunk.end,
          lookahead: chunk.lookahead, mode: chunk.mode, seed: run.seed, speaker: run.speaker,
        });
        if (run.token !== ui.token) return;
        chunk.readyAt = elapsed(run);
        Object.assign(chunk, {
          inferMs: result.infer_seconds * 1000, seconds: result.seconds, voiced: result.voiced,
          encoderText: result.encoder_text, rate: result.sample_rate, pcm: floatPcm(result.wav),
        });
        place(run, lane, chunk);
      } catch (error) {
        if (run.token !== ui.token) return;
        chunk.state = "error";
        chunk.error = error.message;
        lane.errors++;
        showError(`Strategy ${LANES[lane.index].name}: ${error.message}`);
      }
    }
    lane.busy = false;
  }

  /** Queue a chunk's audio right after the previous one, or as soon as it arrives (a stall). */
  function place(run, lane, chunk) {
    const earliest = chunk.readyAt + LEAD;
    let start = earliest;
    if (lane.firstAudio === null) {
      lane.firstAudio = start;
    } else if (earliest > lane.playEnd + 0.001) {
      lane.stalls.push({from: lane.playEnd, to: earliest, cause: chunk.textAt > lane.playEnd ? "text" : "tts"});
    } else {
      start = lane.playEnd;
    }
    chunk.startAt = start;
    chunk.endAt = start + chunk.seconds;
    chunk.state = "placed";
    lane.playEnd = chunk.endAt;
    if (lane.index === run.listen) schedule(chunk.pcm, chunk.rate, run.ctx0 + start);
  }

  function finish(run) {
    run.finished = true;
    clearInterval(run.timer);
    setProgress(progressText(run, run.end));
    lanesChanged();
  }

  function stop() {
    const run = ui.run;
    ui.token++;
    stopAudio();
    if (run && !run.finished) {
      run.finished = true;
      run.stopped = true;
      clearInterval(run.timer);
      setProgress("Stopped.");
      lanesChanged();
    }
  }

  function progressText(run, t) {
    const parts = [`LLM ${run.arrived}/${run.words.length} words`];
    for (const lane of run.lanes) {
      const first = lane.firstAudio === null ? "waiting" : `first audio ${lane.firstAudio.toFixed(2)} s`;
      parts.push(`${LANES[lane.index].name}: ${first}`);
    }
    return `${t.toFixed(1)} s · ${parts.join(" · ")}`;
  }

  // ----------------------------------------------------------------------------------- audio

  function floatPcm(b64) {
    const bytes = pcmFromWav(base64Bytes(b64));
    const ints = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
    return Float32Array.from(ints, (v) => v / 32768);
  }

  function schedule(pcm, rate, when) {
    const buffer = ui.ctx.createBuffer(1, pcm.length, rate);
    buffer.copyToChannel(pcm, 0);
    const source = ui.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ui.ctx.destination);
    source.start(Math.max(when, ui.ctx.currentTime));
    ui.sources.push(source);
  }

  function stopAudio() {
    for (const source of ui.sources) {
      try {
        source.stop();
      } catch (error) {
        // already stopped
      }
    }
    ui.sources = [];
    ui.playback = null;
  }

  /** Replay finished lanes: with the run's waits, or back to back (A/B of the audio alone). */
  function play(indices, withWaits) {
    const run = ui.run;
    if (!run || !run.finished || !ui.ctx) return;
    stopAudio();
    ui.ctx.resume();
    const start = ui.ctx.currentTime + 0.1;
    let t = start;
    const items = [];
    indices.forEach((index, n) => {
      const lane = run.lanes.find((l) => l.index === index);
      if (!lane) return;
      const placed = lane.chunks.filter((c) => c.pcm);
      if (withWaits) {
        for (const c of placed) items.push({lane, chunk: c, at: start + c.startAt, end: start + c.endAt});
      } else {
        if (n) t += AB_GAP;
        for (const c of placed) {
          items.push({lane, chunk: c, at: t, end: t + c.seconds});
          t += c.seconds;
        }
      }
    });
    if (!items.length) return;
    for (const item of items) schedule(item.chunk.pcm, item.chunk.rate, item.at);
    ui.playback = {items, start, withWaits, end: Math.max(...items.map((i) => i.end))};
    for (let i = 0; i < run.words.length; i++) revealWord(i, false);
    loop();
  }

  // ---------------------------------------------------------------------------------- bubble

  function renderBubble(run) {
    const bubble = $a("a-bubble");
    bubble.replaceChildren();
    run.words.forEach((word, i) => {
      if (i) bubble.append(" ");
      bubble.append(el("span", {class: "w pending", "data-i": i, text: word.text}));
    });
  }

  function revealWord(i, animate = true) {
    const span = $a("a-bubble").querySelector(`[data-i="${i}"]`);
    if (!span || !span.classList.contains("pending")) return;
    span.classList.remove("pending");
    if (animate) span.classList.add("new");
  }

  /** The chunk audible now: [lane, chunk] or null. */
  function audible() {
    if (ui.playback) {
      const now = ui.ctx.currentTime;
      const item = ui.playback.items.find((i) => i.at <= now && now < i.end);
      return item ? [item.lane, item.chunk] : null;
    }
    const run = ui.run;
    if (!run || run.finished) return null;
    const lane = run.lanes.find((l) => l.index === run.listen);
    if (!lane) return null;
    const t = elapsed(run);
    const chunk = lane.chunks.find((c) => c.startAt !== undefined && c.startAt <= t && t < c.endAt);
    return chunk ? [lane, chunk] : null;
  }

  function highlightBubble() {
    const now = audible();
    const bubble = $a("a-bubble");
    for (const span of bubble.querySelectorAll(".w.speaking")) span.classList.remove("speaking");
    if (!now) return;
    const [lane, chunk] = now;
    bubble.style.setProperty("--speak", color(lane.index));
    for (let i = chunk.gStart; i < chunk.gEnd; i++) {
      const span = bubble.querySelector(`[data-i="${i}"]`);
      if (span) span.classList.add("speaking");
    }
  }

  function cursorTime() {
    if (ui.playback) return ui.playback.withWaits ? ui.ctx.currentTime - ui.playback.start : null;
    if (ui.run && !ui.run.finished) return elapsed(ui.run);
    return null;
  }

  function loop() {
    cancelAnimationFrame(ui.raf);
    const frame = () => {
      draw();
      highlightBubble();
      const running = isRunning() && ui.run.token === ui.token;
      const playing = ui.playback && ui.ctx.currentTime < ui.playback.end + 0.05;
      if (running || playing) {
        ui.raf = requestAnimationFrame(frame);
      } else {
        ui.playback = null;
        draw();
        highlightBubble();
      }
    };
    ui.raf = requestAnimationFrame(frame);
  }

  // -------------------------------------------------------------------------------- timeline

  function draw() {
    const canvas = $a("a-canvas");
    if (!canvas || canvas.offsetParent === null) return;
    const run = ui.run;
    const lanes = run ? run.lanes : activeLanes().map(({strategy, index}) => ({
      index, strategy, code: codeOf(strategy), chunks: [], stalls: [], firstAudio: null, playEnd: 0}));
    const width = canvas.clientWidth;
    const height = LLM_ROW + lanes.length * LANE_ROW + AXIS_ROW + 6;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const ink = cssVar("--ink");
    const ink2 = cssVar("--ink-2");
    const ink3 = cssVar("--ink-3");
    const rule = cssVar("--rule");
    const stallColor = cssVar("--stall");
    const now = cursorTime();
    let xMax = 8;
    if (run) {
      const estimate = run.finished ? 0
        : run.ttft + run.words.length / run.rate + run.words.length * SECONDS_PER_WORD + 0.5;
      xMax = Math.max(estimate, ...run.lanes.map((l) => l.playEnd + 0.3), (now ?? 0) + 0.3);
    }
    const plotW = width - GUTTER - 10;
    const x = (t) => GUTTER + (t / xMax) * plotW;
    ui.hits = [];
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    // LLM text row
    ctx.fillStyle = ink2;
    ctx.textAlign = "left";
    ctx.fillText("LLM text", 8, LLM_ROW / 2);
    ctx.fillStyle = rule;
    ctx.fillRect(GUTTER, LLM_ROW - 4, plotW, 1);
    if (run) {
      let lastLabelEnd = -Infinity;
      run.words.forEach((word, i) => {
        if (i >= run.arrived && !run.finished) return;
        const px = x(word.at);
        const sentenceStart = word.index === 0;
        ctx.fillStyle = sentenceStart ? ink : ink2;
        ctx.fillRect(px - 0.5, sentenceStart ? 4 : 12, sentenceStart ? 2 : 1, sentenceStart ? LLM_ROW - 8 : LLM_ROW - 16);
        const labelW = ctx.measureText(word.text).width;
        if (px + 3 > lastLabelEnd + 4 && px + 3 + labelW < width) {
          ctx.fillStyle = ink3;
          ctx.fillText(word.text, px + 3, 10);
          lastLabelEnd = px + 3 + labelW;
        }
        ui.hits.push({x0: px - 3, x1: px + 3, y0: 0, y1: LLM_ROW,
          text: `"${word.text}" arrives at ${word.at.toFixed(2)} s (word ${i + 1})`});
      });
    }

    // Lanes
    lanes.forEach((lane, n) => {
      const y = LLM_ROW + n * LANE_ROW;
      const laneColor = color(lane.index);
      const callY = y + 8;
      const playY = y + 22;
      const playH = 22;
      ctx.fillStyle = laneColor;
      ctx.fillRect(8, y + 9, 10, 10);
      ctx.fillStyle = ink;
      ctx.textAlign = "left";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText(`${LANES[lane.index].name} ${lane.code}`, 24, y + 14);
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = ink2;
      const first = lane.firstAudio === null ? "first audio –" : `first audio ${lane.firstAudio.toFixed(2)} s`;
      ctx.fillText(first, 8, y + 30);
      const stall = lane.stalls.reduce((s, st) => s + (st.to - st.from), 0);
      ctx.fillStyle = stall > 0.0005 ? stallColor : ink2;
      ctx.fillText(`stalls ${stall.toFixed(2)} s`, 8, y + 44);
      ctx.fillStyle = rule;
      ctx.fillRect(GUTTER, y + LANE_ROW - 6, plotW, 1);
      if (!run) return;

      // Silence before the first audio.
      const waitEnd = lane.firstAudio ?? (now ?? 0);
      ctx.fillStyle = hexAlpha(ink3, 0.14);
      ctx.fillRect(x(0), playY, x(waitEnd) - x(0), playH);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = hexAlpha(ink3, 0.6);
      ctx.strokeRect(x(0) + 0.5, playY + 0.5, Math.max(0, x(waitEnd) - x(0) - 1), playH - 1);
      ctx.setLineDash([]);

      // TTS calls.
      for (const chunk of lane.chunks) {
        if (chunk.requestAt === undefined) continue;
        const end = chunk.readyAt ?? now ?? chunk.requestAt;
        ctx.fillStyle = chunk.state === "error" ? stallColor : ink3;
        const x0 = x(chunk.requestAt);
        ctx.fillRect(x0, callY, Math.max(2, x(end) - x0), 8);
        if (chunk.readyAt !== undefined) {
          ui.hits.push({x0: x0 - 1, x1: Math.max(x0 + 3, x(end)), y0: callY - 2, y1: callY + 10,
            text: `TTS call for "${chunk.voiced}": requested ${chunk.requestAt.toFixed(3)} s, audio back ` +
              `${chunk.readyAt.toFixed(3)} s (synthesis ${chunk.inferMs.toFixed(0)} ms)`});
        }
      }

      // Audio.
      lane.chunks.forEach((chunk, i) => {
        if (chunk.startAt === undefined) return;
        const x0 = x(chunk.startAt);
        const x1 = x(chunk.endAt);
        const future = now !== null && chunk.startAt > now;
        ctx.fillStyle = hexAlpha(laneColor, future ? 0.3 : i % 2 ? 0.72 : 1);
        ctx.fillRect(x0, playY, Math.max(1, x1 - x0 - 1), playH);
        const label = chunk.voiced;
        ctx.font = "10px system-ui, sans-serif";
        if (ctx.measureText(label).width + 6 < x1 - x0) {
          ctx.fillStyle = "#fff";
          ctx.textAlign = "left";
          ctx.fillText(label, x0 + 3, playY + playH / 2);
        }
        ctx.font = "11px system-ui, sans-serif";
        ui.hits.push({x0, x1, y0: playY, y1: playY + playH,
          text: `"${chunk.voiced}" plays ${chunk.startAt.toFixed(2)}–${chunk.endAt.toFixed(2)} s · encoder read: ` +
            `"${chunk.encoderText}"`});
      });

      // Stalls.
      for (const st of lane.stalls) {
        const x0 = x(st.from);
        const x1 = x(st.to);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0, playY, Math.max(2, x1 - x0), playH);
        ctx.clip();
        ctx.strokeStyle = stallColor;
        ctx.lineWidth = 1;
        for (let hx = x0 - playH; hx < x1 + playH; hx += 5) {
          ctx.beginPath();
          ctx.moveTo(hx, playY + playH);
          ctx.lineTo(hx + playH, playY);
          ctx.stroke();
        }
        ctx.restore();
        ctx.strokeStyle = stallColor;
        ctx.strokeRect(x0 + 0.5, playY + 0.5, Math.max(1, x1 - x0 - 1), playH - 1);
        ui.hits.push({x0, x1: Math.max(x0 + 3, x1), y0: playY, y1: playY + playH,
          text: `Stall ${(st.to - st.from).toFixed(2)} s: ${st.cause === "text"
            ? "the next words had not arrived from the LLM" : "the next chunk was still being synthesised"}`});
      }

      // First-audio marker.
      if (lane.firstAudio !== null) {
        const fx = x(lane.firstAudio);
        ctx.fillStyle = laneColor;
        ctx.beginPath();
        ctx.moveTo(fx - 5, playY - 7);
        ctx.lineTo(fx + 5, playY - 7);
        ctx.lineTo(fx, playY - 1);
        ctx.fill();
      }
    });

    // Time axis.
    const axisY = LLM_ROW + lanes.length * LANE_ROW;
    const pxPerSec = plotW / xMax;
    const tickStep = pxPerSec > 120 ? 0.5 : pxPerSec > 40 ? 1 : pxPerSec > 18 ? 2 : 5;
    ctx.fillStyle = ink3;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let s = 0; s <= xMax + 1e-9; s += tickStep) {
      ctx.fillRect(x(s), axisY, 1, 4);
      ctx.fillText(`${s % 1 ? s.toFixed(1) : s}s`, x(s), axisY + 5);
    }
    ctx.textAlign = "left";
    ctx.fillText("time since sent", 8, axisY + 5);

    if (now !== null) {
      ctx.fillStyle = ink;
      ctx.fillRect(x(now) - 0.75, 0, 1.5, axisY);
    }
    $a("a-timeline-caption").textContent = run
      ? `${run.words.length} words in ${run.sentences.length} sentence${run.sentences.length > 1 ? "s" : ""} · ` +
        `first token ${Math.round(run.ttft * 1000)} ms · ${run.rate} words/s · ${run.model}` +
        (run.stopped ? " · stopped" : "")
      : "Press Send to stream the reply.";
  }

  function hexAlpha(colorValue, alpha) {
    const c = colorValue.trim();
    if (c.startsWith("#") && (c.length === 7 || c.length === 4)) {
      const full = c.length === 4 ? `#${c[1]}${c[1]}${c[2]}${c[2]}${c[3]}${c[3]}` : c;
      const r = parseInt(full.slice(1, 3), 16);
      const g = parseInt(full.slice(3, 5), 16);
      const b = parseInt(full.slice(5, 7), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    return c;
  }

  function hover(event) {
    const rect = event.target.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const tooltip = $("tooltip");
    const hit = ui.hits.slice().reverse().find((h) => px >= h.x0 && px <= h.x1 && py >= h.y0 && py <= h.y1);
    if (!hit) {
      tooltip.hidden = true;
      return;
    }
    tooltip.textContent = hit.text;
    tooltip.hidden = false;
    const left = Math.min(event.clientX + 12, window.innerWidth - tooltip.offsetWidth - 8);
    tooltip.style.left = `${Math.max(4, left)}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  }

  // ----------------------------------------------------------------------------------- cards

  function renderCards() {
    const box = $a("a-cards");
    const run = ui.run;
    const full = ui.study && ui.study.conditions.full;
    const lanes = run ? run.lanes : activeLanes().map(({strategy, index}) => ({index, strategy, code: codeOf(strategy)}));
    box.replaceChildren(...lanes.map((lane) => {
      const card = el("article", {class: "result"});
      card.style.setProperty("--lane", `var(${LANES[lane.index].color})`);
      card.append(el("h4", {}, `${LANES[lane.index].name}: ${nameOf(lane.strategy)}`, el("code", {text: lane.code})));
      const placed = run ? lane.chunks.filter((c) => c.pcm) : [];
      const done = run && run.finished;
      const stall = run ? lane.stalls.reduce((s, st) => s + (st.to - st.from), 0) : 0;
      const stats = el("div", {class: "stats"});
      const stat = (value, label, bad = false) => el("div", {class: `stat${bad ? " bad" : ""}`},
        el("div", {class: "v", text: value}), el("div", {class: "l", text: label}));
      stats.append(
        stat(run && lane.firstAudio !== null ? `${lane.firstAudio.toFixed(2)} s` : "–", "time to first audio"),
        stat(run ? `${stall.toFixed(2)} s` : "–", `total stall${run && lane.stalls.length ? ` (${lane.stalls.length})` : ""}`,
          stall > 0.0005),
        stat(done && placed.length ? `${lane.playEnd.toFixed(2)} s` : "–", "finished speaking"));
      card.append(stats);
      if (placed.length) {
        const meanInfer = placed.reduce((s, c) => s + c.inferMs, 0) / placed.length;
        const meanTrip = placed.reduce((s, c) => s + (c.readyAt - c.requestAt), 0) / placed.length;
        const audio = placed.reduce((s, c) => s + c.seconds, 0);
        card.append(el("div", {class: "calls", text: `${placed.length} TTS call${placed.length > 1 ? "s" : ""} · ` +
          `synthesis ${meanInfer.toFixed(0)} ms mean · round trip ${(meanTrip * 1000).toFixed(0)} ms · ` +
          `${audio.toFixed(1)} s of audio${lane.errors ? ` · ${lane.errors} failed` : ""}`}));
      }
      const condition = ui.study && ui.study.conditions[lane.code];
      if (condition) {
        const line = el("div", {class: "studyline"});
        line.append("Study WER ", el("strong", {text: pct(condition.wer)}));
        if (lane.code !== "full" && full) {
          const diff = 100 * (condition.wer - full.wer);
          const [lo, hi] = condition.vs_full_ci95.map((v) => 100 * v);
          line.append(` · ${diff >= 0 ? "+" : ""}${diff.toFixed(1)} points vs whole sentence [${signed(lo)}, ${signed(hi)}]`);
        }
        line.append(` · waits ${condition.wait_words.toFixed(1)} words on average`);
        card.append(line);
      } else {
        card.append(el("div", {class: "studyline", text: "No study WER for this strategy."}));
      }
      if (done && placed.length) {
        const playButton = el("button", {type: "button", text: "Play audio"});
        playButton.addEventListener("click", () => play([lane.index], false));
        const replay = el("button", {type: "button", text: "Replay with the waits"});
        replay.addEventListener("click", () => play([lane.index], true));
        card.append(el("div", {class: "buttons"}, playButton, replay));
      }
      return card;
    }));
    const note = $a("a-study-note");
    if (ui.study) {
      note.textContent = `Study WER: ${ui.study.label} (${relative(ui.study.analysis)}; ${ui.study.source}). ` +
        "Measured offline with Whisper, not computed from this run. Real recordings of the same sentences: " +
        `${pct(ui.study.natural_wer)}.` + (ui.studyFallback
        ? ` No chunk study exists for the selected checkpoint; these figures are for "${ui.studyFallback}".` : "");
    } else {
      note.textContent = "No chunk study is configured (chunk_studies in webui/config.json).";
    }
  }

  function signed(v) {
    return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
  }

  function relative(path) {
    const i = path.replace(/\\/g, "/").search(/docs\/records|output\//);
    return i >= 0 ? path.replace(/\\/g, "/").slice(i) : path;
  }

  // ---------------------------------------------------------------------------------- charts

  function svg(tag, attrs = {}, ...children) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    node.append(...children);
    return node;
  }

  function renderTradeoff() {
    const box = $a("a-tradeoff");
    box.replaceChildren();
    const study = ui.study;
    if (!study) {
      $a("a-tradeoff-note").textContent = "No chunk study is configured.";
      return;
    }
    const W = 560, H = 320, m = {l: 46, r: 14, t: 14, b: 40};
    const conditions = study.conditions;
    const maxWait = Math.max(...Object.values(conditions).map((c) => c.wait_words)) + 0.8;
    const maxWer = Math.max(...Object.values(conditions).map((c) => c.wer)) * 100;
    const yMax = Math.ceil((maxWer + 4) / 10) * 10;
    const X = (w) => m.l + (w / maxWait) * (W - m.l - m.r);
    const Y = (p) => H - m.b - (p / yMax) * (H - m.t - m.b);
    const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Whisper WER against the words the TTS waits for, from the chunk study"});
    const grid = svg("g", {class: "grid"});
    for (let p = 0; p <= yMax; p += 10) {
      grid.append(svg("line", {x1: m.l, x2: W - m.r, y1: Y(p), y2: Y(p)}));
      root.append(svg("text", {x: m.l - 6, y: Y(p) + 4, "text-anchor": "end", text: `${p}%`}));
    }
    for (let w = 0; w <= maxWait; w += 2) {
      root.append(svg("text", {x: X(w), y: H - m.b + 16, "text-anchor": "middle", text: String(w)}));
    }
    root.prepend(grid);
    root.append(svg("text", {x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle",
      text: "words the TTS waits for before speaking (mean over the study's sentences)"}));
    root.append(svg("text", {x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle",
      transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, text: "Whisper WER"}));
    if (study.natural_wer !== null && study.natural_wer !== undefined) {
      const ny = Y(100 * study.natural_wer);
      root.append(svg("line", {x1: m.l, x2: W - m.r, y1: ny, y2: ny, style: `stroke: ${cssVar("--good")}`,
        "stroke-dasharray": "4 3"}));
      root.append(svg("text", {x: W - m.r - 4, y: ny - 5, "text-anchor": "end", text: "real recordings"}));
    }
    const series = [
      {name: "independent chunks", match: /^naive_/, dash: "5 4", shape: "circle"},
      {name: "context-aware, lookahead 0", match: /^ctx_\d+_k0$/, dash: "1 4", shape: "square"},
      {name: "context-aware, lookahead 2", match: /^ctx_\d+_k2$/, dash: "", shape: "square"},
      {name: "context-aware, lookahead 4", match: /^ctx_\d+_k4$/, dash: "9 3", shape: "square"},
    ];
    const ink2 = cssVar("--ink-2");
    const ink3 = cssVar("--ink-3");
    series.forEach((s, si) => {
      const points = Object.entries(conditions).filter(([code]) => s.match.test(code))
        .map(([code, c]) => ({code, x: X(c.wait_words), y: Y(100 * c.wer)})).sort((a, b) => a.x - b.x);
      if (!points.length) return;
      root.append(svg("polyline", {points: points.map((p) => `${p.x},${p.y}`).join(" "), fill: "none",
        style: `stroke: ${si === 0 ? ink3 : ink2}`, "stroke-width": 1.5, "stroke-dasharray": s.dash}));
      for (const p of points) {
        const title = svg("title", {text: `${p.code}: WER ${pct(conditions[p.code].wer)}, waits ` +
          `${conditions[p.code].wait_words.toFixed(1)} words`});
        root.append(s.shape === "circle"
          ? svg("circle", {cx: p.x, cy: p.y, r: 3.5, style: `fill: ${cssVar("--surface")}; stroke: ${ink3}`}, title)
          : svg("rect", {x: p.x - 3.5, y: p.y - 3.5, width: 7, height: 7, style: `fill: ${ink2}`}, title));
      }
      const ly = m.t + 6 + si * 15;
      root.append(svg("line", {x1: W - m.r - 190, x2: W - m.r - 166, y1: ly, y2: ly,
        style: `stroke: ${si === 0 ? ink3 : ink2}`, "stroke-width": 1.5, "stroke-dasharray": s.dash}));
      root.append(svg("text", {x: W - m.r - 160, y: ly + 4, text: s.name}));
    });
    if (conditions.full) {
      const fx = X(conditions.full.wait_words);
      const fy = Y(100 * conditions.full.wer);
      root.append(svg("path", {d: star(fx, fy, 8, 3.5), style: `fill: ${cssVar("--ink")}`},
        svg("title", {text: `whole sentence: WER ${pct(conditions.full.wer)}`})));
      root.append(svg("text", {x: fx, y: fy - 12, "text-anchor": "middle", text: "whole sentence"}));
    }
    for (const {strategy, index} of activeLanes()) {
      const c = conditions[codeOf(strategy)];
      if (!c) continue;
      const cx = X(c.wait_words);
      const cy = Y(100 * c.wer);
      root.append(svg("circle", {cx, cy, r: 9, fill: "none", style: `stroke: ${color(index)}`, "stroke-width": 2.5}));
      root.append(svg("text", {x: cx + 12, y: cy + 4, style: `fill: ${color(index)}; font-weight: 700`,
        text: LANES[index].name}));
    }
    box.append(root);
    $a("a-tradeoff-note").textContent = `${study.label}. Circles A–C mark the strategies above. ` +
      (ui.studyFallback ? `Shown for "${ui.studyFallback}"; the selected checkpoint has no chunk study.` : "");
  }

  function star(cx, cy, outer, inner) {
    const points = [];
    for (let i = 0; i < 10; i++) {
      const r = i % 2 ? inner : outer;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      points.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
    }
    return `M${points.join("L")}Z`;
  }

  function renderSpeed() {
    const box = $a("a-speed-chart");
    box.replaceChildren();
    const length = Math.max(1, Math.min(60, Math.floor(Number($a("a-length").value) || 12)));
    const rate = rateValue();
    const W = 560, H = 260, m = {l: 46, r: 14, t: 12, b: 40};
    const yMax = length / RATE_MIN;
    const lx = (r) => m.l + (Math.log(r / RATE_MIN) / Math.log(RATE_MAX / RATE_MIN)) * (W - m.l - m.r);
    const Y = (s) => H - m.b - (s / yMax) * (H - m.t - m.b);
    const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Waiting for text before the first audio, against LLM speed"});
    const grid = svg("g", {class: "grid"});
    const yStep = yMax > 8 ? 2 : yMax > 4 ? 1 : 0.5;
    for (let s = 0; s <= yMax + 1e-9; s += yStep) {
      grid.append(svg("line", {x1: m.l, x2: W - m.r, y1: Y(s), y2: Y(s)}));
      root.append(svg("text", {x: m.l - 6, y: Y(s) + 4, "text-anchor": "end", text: `${s}s`}));
    }
    for (const r of [2, 5, 10, 20, 50]) {
      grid.append(svg("line", {x1: lx(r), x2: lx(r), y1: m.t, y2: H - m.b}));
      root.append(svg("text", {x: lx(r), y: H - m.b + 16, "text-anchor": "middle", text: String(r)}));
    }
    root.append(grid);
    root.append(svg("text", {x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", text: "LLM speed (words/s)"}));
    const curves = [{label: "whole sentence", strategy: {kind: "full"}, stroke: cssVar("--ink-3"), dash: "4 3"}];
    for (const {strategy, index} of activeLanes()) {
      if (strategy.kind === "full") curves[0] = {label: `${LANES[index].name} whole sentence`, strategy,
        stroke: color(index), dash: ""};
      else curves.push({label: `${LANES[index].name} ${codeOf(strategy)}`, strategy, stroke: color(index), dash: ""});
    }
    for (const curve of curves) {
      const need = wordsNeeded(curve.strategy, length);
      const points = [];
      for (let i = 0; i <= 60; i++) {
        const r = RATE_MIN * Math.pow(RATE_MAX / RATE_MIN, i / 60);
        points.push(`${lx(r).toFixed(1)},${Y(need / r).toFixed(1)}`);
      }
      root.append(svg("polyline", {points: points.join(" "), fill: "none", style: `stroke: ${curve.stroke}`,
        "stroke-width": 2, "stroke-dasharray": curve.dash}));
      const labelRate = 2.3;
      root.append(svg("text", {x: lx(labelRate) + 4, y: Y(need / labelRate) - 6, style: `fill: ${curve.stroke}`,
        text: curve.label}));
      root.append(svg("circle", {cx: lx(rate), cy: Y(need / rate), r: 3.5, style: `fill: ${curve.stroke}`},
        svg("title", {text: `${curve.label}: ${(need / rate).toFixed(2)} s at ${rate} words/s`})));
    }
    root.append(svg("line", {x1: lx(rate), x2: lx(rate), y1: m.t, y2: H - m.b, style: `stroke: ${cssVar("--ink")}`,
      "stroke-width": 1}));
    box.append(root);

    const fullWait = (r) => length / r;
    const text = $a("a-speed-text");
    text.replaceChildren(`${length}-word sentence at `, el("strong", {text: `${rate} words/s`}), ": whole sentence waits ",
      el("strong", {text: `${fullWait(rate).toFixed(2)} s`}), " for its text");
    for (const {strategy, index} of activeLanes()) {
      if (strategy.kind === "full") continue;
      const need = wordsNeeded(strategy, length);
      text.append(`; ${LANES[index].name} ${codeOf(strategy)} `, el("strong", {text: `${(need / rate).toFixed(2)} s`}),
        `, ${(fullWait(rate) - need / rate).toFixed(2)} s sooner (at 10 / 50 words/s: ` +
        `${(fullWait(10) - need / 10).toFixed(2)} / ${(fullWait(50) - need / 50).toFixed(2)} s sooner)`);
    }
    text.append(". The saving is largest for slow LLMs and long sentences.");
  }

  document.addEventListener("sts-status", (event) => setup(event.detail));
  document.addEventListener("sts-tab", (event) => {
    if (event.detail === "agent") {
      draw();
    } else {
      stop();
    }
  });
})();
