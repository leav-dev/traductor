const { invoke } = window.__TAURI__.core;

const statusEl = document.querySelector("#status");
const srcTextEl = document.querySelector("#src-text");
const tgtTextEl = document.querySelector("#tgt-text");
const srcLabelEl = document.querySelector("#src-label");
const tgtLabelEl = document.querySelector("#tgt-label");
const srcDD = document.querySelector("#src-dd");
const tgtDD = document.querySelector("#tgt-dd");
const recordBig = document.querySelector("#record-btn");
const micMini = document.querySelector("#mic-mini");
const replayBtn = document.querySelector("#replay-btn");
const swapBtn = document.querySelector("#swap-btn");

const NAMES = { en: "English", fr: "Français", es: "Español", it: "Italiano" };
const HINT = "Tap to speak · or type above";

let srcLang = "it";
let tgtLang = "fr";
let outputMode = "text";   // "text" (show) or "voice" (speak, hide text)
let lastOut = "";
let recording = false;
let busy = false;

// ---- custom language dropdowns (fully styled, keyboard-navigable) ----
function renderDD(dd, lang) {
  dd.querySelector(".lang-current").textContent = NAMES[lang];
  dd.querySelectorAll("[role=option]").forEach((li) => {
    li.classList.toggle("selected", li.dataset.lang === lang);
  });
}
function syncLangUI() {
  renderDD(srcDD, srcLang);
  renderDD(tgtDD, tgtLang);
  srcLabelEl.textContent = NAMES[srcLang].toUpperCase();
  tgtLabelEl.textContent = NAMES[tgtLang].toUpperCase();
}
function closeDropdowns() {
  for (const dd of [srcDD, tgtDD]) {
    if (!dd.classList.contains("open")) continue;
    dd.classList.remove("open");
    const trigger = dd.querySelector(".lang-trigger");
    trigger.setAttribute("aria-expanded", "false");
    trigger.removeAttribute("aria-activedescendant");
    dd.querySelectorAll("[role=option]").forEach((o) => o.classList.remove("active"));
  }
}
// keep source and target different
function setSource(lang) {
  if (lang === tgtLang) tgtLang = srcLang;
  srcLang = lang;
  syncLangUI();
}
function setTarget(lang) {
  if (lang === srcLang) srcLang = tgtLang;
  tgtLang = lang;
  syncLangUI();
}
function wireDropdown(dd, isSource) {
  const trigger = dd.querySelector(".lang-trigger");
  const options = [...dd.querySelectorAll("[role=option]")];
  options.forEach((o, i) => { o.id = dd.id + "-opt-" + i; });

  const activeIndex = () => (typeof dd._active === "number" ? dd._active : -1);
  const highlight = (i) => {
    options.forEach((o, k) => o.classList.toggle("active", k === i));
    if (i >= 0) trigger.setAttribute("aria-activedescendant", options[i].id);
    else trigger.removeAttribute("aria-activedescendant");
    dd._active = i;
  };
  const open = () => {
    closeDropdowns();
    dd.classList.add("open");
    trigger.setAttribute("aria-expanded", "true");
    highlight(options.findIndex((o) => o.classList.contains("selected")));
  };
  const choose = (i) => {
    if (i < 0 || i >= options.length) return;
    if (isSource) setSource(options[i].dataset.lang);
    else setTarget(options[i].dataset.lang);
    closeDropdowns();
  };

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();          // keep the document handler from seeing trigger clicks
    if (e.detail === 0) return;   // keyboard-synthesized click: handled in keydown
    dd.classList.contains("open") ? closeDropdowns() : open();
  });
  options.forEach((li, i) => {
    li.addEventListener("click", (e) => { e.stopPropagation(); choose(i); });
    li.addEventListener("mousemove", () => highlight(i));
  });
  trigger.addEventListener("keydown", (e) => {
    const isOpen = dd.classList.contains("open");
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        isOpen ? highlight(Math.min(activeIndex() + 1, options.length - 1)) : open();
        break;
      case "ArrowUp":
        e.preventDefault();
        isOpen ? highlight(Math.max(activeIndex() - 1, 0)) : open();
        break;
      case "Home":
        if (isOpen) { e.preventDefault(); highlight(0); }
        break;
      case "End":
        if (isOpen) { e.preventDefault(); highlight(options.length - 1); }
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        isOpen ? choose(activeIndex()) : open();
        break;
      case "Escape":
        if (isOpen) { e.preventDefault(); closeDropdowns(); }
        break;
      case "Tab":
        if (isOpen) closeDropdowns();
        break;
    }
  });
}
wireDropdown(srcDD, true);
wireDropdown(tgtDD, false);
document.addEventListener("click", closeDropdowns);

swapBtn.addEventListener("click", () => {
  [srcLang, tgtLang] = [tgtLang, srcLang];
  syncLangUI();
  srcTextEl.textContent = "";
  tgtTextEl.textContent = "";
  lastOut = "";
  statusEl.textContent = HINT;
});

// ---- deliver the target text according to the OUTPUT mode ----
async function deliverTarget(out) {
  lastOut = out || "";
  if (outputMode === "voice") {
    tgtTextEl.textContent = "";
    if (out) {
      statusEl.textContent = "Speaking…";
      await invoke("speak", { text: out, lang: tgtLang });
    }
    statusEl.textContent = HINT;
  } else {
    tgtTextEl.textContent = out || "—";
    statusEl.textContent = out ? "Done · tap the speaker to hear it" : HINT;
  }
}

// ---- text input: type source + Enter -> translate (routed) -> deliver ----
srcTextEl.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter" || e.shiftKey) return;
  e.preventDefault();
  const txt = srcTextEl.textContent.trim();
  if (!txt || busy) return;
  busy = true;
  srcTextEl.blur();
  statusEl.textContent = "Translating…";
  tgtTextEl.textContent = "";
  try {
    const out = await invoke("translate", { text: txt, source: srcLang, target: tgtLang });
    await deliverTarget(out);
  } catch (err) {
    statusEl.textContent = "Error: " + err;
  } finally {
    busy = false;
  }
});

// ---- voice input: transcribe (source lang) then translate (routed) ----
async function toggleRecord() {
  if (busy) return;
  if (!recording) {
    try {
      await invoke("start_recording");
      recording = true;
      recordBig.classList.add("recording");
      statusEl.textContent = "Listening… tap again to stop";
      srcTextEl.textContent = "";
      tgtTextEl.textContent = "";
    } catch (e) {
      statusEl.textContent = "Error: " + e;
    }
  } else {
    recording = false;
    busy = true;
    recordBig.classList.remove("recording");
    statusEl.textContent = "Transcribing…";
    try {
      const path = await invoke("stop_recording");
      const heard = await invoke("transcribe", { path, lang: srcLang });
      srcTextEl.textContent = heard || "—";
      if (heard) {
        statusEl.textContent = "Translating…";
        const out = await invoke("translate", { text: heard, source: srcLang, target: tgtLang });
        await deliverTarget(out);
      } else {
        statusEl.textContent = HINT;
      }
    } catch (e) {
      statusEl.textContent = "Error: " + e;
    } finally {
      busy = false;
    }
  }
}
recordBig.addEventListener("click", toggleRecord);
micMini.addEventListener("click", toggleRecord);

// ---- replay the target out loud (works in either mode) ----
replayBtn.addEventListener("click", async () => {
  const out = (lastOut || tgtTextEl.textContent).trim();
  if (!out || busy) return;
  statusEl.textContent = "Speaking…";
  try {
    await invoke("speak", { text: out, lang: tgtLang });
    statusEl.textContent = HINT;
  } catch (e) {
    statusEl.textContent = "Error: " + e;
  }
});

// ---- INPUT toggle: keyboard focuses the field, mic records ----
document.querySelector("#in-keyboard").addEventListener("click", () => srcTextEl.focus());
document.querySelector("#in-mic").addEventListener("click", () => toggleRecord());

// ---- OUTPUT toggle: text (show) vs voice (speak, hide) ----
document.querySelector("#out-text").addEventListener("click", () => { outputMode = "text"; });
document.querySelector("#out-voice").addEventListener("click", () => { outputMode = "voice"; });

// visual active state for the toggle segments
document.querySelectorAll(".toggle").forEach((group) => {
  group.querySelectorAll(".seg").forEach((seg) => {
    seg.addEventListener("click", () => {
      group.querySelectorAll(".seg").forEach((s) => s.classList.remove("active"));
      seg.classList.add("active");
    });
  });
});

syncLangUI();
// ---- Audio device management ----
const { invoke } = window.__TAURI__.core;

async function loadDevices() {
  try {
    const inputs = await invoke("list_input_devices");
    const outputs = await invoke("list_output_devices");
    const inSelect = document.getElementById("input-device");
    const outSelect = document.getElementById("output-device");
    if (inSelect) {
      inSelect.innerHTML = "";
      inputs.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        inSelect.appendChild(opt);
      });
    }
    if (outSelect) {
      outSelect.innerHTML = "";
      outputs.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        outSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.error("Failed to load devices:", e);
  }
}

async function detectVirtual() {
  try {
    const virtual = await invoke("detect_virtual_device");
    const status = document.getElementById("virtual-status");
    if (virtual) {
      status.textContent = "Found: " + virtual;
      const outSelect = document.getElementById("output-device");
      if (outSelect) {
        for (const opt of outSelect.options) {
          if (opt.value === virtual) {
            opt.selected = true;
            break;
          }
        }
      }
    } else {
      status.textContent = "No virtual device found. Install VB-Cable (Windows), BlackHole (macOS), or PulseAudio/PipeWire (Linux).";
    }
  } catch (e) {
    console.error("Failed to detect virtual device:", e);
  }
}

// Load devices on startup
document.addEventListener("DOMContentLoaded", () => {
  loadDevices();
  const detectBtn = document.getElementById("detect-virtual-btn");
  if (detectBtn) {
    detectBtn.addEventListener("click", detectVirtual);
  }
});

// ---- Voice profiles for TTS voice cloning ----
let isCalibrating = false;

async function loadVoiceProfiles() {
  try {
    const samples = await invoke("list_voice_samples");
    const select = document.getElementById("voice-select");
    if (!select) return;
    select.innerHTML = "";
    samples.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
    // Select the active one
    const active = await invoke("get_selected_voice");
    if (active) {
      for (const opt of select.options) {
        if (opt.value === active) {
          opt.selected = true;
          break;
        }
      }
    }
  } catch (e) {
    console.error("Failed to load voice profiles:", e);
  }
}

async function startCalibration() {
  const btn = document.getElementById("calibrate-btn");
  const status = document.getElementById("calibrate-status");
  const nameInput = document.getElementById("voice-name");
  if (isCalibrating) {
    isCalibrating = false;
    btn.textContent = "Processing...";
    btn.disabled = true;
    try {
      await invoke("stop_recording");
      const name = (nameInput?.value || "default").trim();
      const path = await invoke("calibrate_voice", { name });
      status.textContent = "Saved: " + path;
      await loadVoiceProfiles();
    } catch (e) {
      status.textContent = "Error: " + e;
    }
    btn.textContent = "Record new sample (5-30s)";
    btn.disabled = false;
  } else {
    isCalibrating = true;
    btn.textContent = "Stop recording";
    status.textContent = "Recording... speak for 5-30 seconds";
    try {
      const device = document.getElementById("input-device")?.value;
      await invoke("start_recording", { deviceName: device || null });
    } catch (e) {
      status.textContent = "Error: " + e;
      isCalibrating = false;
      btn.textContent = "Record new sample (5-30s)";
    }
  }
}

async function deleteSelectedVoice() {
  const select = document.getElementById("voice-select");
  if (!select || !select.value) return;
  try {
    await invoke("delete_voice_sample", { name: select.value });
    await loadVoiceProfiles();
    document.getElementById("calibrate-status").textContent = "Deleted: " + select.value;
  } catch (e) {
    document.getElementById("calibrate-status").textContent = "Error: " + e;
  }
}

async function onVoiceSelectChange() {
  const select = document.getElementById("voice-select");
  if (!select || !select.value) return;
  try {
    await invoke("select_voice_sample", { name: select.value });
    document.getElementById("calibrate-status").textContent = "Active: " + select.value;
  } catch (e) {
    document.getElementById("calibrate-status").textContent = "Error: " + e;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadVoiceProfiles();
  const calibrateBtn = document.getElementById("calibrate-btn");
  if (calibrateBtn) calibrateBtn.addEventListener("click", startCalibration);
  const deleteBtn = document.getElementById("delete-voice-btn");
  if (deleteBtn) deleteBtn.addEventListener("click", deleteSelectedVoice);
  const voiceSelect = document.getElementById("voice-select");
  if (voiceSelect) voiceSelect.addEventListener("change", onVoiceSelectChange);
});
