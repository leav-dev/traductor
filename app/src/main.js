const { invoke } = window.__TAURI__.core;

const statusEl = document.querySelector("#status");
const italianEl = document.querySelector("#italian");
const frenchEl = document.querySelector("#french");
const recordBig = document.querySelector("#record-btn");
const micMini = document.querySelector("#mic-mini");
const replayBtn = document.querySelector("#replay-btn");
const swapBtn = document.querySelector("#swap-btn");

const HINT = "Tap to speak · or type above";
let recording = false;
let busy = false;
let outputMode = "text"; // "text" (show) or "voice" (speak, hide text)
let lastFrench = "";

// deliver the French result according to the chosen OUTPUT mode
async function deliverFrench(fr) {
  lastFrench = fr || "";
  if (outputMode === "voice") {
    frenchEl.textContent = "";
    if (fr) {
      statusEl.textContent = "Speaking…";
      await invoke("speak", { text: fr });
    }
    statusEl.textContent = HINT;
  } else {
    frenchEl.textContent = fr || "—";
    statusEl.textContent = fr ? "Done · tap the speaker to hear it" : HINT;
  }
}

// ---- voice input: record -> interpret -> deliver ----
async function toggleRecord() {
  if (busy) return;
  if (!recording) {
    try {
      await invoke("start_recording");
      recording = true;
      recordBig.classList.add("recording");
      statusEl.textContent = "Listening… tap again to stop";
      italianEl.textContent = "";
      frenchEl.textContent = "";
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
      const t = await invoke("interpret", { path });
      italianEl.textContent = t.italian || "—";
      await deliverFrench(t.french);
    } catch (e) {
      statusEl.textContent = "Error: " + e;
    } finally {
      busy = false;
    }
  }
}

recordBig.addEventListener("click", toggleRecord);
micMini.addEventListener("click", toggleRecord);

// ---- text input: type Italian + Enter -> translate -> deliver ----
italianEl.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter" || e.shiftKey) return;
  e.preventDefault();
  const txt = italianEl.textContent.trim();
  if (!txt || busy) return;
  busy = true;
  italianEl.blur();
  statusEl.textContent = "Translating…";
  frenchEl.textContent = "";
  try {
    const fr = await invoke("translate", { text: txt });
    await deliverFrench(fr);
  } catch (err) {
    statusEl.textContent = "Error: " + err;
  } finally {
    busy = false;
  }
});

// ---- replay the French out loud (works in either mode) ----
replayBtn.addEventListener("click", async () => {
  const fr = (lastFrench || frenchEl.textContent).trim();
  if (!fr || busy) return;
  statusEl.textContent = "Speaking…";
  try {
    await invoke("speak", { text: fr });
    statusEl.textContent = HINT;
  } catch (e) {
    statusEl.textContent = "Error: " + e;
  }
});

swapBtn.addEventListener("click", () => {
  statusEl.textContent = "Italian → French only, for now.";
});

// ---- INPUT toggle: keyboard focuses the field, mic records ----
document.querySelector("#in-keyboard").addEventListener("click", () => italianEl.focus());
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