use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tauri::Manager;

// EDIT this to your repo's absolute path (run `pwd` at the repo root):
// Resolve repo path: env var SUSSURRO_REPO, or current working directory
fn repo_path() -> String {
    std::env::var("SUSSURRO_REPO").unwrap_or_else(|_| {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string())
    })
}

// Playback processes we spawn, so they can be killed when the window closes
// (Windows doesn't auto-kill children, so audio would otherwise keep playing).
#[derive(Default)]
struct Children(Arc<Mutex<Vec<Child>>>);

impl Children {
    fn kill_all(&self) {
        for mut child in self.0.lock().unwrap().drain(..) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn voice_dir(lang: &str) -> &'static str {
    match lang {
        "en" => "models/vits-piper-en_US-ryan-medium",
        "es" => "models/vits-piper-es_ES-davefx-medium",
        "it" => "models/vits-piper-it_IT-paola-medium",
        _ => "models/vits-piper-fr_FR-tom-medium", // fr and fallback
    }
}

#[tauri::command(async)]
fn speak(
    text: String,
    lang: String,
    children: tauri::State<Children>,
) -> Result<String, String> {
    let voice = voice_dir(&lang);
    let child = Command::new(format!("{}/build/sussurro-speak", repo_path()))
        .current_dir(repo_path())
        .arg("-k").arg(voice).arg("-t").arg(&text).arg("--play")
        .spawn()
        .map_err(|e| e.to_string())?;

    let pid = child.id();
    children.0.lock().unwrap().push(child);

    loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mut guard = children.0.lock().unwrap();
        let Some(pos) = guard.iter().position(|c| c.id() == pid) else {
            return Ok(String::new()); // killed on window close
        };
        match guard[pos].try_wait() {
            Ok(Some(status)) => {
                guard.remove(pos);
                return if status.success() {
                    Ok(String::new())
                } else {
                    Err(format!("sussurro-speak exited with status {status}"))
                };
            }
            Ok(None) => {}
            Err(e) => {
                guard.remove(pos);
                return Err(e.to_string());
            }
        }
    }
}

// (source, target) -> (model file under models/, optional >>lang<< code for -l)
fn route(source: &str, target: &str) -> Option<(&'static str, &'static str)> {
    let pair = match (source, target) {
        ("en", "fr") => ("tc-en-fr.gguf", ""),
        ("en", "es") => ("tc-en-es.gguf", ""),
        ("en", "it") => ("tc-en-it.gguf", ""),
        ("fr", "en") => ("tc-fr-en.gguf", ""),
        ("it", "en") => ("tc-it-en.gguf", ""),
        ("es", "en") => ("es-en.gguf", ""),
        ("it", "fr") | ("es", "fr") => ("tc-itc-itc.gguf", "fra"),
        ("fr", "it") | ("es", "it") => ("tc-itc-itc.gguf", "ita"),
        ("fr", "es") | ("it", "es") => ("tc-itc-itc.gguf", "spa"),
        _ => return None,
    };
    Some(pair)
}

#[tauri::command(async)]
fn translate(text: String, source: String, target: String) -> Result<String, String> {
    let (model, lang) = route(&source, &target)
        .ok_or_else(|| format!("unsupported language pair: {source} -> {target}"))?;
    let model_path = format!("models/{model}");
    let mut cmd = Command::new(format!("{}/build/sussurro", repo_path()));
    cmd.current_dir(repo_path()).arg("-m").arg(&model_path).arg("-p").arg(&text);
    if !lang.is_empty() {
        cmd.arg("-l").arg(lang);
    }
    let output = cmd.output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("translation:") {
            return Ok(rest.trim().to_string());
        }
    }
    Err("no translation found in output".into())
}

#[tauri::command(async)]
fn transcribe(path: String, lang: String) -> Result<String, String> {
    let output = Command::new(format!("{}/build/sussurro-interpret", repo_path()))
        .current_dir(repo_path())
        .arg("-w").arg("third_party/whisper.cpp/models/ggml-small.bin")
        .arg("-a").arg(&path)
        .arg("-l").arg(&lang)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("heard:") {
            return Ok(rest.trim().to_string());
        }
    }
    Err("no transcription found in output".into())
}

// ---- microphone capture (native, via cpal) ----

struct Recorder {
    samples: Arc<Mutex<Vec<f32>>>,
    in_rate: Arc<AtomicU32>,
    recording: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

impl Default for Recorder {
    fn default() -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            in_rate: Arc::new(AtomicU32::new(0)),
            recording: Arc::new(AtomicBool::new(false)),
            finished: Arc::new(AtomicBool::new(true)),
        }
    }
}

#[tauri::command]
fn start_recording(state: tauri::State<Recorder>) -> Result<(), String> {
    if state.recording.load(Ordering::SeqCst) {
        return Err("already recording".into());
    }
    state.samples.lock().unwrap().clear();
    state.in_rate.store(0, Ordering::SeqCst);
    state.recording.store(true, Ordering::SeqCst);
    state.finished.store(false, Ordering::SeqCst);

    let samples = state.samples.clone();
    let in_rate = state.in_rate.clone();
    let recording = state.recording.clone();
    let finished = state.finished.clone();

    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                finished.store(true, Ordering::SeqCst);
                return;
            }
        };
        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(_) => {
                finished.store(true, Ordering::SeqCst);
                return;
            }
        };
        in_rate.store(config.sample_rate(), Ordering::SeqCst);
        let channels = config.channels() as usize;
        let stream_config: cpal::StreamConfig = config.clone().into();
        let err_fn = |e| eprintln!("audio stream error: {e}");

        let samples_cb = samples.clone();
        let recording_cb = recording.clone();

        let built = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                stream_config,
                move |data: &[f32], _: &_| {
                    if !recording_cb.load(Ordering::SeqCst) {
                        return;
                    }
                    let mut buf = samples_cb.lock().unwrap();
                    for frame in data.chunks(channels) {
                        buf.push(frame.iter().sum::<f32>() / channels as f32);
                    }
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                stream_config,
                move |data: &[i16], _: &_| {
                    if !recording_cb.load(Ordering::SeqCst) {
                        return;
                    }
                    let mut buf = samples_cb.lock().unwrap();
                    for frame in data.chunks(channels) {
                        let s: f32 =
                            frame.iter().map(|&v| v as f32 / 32768.0).sum::<f32>() / channels as f32;
                        buf.push(s);
                    }
                },
                err_fn,
                None,
            ),
            _ => {
                finished.store(true, Ordering::SeqCst);
                return;
            }
        };

        let stream = match built {
            Ok(s) => s,
            Err(_) => {
                finished.store(true, Ordering::SeqCst);
                return;
            }
        };
        if stream.play().is_err() {
            finished.store(true, Ordering::SeqCst);
            return;
        }

        while recording.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        drop(stream);
        finished.store(true, Ordering::SeqCst);
    });

    Ok(())
}

#[tauri::command(async)]
fn stop_recording(state: tauri::State<Recorder>) -> Result<String, String> {
    state.recording.store(false, Ordering::SeqCst);
    for _ in 0..150 {
        if state.finished.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let in_rate = state.in_rate.load(Ordering::SeqCst);
    if in_rate == 0 {
        return Err("recording never started (no input device, or mic permission denied)".into());
    }
    let raw = state.samples.lock().unwrap().clone();
    if raw.is_empty() {
        return Err("no audio captured (check microphone permission in System Settings)".into());
    }

    let samples = resample_to_16k(&raw, in_rate);
    let path = std::env::temp_dir().join("sussurro_input.wav");
    write_wav_16k(&path, &samples)?;
    Ok(path.to_string_lossy().to_string())
}

fn resample_to_16k(input: &[f32], in_rate: u32) -> Vec<f32> {
    if in_rate == 16_000 || input.is_empty() {
        return input.to_vec();
    }
    let ratio = in_rate as f64 / 16_000.0;
    let out_len = (input.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let idx = i as f64 * ratio;
        let i0 = idx.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = (idx - i0 as f64) as f32;
        out.push(input[i0] * (1.0 - frac) + input[i1] * frac);
    }
    out
}

fn write_wav_16k(path: &std::path::Path, samples: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Recorder::default())
        .manage(Children::default())
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                window.state::<Children>().kill_all();
            }
        })
        .invoke_handler(tauri::generate_handler![
            speak,
            translate,
            transcribe,
            start_recording,
            stop_recording
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}