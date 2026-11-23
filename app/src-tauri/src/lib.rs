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

#[allow(clippy::zombie_processes)]
#[tauri::command(async)]
fn speak(text: String, lang: String, children: tauri::State<Children>) -> Result<String, String> {
    let voice = voice_dir(&lang);
    let child = Command::new(format!("{}/build/sussurro-speak", repo_path()))
        .current_dir(repo_path())
        .arg("-k")
        .arg(voice)
        .arg("-t")
        .arg(&text)
        .arg("--play")
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
                let _ = guard[pos].wait();
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
    cmd.current_dir(repo_path())
        .arg("-m")
        .arg(&model_path)
        .arg("-p")
        .arg(&text);
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
        .arg("-w")
        .arg("third_party/whisper.cpp/models/ggml-small.bin")
        .arg("-a")
        .arg(&path)
        .arg("-l")
        .arg(&lang)
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

#[derive(Default)]
struct VoiceProfiles {
    selected: Arc<Mutex<Option<String>>>,
}

#[tauri::command]
fn get_config() -> Result<String, String> {
    let base = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let path = base.join(".traductor").join("config.json");
    if path.exists() {
        std::fs::read_to_string(&path).map_err(|e| e.to_string())
    } else {
        Ok(r#"{"ask_voice_on_startup": true}"#.to_string())
    }
}

#[tauri::command]
fn set_config(config: String) -> Result<(), String> {
    let base = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let dir = base.join(".traductor");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("config.json");
    std::fs::write(&path, config).map_err(|e| e.to_string())
}

impl VoiceProfiles {
    fn voices_dir() -> std::path::PathBuf {
        let base = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."));
        base.join(".traductor").join("voices")
    }

    fn config_path() -> std::path::PathBuf {
        let base = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."));
        base.join(".traductor").join("selected_voice.txt")
    }

    fn list_samples() -> Vec<String> {
        let dir = Self::voices_dir();
        if !dir.exists() {
            return Vec::new();
        }
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".wav") {
                        names.push(name.trim_end_matches(".wav").to_string());
                    }
                }
            }
        }
        names.sort();
        names
    }

    fn selected_name() -> Option<String> {
        let path = Self::config_path();
        if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        } else {
            None
        }
    }

    fn set_selected(name: &str) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, name);
    }
}

#[tauri::command]
fn list_input_devices() -> Result<Vec<String>, String> {
    let host = cpal::default_host();
    let devices = host.input_devices().map_err(|e| e.to_string())?;
    let mut names = Vec::new();
    for device in devices {
        if let Ok(desc) = device.description() {
            names.push(desc.name().to_string());
        }
    }
    if names.is_empty() {
        // Fallback: try default device name
        if let Some(dev) = host.default_input_device() {
            if let Ok(desc) = dev.description() {
                names.push(desc.name().to_string());
            }
        }
    }
    Ok(names)
}

#[tauri::command]
fn list_output_devices() -> Result<Vec<String>, String> {
    let host = cpal::default_host();
    let devices = host.output_devices().map_err(|e| e.to_string())?;
    let mut names = Vec::new();
    for device in devices {
        if let Ok(desc) = device.description() {
            names.push(desc.name().to_string());
        }
    }
    if names.is_empty() {
        if let Some(dev) = host.default_output_device() {
            if let Ok(desc) = dev.description() {
                names.push(desc.name().to_string());
            }
        }
    }
    Ok(names)
}

#[tauri::command]
fn detect_virtual_device() -> Result<Option<String>, String> {
    let host = cpal::default_host();
    let devices = host.output_devices().map_err(|e| e.to_string())?;
    let virtual_keywords = [
        "virtual",
        "vb-cable",
        "vb cable",
        "blackhole",
        "null",
        "dummy",
        "loopback",
    ];
    for device in devices {
        if let Ok(desc) = device.description() {
            let name = desc.name().to_string();
            let lower = name.to_lowercase();
            for kw in &virtual_keywords {
                if lower.contains(kw) {
                    return Ok(Some(name));
                }
            }
        }
    }
    Ok(None)
}

#[tauri::command]
fn start_recording(
    state: tauri::State<Recorder>,
    device_name: Option<String>,
) -> Result<(), String> {
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
        let device = if let Some(ref name) = device_name {
            match host.input_devices() {
                Ok(mut devs) => devs
                    .find(|d| {
                        d.description()
                            .map(|desc| desc.name() == *name)
                            .unwrap_or(false)
                    })
                    .or_else(|| host.default_input_device()),
                Err(_) => host.default_input_device(),
            }
        } else {
            host.default_input_device()
        };
        let device = match device {
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
        let stream_config: cpal::StreamConfig = config.into();
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
                        let s: f32 = frame.iter().map(|&v| v as f32 / 32768.0).sum::<f32>()
                            / channels as f32;
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

#[tauri::command]
fn calibrate_voice(
    state: tauri::State<Recorder>,
    profiles: tauri::State<VoiceProfiles>,
    name: String,
) -> Result<String, String> {
    let dir = VoiceProfiles::voices_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    // Sanitize name
    let safe_name: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if safe_name.is_empty() {
        return Err("Invalid name. Use alphanumeric characters.".into());
    }

    let voice_path = dir.join(format!("{}.wav", safe_name));

    let raw = state.samples.lock().unwrap().clone();
    if raw.is_empty() {
        return Err("No audio captured. Record a voice sample first (5-30 seconds).".into());
    }

    let in_rate = state.in_rate.load(Ordering::SeqCst);
    let samples = if in_rate == 16_000 || raw.is_empty() {
        raw
    } else {
        resample_to_16k(&raw, in_rate)
    };

    write_wav_16k(&voice_path, &samples)?;
    VoiceProfiles::set_selected(&safe_name);
    *profiles.selected.lock().unwrap() = Some(safe_name.clone());
    Ok(voice_path.to_string_lossy().to_string())
}

#[tauri::command]
fn list_voice_samples() -> Result<Vec<String>, String> {
    Ok(VoiceProfiles::list_samples())
}

#[tauri::command]
fn select_voice_sample(profiles: tauri::State<VoiceProfiles>, name: String) -> Result<(), String> {
    let samples = VoiceProfiles::list_samples();
    if !samples.contains(&name) {
        return Err(format!("Voice sample '{}' not found", name));
    }
    VoiceProfiles::set_selected(&name);
    *profiles.selected.lock().unwrap() = Some(name);
    Ok(())
}

#[tauri::command]
fn delete_voice_sample(name: String) -> Result<(), String> {
    let dir = VoiceProfiles::voices_dir();
    let wav = dir.join(format!("{}.wav", name));
    let txt = dir.join(format!("{}.txt", name));
    if wav.exists() {
        std::fs::remove_file(&wav).map_err(|e| e.to_string())?;
    }
    if txt.exists() {
        let _ = std::fs::remove_file(&txt);
    }
    // If deleted was selected, clear selection
    if VoiceProfiles::selected_name().as_deref() == Some(&name) {
        let _ = std::fs::remove_file(VoiceProfiles::config_path());
    }
    Ok(())
}

#[tauri::command]
fn get_selected_voice() -> Result<Option<String>, String> {
    Ok(VoiceProfiles::selected_name())
}

#[tauri::command]
fn get_voice_sample_path(name: String) -> Result<Option<String>, String> {
    let dir = VoiceProfiles::voices_dir();
    let wav = dir.join(format!("{}.wav", name));
    if wav.exists() {
        Ok(Some(wav.to_string_lossy().to_string()))
    } else {
        Ok(None)
    }
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
        .manage(VoiceProfiles::default())
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
            stop_recording,
            list_input_devices,
            list_output_devices,
            detect_virtual_device,
            calibrate_voice,
            list_voice_samples,
            select_voice_sample,
            delete_voice_sample,
            get_selected_voice,
            get_voice_sample_path,
            get_config,
            set_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
