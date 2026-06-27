#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <dirent.h>

#include "sherpa-onnx/c-api/c-api.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

static std::string find_onnx(const std::string & dir) {
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) return "";
    std::string found;
    for (struct dirent * e = readdir(d); e != nullptr; e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() > 5 && name.substr(name.size() - 5) == ".onnx") {
            found = dir + "/" + name;
            break;
        }
    }
    closedir(d);
    return found;
}

struct PlaybackData {
    const float * samples;
    ma_uint64 total;
    std::atomic<ma_uint64> cursor;
};

static void data_callback(ma_device * device, void * output, const void * input, ma_uint32 frames) {
    (void) input;
    PlaybackData * pd = static_cast<PlaybackData *>(device->pUserData);
    float * out = static_cast<float *>(output);
    const ma_uint64 cur = pd->cursor.load(std::memory_order_relaxed);
    const ma_uint64 remaining = pd->total > cur ? pd->total - cur : 0;
    const ma_uint32 n = frames < remaining ? frames : static_cast<ma_uint32>(remaining);
    for (ma_uint32 i = 0; i < n; ++i) out[i] = pd->samples[cur + i];
    for (ma_uint32 i = n; i < frames; ++i) out[i] = 0.0f;
    pd->cursor.store(cur + n, std::memory_order_relaxed);
}

static void play_audio(const float * samples, int n, int sample_rate) {
    PlaybackData pd;
    pd.samples = samples;
    pd.total = static_cast<ma_uint64>(n);
    pd.cursor.store(0, std::memory_order_relaxed);

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format   = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate        = static_cast<ma_uint32>(sample_rate);
    config.dataCallback      = data_callback;
    config.pUserData         = &pd;

    ma_device device;
    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS) {
        fprintf(stderr, "could not open an audio device for playback\n");
        return;
    }
    ma_device_start(&device);
    while (pd.cursor.load(std::memory_order_relaxed) < pd.total) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ma_device_uninit(&device);
}

int main(int argc, char ** argv) {
    std::string dir, text, out = "speak.wav";
    int sid = 0;
    float speed = 1.0f;
    bool play = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-k" && i + 1 < argc) dir = argv[++i];
        else if (a == "-t" && i + 1 < argc) text = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-v" && i + 1 < argc) sid = std::stoi(argv[++i]);
        else if (a == "-s" && i + 1 < argc) speed = std::stof(argv[++i]);
        else if (a == "--play") play = true;
    }
    if (dir.empty() || text.empty()) {
        printf("usage: sussurro-speak -k <voice-dir> -t \"texte\" [-o out.wav] [-v sid=0] [-s speed=1.0] [--play]\n");
        return 1;
    }

    const std::string model  = find_onnx(dir);
    const std::string tokens = dir + "/tokens.txt";
    const std::string espeak = dir + "/espeak-ng-data";
    if (model.empty()) {
        fprintf(stderr, "no .onnx model found in %s\n", dir.c_str());
        return 1;
    }

    SherpaOnnxOfflineTtsConfig config;
    memset(&config, 0, sizeof(config));
    config.model.vits.model         = model.c_str();
    config.model.vits.tokens        = tokens.c_str();
    config.model.vits.data_dir      = espeak.c_str();
    config.model.vits.noise_scale   = 0.667f;
    config.model.vits.noise_scale_w = 0.8f;
    config.model.vits.length_scale  = speed > 0.0f ? speed : 1.0f;
    config.model.num_threads        = 4;
    config.model.provider           = "cpu";
    config.max_num_sentences        = 1;
    config.silence_scale            = 0.2f;

    const SherpaOnnxOfflineTts * tts = SherpaOnnxCreateOfflineTts(&config);
    if (tts == nullptr) {
        fprintf(stderr, "failed to create TTS (check the files under %s)\n", dir.c_str());
        return 1;
    }

    SherpaOnnxGenerationConfig gen;
    memset(&gen, 0, sizeof(gen));
    gen.speed         = 1.0f;
    gen.sid           = sid;
    gen.silence_scale = 0.2f;

    const SherpaOnnxGeneratedAudio * audio =
        SherpaOnnxOfflineTtsGenerateWithConfig(tts, text.c_str(), &gen, nullptr, nullptr);
    if (audio == nullptr || audio->n == 0) {
        fprintf(stderr, "generation failed\n");
        SherpaOnnxDestroyOfflineTts(tts);
        return 1;
    }

    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, out.c_str());
    printf("wrote %s  (%.1f s @ %d Hz)\n", out.c_str(),
           (double) audio->n / audio->sample_rate, audio->sample_rate);

    if (play) play_audio(audio->samples, audio->n, audio->sample_rate);

    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    SherpaOnnxDestroyOfflineTts(tts);
    return 0;
}