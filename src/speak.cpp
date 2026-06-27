#include <cstdio>
#include <cstring>
#include <string>

#include <dirent.h>

#include "sherpa-onnx/c-api/c-api.h"

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

int main(int argc, char ** argv) {
    std::string dir, text, out = "speak.wav";
    int sid = 0;
    float speed = 1.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-k" && i + 1 < argc) dir = argv[++i];
        else if (a == "-t" && i + 1 < argc) text = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-v" && i + 1 < argc) sid = std::stoi(argv[++i]);
        else if (a == "-s" && i + 1 < argc) speed = std::stof(argv[++i]);
    }
    if (dir.empty() || text.empty()) {
        printf("usage: sussurro-speak -k <voice-dir> -t \"texte\" [-o out.wav] [-v sid=0] [-s speed=1.0]\n");
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

    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    SherpaOnnxDestroyOfflineTts(tts);
    return 0;
}