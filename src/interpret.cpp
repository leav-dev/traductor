#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sussurro.h"
#include "whisper.h"

static bool read_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    char riff[4], wave[4];
    uint32_t riff_size = 0;
    if (fread(riff, 1, 4, f) != 4 || fread(&riff_size, 4, 1, f) != 1 || fread(wave, 1, 4, f) != 4 ||
        std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "%s is not a RIFF/WAVE file\n", path.c_str());
        fclose(f);
        return false;
    }

    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<int16_t> pcm;
    char id[4];
    uint32_t sz = 0;
    while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (std::strncmp(id, "fmt ", 4) == 0) {
            uint8_t buf[64];
            const uint32_t n = sz < sizeof(buf) ? sz : (uint32_t) sizeof(buf);
            if (fread(buf, 1, n, f) != n) break;
            std::memcpy(&fmt, buf + 0, 2);
            std::memcpy(&channels, buf + 2, 2);
            std::memcpy(&rate, buf + 4, 4);
            std::memcpy(&bits, buf + 14, 2);
            if (sz > n) fseek(f, sz - n, SEEK_CUR);
        } else if (std::strncmp(id, "data", 4) == 0) {
            pcm.resize(sz / 2);
            if (fread(pcm.data(), 1, sz, f) != sz) pcm.clear();
            break;
        } else {
            fseek(f, sz + (sz & 1), SEEK_CUR);
        }
    }
    fclose(f);

    if (fmt != 1 || channels != 1 || rate != 16000 || bits != 16) {
        fprintf(stderr, "need 16 kHz mono 16-bit PCM WAV (got format=%u channels=%u rate=%u bits=%u)\n",
                fmt, channels, rate, bits);
        fprintf(stderr, "convert with: ffmpeg -i input -ar 16000 -ac 1 -c:a pcm_s16le out.wav\n");
        return false;
    }

    out.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) out[i] = (float) pcm[i] / 32768.0f;
    return true;
}

static std::string transcribe(const std::string & model_path, const std::vector<float> & pcm,
                              const std::string & lang) {
    whisper_context_params cparams = whisper_context_default_params();
    whisper_context * wctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (wctx == nullptr) {
        fprintf(stderr, "failed to load whisper model: %s\n", model_path.c_str());
        return "";
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language       = lang.c_str();
    wparams.translate      = false;
    wparams.no_timestamps  = true;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_special  = false;
    wparams.n_threads      = 4;

    std::string text;
    if (whisper_full(wctx, wparams, pcm.data(), (int) pcm.size()) == 0) {
        const int n = whisper_full_n_segments(wctx);
        for (int i = 0; i < n; ++i) text += whisper_full_get_segment_text(wctx, i);
    } else {
        fprintf(stderr, "whisper transcription failed\n");
    }
    whisper_free(wctx);

    const size_t s = text.find_first_not_of(' ');
    return s == std::string::npos ? text : text.substr(s);
}

int main(int argc, char ** argv) {
    std::string whisper_model, translate_model, audio, lang = "it";
    int beam = 1, max_new = 256;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-w" && i + 1 < argc) whisper_model = argv[++i];
        else if (a == "-m" && i + 1 < argc) translate_model = argv[++i];
        else if (a == "-a" && i + 1 < argc) audio = argv[++i];
        else if (a == "-l" && i + 1 < argc) lang = argv[++i];
        else if (a == "-b" && i + 1 < argc) beam = std::stoi(argv[++i]);
    }
    if (whisper_model.empty() || audio.empty()) {
        printf("usage: sussurro-interpret -w <whisper.bin> -a <audio.wav> [-m <translate.gguf>] [-l it] [-b beam]\n");
        return 1;
    }

    std::vector<float> pcm;
    if (!read_wav_16k_mono(audio, pcm)) return 1;
    printf("audio:       %.1f s\n", pcm.size() / 16000.0);

    const std::string source = transcribe(whisper_model, pcm, lang);
    printf("heard:       %s\n", source.c_str());
    if (source.empty()) return 1;

    if (!translate_model.empty()) {
        sussurro_model model;
        if (!sussurro_model_load(translate_model, model)) return 1;
        sussurro_tokenizer * tok = sussurro_tokenizer_create(model);
        const std::string out = sussurro_translate(model, tok, source, max_new, beam);
        printf("translation: %s\n", out.c_str());
        sussurro_tokenizer_free(tok);
        sussurro_model_free(model);
    }
    return 0;
}