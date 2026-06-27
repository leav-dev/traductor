#include <cstdio>
#include <string>
#include <vector>

#include "sussurro.h"

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt;
    int max_new = 256;
    int beam = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "-n" && i + 1 < argc) max_new = std::stoi(argv[++i]);
        else if (arg == "-b" && i + 1 < argc) beam = std::stoi(argv[++i]);
    }
    if (model_path.empty()) {
        printf("usage: sussurro -m <model.gguf> [-p \"text\"] [-n max_tokens] [-b beam_size]\n");
        return 1;
    }

    sussurro_model model;
    if (!sussurro_model_load(model_path, model)) return 1;
    printf("sussurro 0.5.0 - loaded (vocab %u, %zu tensors, act=%s, scale=%d)\n",
           model.hparams.vocab_size, model.tensors.size(),
           model.hparams.activation.c_str(), model.hparams.scale_embedding);

    if (!prompt.empty()) {
        sussurro_tokenizer * tok = sussurro_tokenizer_create(model);
        const std::string out = sussurro_translate(model, tok, prompt, max_new, beam);
        printf("source:      %s\n", prompt.c_str());
        printf("translation: %s\n", out.c_str());
        sussurro_tokenizer_free(tok);
    }

    sussurro_model_free(model);
    return 0;
}