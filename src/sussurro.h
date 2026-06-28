#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

struct sussurro_hparams {
    uint32_t vocab_size = 0;
    uint32_t d_model = 0;
    uint32_t enc_layers = 0;
    uint32_t dec_layers = 0;
    uint32_t enc_heads = 0;
    uint32_t dec_heads = 0;
    uint32_t enc_ffn = 0;
    uint32_t dec_ffn = 0;
    uint32_t max_pos = 0;
    int32_t eos_token_id = -1;
    int32_t pad_token_id = -1;
    int32_t decoder_start_token_id = -1;
    std::string activation = "gelu";
    bool scale_embedding = false;
};

struct sussurro_vocab {
    std::vector<std::string> id_to_token;
    std::vector<uint8_t> source_spm;
    std::vector<uint8_t> target_spm;
};

struct sussurro_model {
    sussurro_hparams hparams;
    sussurro_vocab vocab;
    ggml_context * ctx = nullptr;
    gguf_context * gguf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
};

bool sussurro_model_load(const std::string & path, sussurro_model & model);
void sussurro_model_free(sussurro_model & model);

struct sussurro_tokenizer;

sussurro_tokenizer * sussurro_tokenizer_create(const sussurro_model & model);
void sussurro_tokenizer_free(sussurro_tokenizer * tokenizer);
std::vector<int32_t> sussurro_encode(sussurro_tokenizer * tokenizer, const std::string & text, const std::string & lang_token = "");
std::string sussurro_decode(sussurro_tokenizer * tokenizer, const std::vector<int32_t> & ids);

std::vector<float> sussurro_run_encoder(const sussurro_model & model, const std::vector<int32_t> & ids);

std::string sussurro_translate(const sussurro_model & model, sussurro_tokenizer * tokenizer,
                               const std::string & source, int max_new_tokens, int beam_size = 1,
                               const std::string & lang_token = "");