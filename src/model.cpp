#include "sussurro.h"

#include <cstdio>
#include <stdexcept>

#include "ggml.h"
#include "gguf.h"

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int rev[256];
    for (int i = 0; i < 256; ++i) rev[i] = -1;
    for (int i = 0; i < 64; ++i) rev[(unsigned char) chars[i]] = i;

    std::vector<uint8_t> out;
    int acc = 0;
    int bits = -8;
    for (unsigned char c : in) {
        if (rev[c] < 0) continue;
        acc = (acc << 6) + rev[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back((uint8_t) ((acc >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

static uint32_t read_u32(gguf_context * g, const char * key) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) throw std::runtime_error(std::string("missing key: ") + key);
    return gguf_get_val_u32(g, id);
}

static int32_t read_u32_or(gguf_context * g, const char * key, int32_t fallback) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return (int32_t) gguf_get_val_u32(g, id);
}

static std::string read_str(gguf_context * g, const char * key) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) throw std::runtime_error(std::string("missing key: ") + key);
    return gguf_get_val_str(g, id);
}

static std::string read_str_or(gguf_context * g, const char * key, const char * fallback) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_str(g, id);
}

static void load_tokenizer(gguf_context * g, sussurro_vocab & vocab) {
    const int64_t id = gguf_find_key(g, "sussurro-marian.tokenizer.tokens");
    if (id < 0) throw std::runtime_error("missing tokenizer tokens");
    const size_t n = gguf_get_arr_n(g, id);
    vocab.id_to_token.resize(n);
    for (size_t i = 0; i < n; ++i) {
        vocab.id_to_token[i] = gguf_get_arr_str(g, id, i);
    }
    vocab.source_spm = base64_decode(read_str(g, "sussurro-marian.tokenizer.source_spm"));
    vocab.target_spm = base64_decode(read_str(g, "sussurro-marian.tokenizer.target_spm"));
}

bool sussurro_model_load(const std::string & path, sussurro_model & model) {
    try {
        gguf_init_params params = { false, &model.ctx };
        model.gguf = gguf_init_from_file(path.c_str(), params);
        if (!model.gguf) throw std::runtime_error("gguf_init_from_file failed");

        gguf_context * g = model.gguf;
        sussurro_hparams & hp = model.hparams;
        hp.vocab_size = read_u32(g, "sussurro-marian.vocab_size");
        hp.d_model    = read_u32(g, "sussurro-marian.d_model");
        hp.enc_layers = read_u32(g, "sussurro-marian.encoder.block_count");
        hp.dec_layers = read_u32(g, "sussurro-marian.decoder.block_count");
        hp.enc_heads  = read_u32(g, "sussurro-marian.encoder.head_count");
        hp.dec_heads  = read_u32(g, "sussurro-marian.decoder.head_count");
        hp.enc_ffn    = read_u32(g, "sussurro-marian.encoder.ffn_dim");
        hp.dec_ffn    = read_u32(g, "sussurro-marian.decoder.ffn_dim");
        hp.max_pos    = read_u32(g, "sussurro-marian.max_position");
        hp.eos_token_id = read_u32_or(g, "sussurro-marian.eos_token_id", -1);
        hp.pad_token_id = read_u32_or(g, "sussurro-marian.pad_token_id", -1);
        hp.decoder_start_token_id =
            read_u32_or(g, "sussurro-marian.decoder_start_token_id", hp.pad_token_id);
        hp.activation = read_str_or(g, "sussurro-marian.activation", "gelu");
        hp.scale_embedding = read_u32_or(g, "sussurro-marian.scale_embedding", 0) != 0;

        const int64_t n_tensors = gguf_get_n_tensors(g);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(model.ctx, name);
            if (t != nullptr) model.tensors[name] = t;
        }

        load_tokenizer(g, model.vocab);
        return true;
    } catch (const std::exception & e) {
        fprintf(stderr, "load error: %s\n", e.what());
        sussurro_model_free(model);
        return false;
    }
}

void sussurro_model_free(sussurro_model & model) {
    if (model.gguf != nullptr) gguf_free(model.gguf);
    if (model.ctx != nullptr) ggml_free(model.ctx);
    model.gguf = nullptr;
    model.ctx = nullptr;
}
