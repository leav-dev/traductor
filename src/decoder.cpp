#include "sussurro.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cctype>

#include "ggml.h"
#include "ggml-cpu.h"

static ggml_tensor * get(const sussurro_model & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second;
}

static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    x = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) x = ggml_add(ctx, x, b);
    return x;
}

static ggml_tensor * activation(ggml_context * ctx, const std::string & kind, ggml_tensor * x) {
    if (kind == "swish" || kind == "silu") return ggml_silu(ctx, x);
    if (kind == "relu") return ggml_relu(ctx, x);
    return ggml_gelu(ctx, x);
}

static void fill_position_at(float * pd, int d, int abs_pos) {
    const int half = d / 2;
    for (int k = 0; k < half; ++k) {
        const float freq = std::pow(10000.0f, -2.0f * (float) k / (float) d);
        pd[k]        = std::sin((float) abs_pos * freq);
        pd[half + k] = std::cos((float) abs_pos * freq);
    }
}

struct sussurro_kv_cache {
    std::vector<std::vector<float>> k;  // per layer, self-attn keys, grows by d_model each step
    std::vector<std::vector<float>> v;
};

static std::vector<float> decode_step(const sussurro_model & model,
                                      const std::vector<float> & memory, int n_src,
                                      int32_t token, int t, sussurro_kv_cache & cache) {
    const sussurro_hparams & hp = model.hparams;
    const int d = (int) hp.d_model;
    const int n_head = (int) hp.dec_heads;
    const int head_dim = d / n_head;
    const int vocab = (int) hp.vocab_size;
    const int n_layer = (int) hp.dec_layers;
    const float scale = 1.0f / std::sqrt((float) head_dim);
    const int Lk = t + 1;

    ggml_init_params ip = { 128ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t *) tok->data)[0] = token;

    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1);
    fill_position_at((float *) pos->data, d, t);

    ggml_tensor * mem = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_src);
    std::memcpy(mem->data, memory.data(), (size_t) d * n_src * sizeof(float));

    ggml_tensor * cur = ggml_get_rows(ctx, get(model, "model.shared.weight"), tok);
    if (hp.scale_embedding) cur = ggml_scale(ctx, cur, std::sqrt((float) d));
    cur = ggml_add(ctx, cur, pos);

    std::vector<ggml_tensor *> knew_t(n_layer), vnew_t(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "model.decoder.layers." + std::to_string(il) + ".";
        ggml_tensor * residual = cur;

        ggml_tensor * q = linear(ctx, get(model, p + "self_attn.q_proj.weight"), get(model, p + "self_attn.q_proj.bias"), cur);
        ggml_tensor * knew = linear(ctx, get(model, p + "self_attn.k_proj.weight"), get(model, p + "self_attn.k_proj.bias"), cur);
        ggml_tensor * vnew = linear(ctx, get(model, p + "self_attn.v_proj.weight"), get(model, p + "self_attn.v_proj.bias"), cur);
        q = ggml_scale(ctx, q, scale);
        knew_t[il] = knew;
        vnew_t[il] = vnew;

        ggml_tensor * Kf;
        ggml_tensor * Vf;
        if (t > 0) {
            ggml_tensor * kprev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, t);
            std::memcpy(kprev->data, cache.k[il].data(), (size_t) d * t * sizeof(float));
            ggml_tensor * vprev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, t);
            std::memcpy(vprev->data, cache.v[il].data(), (size_t) d * t * sizeof(float));
            Kf = ggml_concat(ctx, kprev, knew, 1);
            Vf = ggml_concat(ctx, vprev, vnew, 1);
        } else {
            Kf = knew;
            Vf = vnew;
        }

        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, n_head, 1), 0, 2, 1, 3));
        ggml_tensor * Kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Kf, head_dim, n_head, Lk), 0, 2, 1, 3));
        ggml_tensor * KQ = ggml_mul_mat(ctx, Kh, qh);
        KQ = ggml_soft_max(ctx, KQ);
        ggml_tensor * Vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Vf, head_dim, n_head, Lk), 1, 2, 0, 3));
        ggml_tensor * KQV = ggml_mul_mat(ctx, Vh, KQ);
        KQV = ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx, KQV, d, 1);
        cur = linear(ctx, get(model, p + "self_attn.out_proj.weight"), get(model, p + "self_attn.out_proj.bias"), cur);
        cur = ggml_add(ctx, cur, residual);
        cur = layer_norm(ctx, cur, get(model, p + "self_attn_layer_norm.weight"), get(model, p + "self_attn_layer_norm.bias"));

        residual = cur;
        ggml_tensor * qc = linear(ctx, get(model, p + "encoder_attn.q_proj.weight"), get(model, p + "encoder_attn.q_proj.bias"), cur);
        ggml_tensor * Kc = linear(ctx, get(model, p + "encoder_attn.k_proj.weight"), get(model, p + "encoder_attn.k_proj.bias"), mem);
        ggml_tensor * Vc = linear(ctx, get(model, p + "encoder_attn.v_proj.weight"), get(model, p + "encoder_attn.v_proj.bias"), mem);
        qc = ggml_scale(ctx, qc, scale);
        ggml_tensor * qch = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, qc, head_dim, n_head, 1), 0, 2, 1, 3));
        ggml_tensor * Kch = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Kc, head_dim, n_head, n_src), 0, 2, 1, 3));
        ggml_tensor * KQc = ggml_mul_mat(ctx, Kch, qch);
        KQc = ggml_soft_max(ctx, KQc);
        ggml_tensor * Vch = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Vc, head_dim, n_head, n_src), 1, 2, 0, 3));
        ggml_tensor * KQVc = ggml_mul_mat(ctx, Vch, KQc);
        KQVc = ggml_cont(ctx, ggml_permute(ctx, KQVc, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx, KQVc, d, 1);
        cur = linear(ctx, get(model, p + "encoder_attn.out_proj.weight"), get(model, p + "encoder_attn.out_proj.bias"), cur);
        cur = ggml_add(ctx, cur, residual);
        cur = layer_norm(ctx, cur, get(model, p + "encoder_attn_layer_norm.weight"), get(model, p + "encoder_attn_layer_norm.bias"));

        residual = cur;
        cur = linear(ctx, get(model, p + "fc1.weight"), get(model, p + "fc1.bias"), cur);
        cur = activation(ctx, hp.activation, cur);
        cur = linear(ctx, get(model, p + "fc2.weight"), get(model, p + "fc2.bias"), cur);
        cur = ggml_add(ctx, cur, residual);
        cur = layer_norm(ctx, cur, get(model, p + "final_layer_norm.weight"), get(model, p + "final_layer_norm.bias"));
    }

    ggml_tensor * logits = ggml_mul_mat(ctx, get(model, "model.shared.weight"), cur);
    auto bit = model.tensors.find("final_logits_bias");
    if (bit != model.tensors.end()) logits = ggml_add(ctx, logits, bit->second);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    for (int il = 0; il < n_layer; ++il) {
        ggml_build_forward_expand(gf, knew_t[il]);
        ggml_build_forward_expand(gf, vnew_t[il]);
    }
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    for (int il = 0; il < n_layer; ++il) {
        const float * kp = (const float *) knew_t[il]->data;
        const float * vp = (const float *) vnew_t[il]->data;
        cache.k[il].insert(cache.k[il].end(), kp, kp + d);
        cache.v[il].insert(cache.v[il].end(), vp, vp + d);
    }

    std::vector<float> out(vocab);
    std::memcpy(out.data(), logits->data, (size_t) vocab * sizeof(float));
    ggml_free(ctx);
    return out;
}

static std::vector<int32_t> beam_search(const sussurro_model & model,
                                        const std::vector<float> & memory, int n_src,
                                        int32_t start, int32_t eos,
                                        int max_new, int beam_size) {
    struct Hyp { std::vector<int32_t> tokens; double score; sussurro_kv_cache cache; };
    struct Cand { int parent; int32_t tok; double score; };
    const int vocab = (int) model.hparams.vocab_size;
    const int n_layer = (int) model.hparams.dec_layers;
    const double alpha = 0.6;

    std::vector<Hyp> beams(1);
    beams[0].tokens = { start };
    beams[0].score = 0.0;
    beams[0].cache.k.resize(n_layer);
    beams[0].cache.v.resize(n_layer);

    std::vector<Hyp> finished;

    for (int t = 0; t < max_new && !beams.empty(); ++t) {
        std::vector<Cand> cand;
        cand.reserve(beams.size() * beam_size);
        for (int bi = 0; bi < (int) beams.size(); ++bi) {
            const std::vector<float> logits = decode_step(model, memory, n_src, beams[bi].tokens.back(), t, beams[bi].cache);
            float mx = logits[0];
            for (float v : logits) if (v > mx) mx = v;
            double sum = 0.0;
            for (float v : logits) sum += std::exp((double) (v - mx));
            const double logZ = std::log(sum) + (double) mx;

            std::vector<std::pair<float, int>> sc(vocab);
            for (int i = 0; i < vocab; ++i) sc[i] = { logits[i], i };
            const int k = std::min(beam_size, vocab);
            std::partial_sort(sc.begin(), sc.begin() + k, sc.end(),
                              [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
            for (int j = 0; j < k; ++j)
                cand.push_back({ bi, sc[j].second, beams[bi].score + ((double) sc[j].first - logZ) });
        }

        const int keep = std::min((int) cand.size(), beam_size);
        std::partial_sort(cand.begin(), cand.begin() + keep, cand.end(),
                          [](const Cand & a, const Cand & b) { return a.score > b.score; });

        std::vector<Hyp> next;
        for (int i = 0; i < keep; ++i) {
            const Cand & c = cand[i];
            if (c.tok == eos) {
                Hyp f;
                f.tokens = beams[c.parent].tokens;
                f.score = c.score;
                finished.push_back(std::move(f));
            } else {
                Hyp h;
                h.tokens = beams[c.parent].tokens;
                h.tokens.push_back(c.tok);
                h.score = c.score;
                h.cache = beams[c.parent].cache;
                next.push_back(std::move(h));
            }
        }
        beams = std::move(next);
        if ((int) finished.size() >= beam_size) break;
    }

    const Hyp * best = nullptr;
    double best_norm = -1e300;
    auto consider = [&](const Hyp & h) {
        const int len = std::max(1, (int) h.tokens.size() - 1);
        const double norm = h.score / std::pow((double) len, alpha);
        if (norm > best_norm) { best_norm = norm; best = &h; }
    };
    for (const Hyp & h : finished) consider(h);
    for (const Hyp & h : beams) consider(h);

    std::vector<int32_t> out;
    if (best != nullptr) out.assign(best->tokens.begin() + 1, best->tokens.end());
    return out;
}

static std::string translate_sentence(const sussurro_model & model, sussurro_tokenizer * tok,
                                      const std::string & source, int max_new_tokens, int beam_size,
                                      int & n_decoded, const std::string & lang_token) {
    const std::vector<int32_t> src_ids = sussurro_encode(tok, source, lang_token);
    const int n_src = (int) src_ids.size();
    const std::vector<float> memory = sussurro_run_encoder(model, src_ids);

    int32_t start = model.hparams.decoder_start_token_id;
    if (start < 0) start = model.hparams.pad_token_id;
    const int32_t eos = model.hparams.eos_token_id;

    std::vector<int32_t> generated;
    if (beam_size <= 1) {
        const int n_layer = (int) model.hparams.dec_layers;
        sussurro_kv_cache cache;
        cache.k.resize(n_layer);
        cache.v.resize(n_layer);
        int32_t cur_tok = start;
        for (int t = 0; t < max_new_tokens; ++t) {
            const std::vector<float> logits = decode_step(model, memory, n_src, cur_tok, t, cache);
            int32_t next = 0;
            float best = logits[0];
            for (int v = 1; v < (int) logits.size(); ++v) {
                if (logits[v] > best) { best = logits[v]; next = v; }
            }
            if (next == eos) break;
            generated.push_back(next);
            cur_tok = next;
        }
    } else {
        generated = beam_search(model, memory, n_src, start, eos, max_new_tokens, beam_size);
    }

    n_decoded = (int) generated.size();
    return sussurro_decode(tok, generated);
}

static std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) ++a;
    while (b > a && std::isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}

static bool is_terminator(char c) { return c == '.' || c == '!' || c == '?'; }

static std::vector<std::string> split_sentences(const std::string & text) {
    std::vector<std::string> out;
    const size_t n = text.size();
    size_t start = 0, i = 0;
    while (i < n) {
        if (is_terminator(text[i])) {
            size_t j = i + 1;
            while (j < n && is_terminator(text[j])) ++j;
            while (j < n && (text[j] == '"' || text[j] == '\'' || text[j] == ')' ||
                             text[j] == ']' || text[j] == '}')) ++j;
            if (j >= n || std::isspace((unsigned char) text[j])) {
                const std::string s = trim(text.substr(start, j - start));
                if (!s.empty()) out.push_back(s);
                while (j < n && std::isspace((unsigned char) text[j])) ++j;
                start = j;
                i = j;
                continue;
            }
            i = j;
        } else {
            ++i;
        }
    }
    if (start < n) {
        const std::string s = trim(text.substr(start));
        if (!s.empty()) out.push_back(s);
    }
    if (out.empty()) {
        const std::string s = trim(text);
        if (!s.empty()) out.push_back(s);
    }
    return out;
}

std::string sussurro_translate(const sussurro_model & model, sussurro_tokenizer * tok,
                               const std::string & source, int max_new_tokens, int beam_size,
                               const std::string & lang_token) {
    const std::vector<std::string> sentences = split_sentences(source);

    const auto t_start = std::chrono::steady_clock::now();
    std::string result;
    int total = 0;
    for (const std::string & s : sentences) {
        int n = 0;
        const std::string piece = translate_sentence(model, tok, s, max_new_tokens, beam_size, n, lang_token);
        total += n;
        if (!piece.empty()) {
            if (!result.empty()) result += ' ';
            result += piece;
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    fprintf(stderr, "[decoded %d tokens in %.0f ms, %.1f tok/s, beam=%d, %zu sentence(s)]\n",
            total, ms, total * 1000.0 / (ms > 0.0 ? ms : 1.0), beam_size, sentences.size());
    return result;
}