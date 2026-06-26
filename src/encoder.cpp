#include "sussurro.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

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

std::vector<float> sussurro_run_encoder(const sussurro_model & model, const std::vector<int32_t> & ids) {
    const sussurro_hparams & hp = model.hparams;
    const int n = (int) ids.size();
    const int d = (int) hp.d_model;
    const int n_head = (int) hp.enc_heads;
    const int head_dim = d / n_head;
    const int half = d / 2;

    static const size_t MEM = 256ull * 1024 * 1024;
    ggml_init_params ip = { MEM, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    std::memcpy(tok->data, ids.data(), (size_t) n * sizeof(int32_t));

    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    float * pd = (float *) pos->data;
    for (int t = 0; t < n; ++t) {
        for (int k = 0; k < half; ++k) {
            const float freq = std::pow(10000.0f, -2.0f * (float) k / (float) d);
            pd[(size_t) t * d + k]        = std::sin((float) t * freq);
            pd[(size_t) t * d + half + k] = std::cos((float) t * freq);
        }
    }

    ggml_tensor * cur = ggml_get_rows(ctx, get(model, "model.shared.weight"), tok);
    if (hp.scale_embedding) cur = ggml_scale(ctx, cur, std::sqrt((float) d));
    cur = ggml_add(ctx, cur, pos);

    for (int il = 0; il < (int) hp.enc_layers; ++il) {
        const std::string p = "model.encoder.layers." + std::to_string(il) + ".";
        ggml_tensor * residual = cur;

        ggml_tensor * Q = linear(ctx, get(model, p + "self_attn.q_proj.weight"), get(model, p + "self_attn.q_proj.bias"), cur);
        ggml_tensor * K = linear(ctx, get(model, p + "self_attn.k_proj.weight"), get(model, p + "self_attn.k_proj.bias"), cur);
        ggml_tensor * V = linear(ctx, get(model, p + "self_attn.v_proj.weight"), get(model, p + "self_attn.v_proj.bias"), cur);

        Q = ggml_scale(ctx, Q, 1.0f / std::sqrt((float) head_dim));

        Q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Q, head_dim, n_head, n), 0, 2, 1, 3));
        K = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, K, head_dim, n_head, n), 0, 2, 1, 3));

        ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max(ctx, KQ);

        ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, V, head_dim, n_head, n), 1, 2, 0, 3));
        ggml_tensor * KQV = ggml_mul_mat(ctx, Vt, KQ);
        KQV = ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx, KQV, d, n);

        cur = linear(ctx, get(model, p + "self_attn.out_proj.weight"), get(model, p + "self_attn.out_proj.bias"), cur);
        cur = ggml_add(ctx, cur, residual);
        cur = layer_norm(ctx, cur, get(model, p + "self_attn_layer_norm.weight"), get(model, p + "self_attn_layer_norm.bias"));

        residual = cur;
        cur = linear(ctx, get(model, p + "fc1.weight"), get(model, p + "fc1.bias"), cur);
        cur = activation(ctx, hp.activation, cur);
        cur = linear(ctx, get(model, p + "fc2.weight"), get(model, p + "fc2.bias"), cur);
        cur = ggml_add(ctx, cur, residual);
        cur = layer_norm(ctx, cur, get(model, p + "final_layer_norm.weight"), get(model, p + "final_layer_norm.bias"));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::vector<float> out((size_t) n * d);
    std::memcpy(out.data(), cur->data, out.size() * sizeof(float));

    ggml_free(ctx);
    return out;
}
