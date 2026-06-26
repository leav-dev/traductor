#include <cstdio>
#include <cstring>
#include <string>

#include "ggml.h"
#include "gguf.h"

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: sussurro-quantize <in.gguf> <out.gguf> [q4_0|q8_0]\n");
        return 1;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    const std::string type_str = argc > 3 ? argv[3] : "q8_0";
    const ggml_type qtype = (type_str == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;

    ggml_context * src_ctx = nullptr;
    gguf_init_params ip = { false, &src_ctx };
    gguf_context * src = gguf_init_from_file(in_path.c_str(), ip);
    if (src == nullptr) { printf("failed to open %s\n", in_path.c_str()); return 1; }

    const int64_t n_tensors = gguf_get_n_tensors(src);
    size_t total = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        total += ggml_nbytes(ggml_get_tensor(src_ctx, gguf_get_tensor_name(src, i)));
    }
    const size_t mem = total + (size_t) (n_tensors + 16) * ggml_tensor_overhead() + 64ull * 1024 * 1024;
    ggml_init_params np = { mem, nullptr, false };
    ggml_context * dst_ctx = ggml_init(np);

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, src);

    int64_t n_quant = 0;
    size_t bytes_in = 0, bytes_out = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(src, i);
        ggml_tensor * t = ggml_get_tensor(src_ctx, name);
        bytes_in += ggml_nbytes(t);

        const bool is_matrix = ggml_n_dims(t) == 2 && t->ne[1] > 1;
        const bool quantizable = is_matrix && (t->ne[0] % 32 == 0) && ends_with(name, ".weight");

        ggml_tensor * nt;
        if (quantizable && t->type == GGML_TYPE_F32) {
            nt = ggml_new_tensor_2d(dst_ctx, qtype, t->ne[0], t->ne[1]);
            ggml_quantize_chunk(qtype, (const float *) t->data, nt->data, 0, t->ne[1], t->ne[0], nullptr);
            ++n_quant;
        } else {
            nt = ggml_new_tensor(dst_ctx, t->type, ggml_n_dims(t), t->ne);
            std::memcpy(nt->data, t->data, ggml_nbytes(t));
        }
        ggml_set_name(nt, name);
        gguf_add_tensor(out, nt);
        bytes_out += ggml_nbytes(nt);
    }

    if (!gguf_write_to_file(out, out_path.c_str(), false)) {
        printf("failed to write %s\n", out_path.c_str());
        return 1;
    }
    printf("quantized %lld/%lld weight tensors to %s\n",
           (long long) n_quant, (long long) n_tensors, type_str.c_str());
    printf("tensor data: %.1f MB -> %.1f MB\n", bytes_in / 1e6, bytes_out / 1e6);

    gguf_free(out);
    gguf_free(src);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    return 0;
}