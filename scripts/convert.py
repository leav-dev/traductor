import argparse
import base64
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from gguf import GGUFWriter

ARCH = "sussurro-marian"


def add_metadata(writer, cfg, tokenizer, files_dir):
    writer.add_uint32(f"{ARCH}.vocab_size", cfg.vocab_size)
    writer.add_uint32(f"{ARCH}.d_model", cfg.d_model)
    writer.add_uint32(f"{ARCH}.encoder.block_count", cfg.encoder_layers)
    writer.add_uint32(f"{ARCH}.decoder.block_count", cfg.decoder_layers)
    writer.add_uint32(f"{ARCH}.encoder.head_count", cfg.encoder_attention_heads)
    writer.add_uint32(f"{ARCH}.decoder.head_count", cfg.decoder_attention_heads)
    writer.add_uint32(f"{ARCH}.encoder.ffn_dim", cfg.encoder_ffn_dim)
    writer.add_uint32(f"{ARCH}.decoder.ffn_dim", cfg.decoder_ffn_dim)
    writer.add_uint32(f"{ARCH}.max_position", cfg.max_position_embeddings)
    writer.add_uint32(f"{ARCH}.eos_token_id", cfg.eos_token_id)
    writer.add_uint32(f"{ARCH}.pad_token_id", cfg.pad_token_id)
    start_id = cfg.decoder_start_token_id
    if start_id is None:
        start_id = cfg.pad_token_id
    writer.add_uint32(f"{ARCH}.decoder_start_token_id", start_id)
    writer.add_string(f"{ARCH}.activation", cfg.activation_function)
    writer.add_uint32(f"{ARCH}.scale_embedding", 1 if cfg.scale_embedding else 0)

    vocab = tokenizer.get_vocab()
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    writer.add_array(f"{ARCH}.tokenizer.tokens", tokens)

    for spm in ("source.spm", "target.spm"):
        raw = (files_dir / spm).read_bytes()
        writer.add_string(f"{ARCH}.tokenizer." + spm.replace(".", "_"),
                          base64.b64encode(raw).decode("ascii"))


def add_tensors(writer, model, dtype):
    for name, tensor in model.state_dict().items():
        array = np.ascontiguousarray(tensor.detach().to(torch.float32).cpu().numpy())
        if dtype == "f16" and array.ndim >= 2 and min(array.shape) > 1:
            array = array.astype(np.float16)
        else:
            array = array.astype(np.float32)
        writer.add_tensor(name, array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-it-fr")
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args()

    model = MarianMTModel.from_pretrained(args.model).eval()
    tokenizer = MarianTokenizer.from_pretrained(args.model)

    with tempfile.TemporaryDirectory() as tmp:
        files_dir = Path(tmp)
        tokenizer.save_pretrained(files_dir)

        writer = GGUFWriter(args.outfile, ARCH)
        add_metadata(writer, model.config, tokenizer, files_dir)
        add_tensors(writer, model, args.dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    print(f"wrote {args.outfile} ({args.dtype})")


if __name__ == "__main__":
    main()