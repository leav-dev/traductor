#include "sussurro.h"

#include <stdexcept>

#include "sentencepiece_processor.h"

struct sussurro_tokenizer {
    sentencepiece::SentencePieceProcessor source;
    sentencepiece::SentencePieceProcessor target;
    const std::vector<std::string> * id_to_token = nullptr;
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t unk_id = 0;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
};

sussurro_tokenizer * sussurro_tokenizer_create(const sussurro_model & model) {
    auto * tok = new sussurro_tokenizer();

    const std::string src(model.vocab.source_spm.begin(), model.vocab.source_spm.end());
    const std::string tgt(model.vocab.target_spm.begin(), model.vocab.target_spm.end());
    if (!tok->source.LoadFromSerializedProto(src).ok() ||
        !tok->target.LoadFromSerializedProto(tgt).ok()) {
        delete tok;
        throw std::runtime_error("failed to load SentencePiece models");
    }

    tok->id_to_token = &model.vocab.id_to_token;
    for (int32_t id = 0; id < (int32_t) model.vocab.id_to_token.size(); ++id) {
        tok->token_to_id.emplace(model.vocab.id_to_token[id], id);
    }
    auto it = tok->token_to_id.find("<unk>");
    if (it != tok->token_to_id.end()) tok->unk_id = it->second;
    tok->eos_id = model.hparams.eos_token_id;
    tok->pad_id = model.hparams.pad_token_id;

    return tok;
}

void sussurro_tokenizer_free(sussurro_tokenizer * tok) {
    delete tok;
}

std::vector<int32_t> sussurro_encode(sussurro_tokenizer * tok, const std::string & text, const std::string & lang_token) {
    const std::vector<std::string> pieces = tok->source.EncodeAsPieces(text);

    std::vector<int32_t> ids;
    ids.reserve(pieces.size() + 2);
    if (!lang_token.empty()) {
        auto lt = tok->token_to_id.find(lang_token);
        if (lt != tok->token_to_id.end()) ids.push_back(lt->second);
    }
    for (const std::string & piece : pieces) {
        auto it = tok->token_to_id.find(piece);
        ids.push_back(it != tok->token_to_id.end() ? it->second : tok->unk_id);
    }
    if (tok->eos_id >= 0) ids.push_back(tok->eos_id);
    return ids;
}

std::string sussurro_decode(sussurro_tokenizer * tok, const std::vector<int32_t> & ids) {
    const std::vector<std::string> & vocab = *tok->id_to_token;

    std::vector<std::string> pieces;
    pieces.reserve(ids.size());
    for (int32_t id : ids) {
        if (id == tok->eos_id || id == tok->pad_id) continue;
        if (id >= 0 && id < (int32_t) vocab.size()) pieces.push_back(vocab[id]);
    }
    return tok->target.DecodePieces(pieces);
}