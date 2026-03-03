include { CLIP_TOKENIZE } from "../../../modules/local/clip/tokenize"
include { CLIP_EMBEDDINGS } from "../../../modules/local/clip/embeddings"
include { CLIP_ENCODE } from "../../../modules/local/clip/encode"

workflow CLIP_CONDITION {
    take: 
    prompt
    clip_tokenizer
    clip_model

    main:
    CLIP_TOKENIZE(prompt, clip_tokenizer)
    ch_tokens = CLIP_TOKENIZE.out.tokens

    CLIP_EMBEDDINGS(ch_tokens, clip_model)
    ch_embeddings = CLIP_EMBEDDINGS.out.embeddings

    CLIP_ENCODE(ch_embeddings, clip_model)
    ch_conditioning = CLIP_ENCODE.out.conditioning

    emit:
    conditioning = ch_conditioning

}