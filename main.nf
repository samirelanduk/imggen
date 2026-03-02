include { CLIP_TOKENIZE } from "./modules/local/clip/tokenize"
include { CLIP_EMBEDDINGS } from "./modules/local/clip/embeddings"

workflow {

    positive_ch = channel.fromPath(params.positive, checkIfExists: true)
    model_ch = channel.fromPath(params.model, checkIfExists: true)
    clip_tokenizer_ch = channel.fromPath("${projectDir}/assets/clip_tokenizer", checkIfExists: true)
    
    CLIP_TOKENIZE(positive_ch, clip_tokenizer_ch)
    ch_tokens = CLIP_TOKENIZE.out.tokens

    CLIP_EMBEDDINGS(ch_tokens, model_ch)
}