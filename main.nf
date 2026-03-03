include { CLIP_TOKENIZE } from "./modules/local/clip/tokenize"
include { CLIP_EMBEDDINGS } from "./modules/local/clip/embeddings"
include { CLIP_ENCODE } from "./modules/local/clip/encode"
include { CLIP_CONDITION } from "./subworkflows/local/clip_condition"

workflow {

    positive_ch = channel.fromPath(params.positive, checkIfExists: true)
    clip_tokenizer_ch = channel.fromPath("${projectDir}/assets/clip_tokenizer", checkIfExists: true)
    model_ch = channel.fromPath(params.model, checkIfExists: true)
    
    CLIP_CONDITION(positive_ch, clip_tokenizer_ch, model_ch)
}