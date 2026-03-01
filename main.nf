include { CLIP_TOKENIZE } from "./modules/local/clip/tokenize"
include { CLIP_ENCODE } from "./modules/local/clip/encode"

workflow {

    positive_ch = channel.fromPath(params.positive, checkIfExists: true)
    clip_tokenizer_ch = channel.fromPath("${projectDir}/assets/clip_tokenizer", checkIfExists: true)
    
    CLIP_TOKENIZE(positive_ch, clip_tokenizer_ch)
    CLIP_ENCODE()
}