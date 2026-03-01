include { TOKENIZE } from "./modules/local/tokenize"

workflow {

    positive_ch = channel.fromPath(params.positive, checkIfExists: true)
    clip_tokenizer_ch = channel.fromPath("${projectDir}/assets/clip_tokenizer", checkIfExists: true)
    
    TOKENIZE(positive_ch, clip_tokenizer_ch)
}