include { CLIP_TOKENIZE } from "./modules/local/clip/tokenize"
include { CLIP_EMBEDDINGS } from "./modules/local/clip/embeddings"
include { CLIP_ENCODE } from "./modules/local/clip/encode"
include { CLIP_CONDITION } from "./subworkflows/local/clip_condition"

workflow {

    if (params.prompt && params.prompt_file) {
        error "Please provide either --prompt or --prompt_file, not both."
    }
    if (!params.prompt && !params.prompt_file) {
        error "Please provide either --prompt or --prompt_file."
    }

    def width = params.width ?: (params.height ?: 512)
    def height = params.height ?: width

    def prompt_text = params.prompt_file
        ? file(params.prompt_file, checkIfExists: true).text.trim()
        : params.prompt.trim()

    def id = prompt_text.tokenize().take(3).join('_')
    def meta = [id: id]

    prompt_ch = Channel.of(tuple(meta, prompt_text))
    clip_tokenizer_ch = channel.fromPath("${projectDir}/assets/clip_tokenizer", checkIfExists: true)
    model_ch = channel.fromPath(params.model, checkIfExists: true)

    CLIP_CONDITION(prompt_ch, clip_tokenizer_ch, model_ch)
}