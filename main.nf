include { TOKENIZE } from "./modules/tokenize"

workflow {

    positive_ch = channel.fromPath(params.positive, checkIfExists: true)

    TOKENIZE(positive_ch)
}