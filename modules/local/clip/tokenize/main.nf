process CLIP_TOKENIZE {
    tag "$text"
    label "process_low"

    input:
    path text
    path clip_tokenizer

    output:
    path "tokens.json", emit: tokens
    path "mapping.csv", emit: mapping

    script:
    template "tokenize.py"
}