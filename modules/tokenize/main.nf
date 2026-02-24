process TOKENIZE {
    tag "$text"

    input:
    path text
    path clip_tokenizer

    output:
    path "tokens.json", emit: tokens
    path "mapping.csv", emit: mapping

    script:
    template "tokenize.py"
}