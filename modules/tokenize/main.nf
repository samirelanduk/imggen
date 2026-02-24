process TOKENIZE {
    tag "$text"

    input:
    path text

    output:
    path "tokens.json", emit: tokens
    path "mapping.csv", emit: mapping

    script:
    template "tokenize.py"
}