process TOKENIZE {
    tag "$text"

    input:
    path text

    output:
    path "tokens.json", emit: tokens

    script:
    template "tokenize.py"
}