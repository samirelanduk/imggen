process TOKENIZE {
    tag "$text"

    input:
    path text

    script:
    template "tokenize.py"
}