process CLIP_TOKENIZE {
    tag "$text"
    label "process_low"

    input:
    path text
    path clip_tokenizer

    output:
    path "tokens.json", emit: tokens
    path "mapping.csv", emit: mapping
    tuple val("${task.process}"), val('python'), eval('python3 --version | cut -d" " -f2'), topic: versions
    tuple val("${task.process}"), val('transformers'), eval('python3 -c "import transformers; print(transformers.__version__)"'), topic: versions

    script:
    """
    python3 ${moduleDir}/clip_tokenize.py ${text} ${clip_tokenizer}
    """
}