process CLIP_TOKENIZE {
    tag "${meta.id}"
    label "process_low"

    input:
    tuple val(meta), val(prompt)
    path clip_tokenizer

    output:
    tuple val(meta), path("tokens.json"), emit: tokens
    tuple val(meta), path("mapping.csv"), emit: mapping
    tuple val("${task.process}"), val('python'), eval('python3 --version | cut -d" " -f2'), topic: versions
    tuple val("${task.process}"), val('transformers'), eval('python3 -c "import transformers; print(transformers.__version__)"'), topic: versions

    script:
    """
    cat << 'PROMPT_EOF' > prompt.txt
${prompt}
PROMPT_EOF
    python3 ${moduleDir}/clip_tokenize.py prompt.txt ${clip_tokenizer}
    """
}