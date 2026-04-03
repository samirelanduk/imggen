process CLIP_EMBEDDINGS {
    tag "${meta.id}"
    label "process_low"

    input:
    tuple val(meta), path(tokens)
    path clip_model

    output:
    tuple val(meta), path("embeddings.pt"), emit: embeddings
    tuple val("${task.process}"), val('python'), eval('python3 --version | cut -d" " -f2'), topic: versions
    tuple val("${task.process}"), val('pytorch'), eval('python3 -c "import torch; print(torch.__version__)"'), topic: versions
    tuple val("${task.process}"), val('safetensors'), eval('python3 -c "import safetensors; print(safetensors.__version__)"'), topic: versions
    tuple val("${task.process}"), val('numpy'), eval('python3 -c "import numpy; print(numpy.__version__)"'), topic: versions

    script:
    """
    python3 -m pydiffuser clip_embed --tokens ${tokens} --model ${clip_model} --output embeddings.pt
    """
}