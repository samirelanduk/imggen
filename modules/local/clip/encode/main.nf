process CLIP_ENCODE {
    tag "$embeddings"
    label "process_low"

    input:
    path embeddings
    path clip_model

    output:
    path "conditioning.pt", emit: conditioning
    tuple val("${task.process}"), val('python'), eval('python3 --version | cut -d" " -f2'), topic: versions
    tuple val("${task.process}"), val('pytorch'), eval('python3 -c "import torch; print(torch.__version__)"'), topic: versions
    tuple val("${task.process}"), val('safetensors'), eval('python3 -c "import safetensors; print(safetensors.__version__)"'), topic: versions
    tuple val("${task.process}"), val('numpy'), eval('python3 -c "import numpy; print(numpy.__version__)"'), topic: versions

    script:
    """
    python3 ${moduleDir}/clip_encode.py ${embeddings} ${clip_model}
    """
}