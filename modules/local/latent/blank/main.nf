process BLANK_LATENT {
    tag "${meta.id}"
    label "process_low"

    input:
    tuple val(meta), val(width), val(height)

    output:
    tuple val(meta), path("latent.pt"), emit: latent
    tuple val("${task.process}"), val('python'), eval('python3 --version | cut -d" " -f2'), topic: versions
    tuple val("${task.process}"), val('pytorch'), eval('python3 -c "import torch; print(torch.__version__)"'), topic: versions

    script:
    """
    python3 -c "import torch; torch.save(torch.zeros(1, 4, ${height} // 8, ${width} // 8), 'latent.pt')"
    """
}
