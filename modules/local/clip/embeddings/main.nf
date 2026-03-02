process CLIP_EMBEDDINGS {
    tag "$tokens"
    label "process_low"

    input:
    path tokens
    path clip_model

    output:
    path "embeddings.pt", emit: embeddings

    script:
    template "embeddings.py"
}