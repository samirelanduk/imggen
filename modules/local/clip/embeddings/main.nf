process CLIP_EMBEDDINGS {

    input:
    path tokens
    path clip_model

    script:
    template "embeddings.py"
}