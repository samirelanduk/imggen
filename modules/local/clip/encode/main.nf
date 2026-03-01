process CLIP_ENCODE {

    input:
    path tokens
    path clip_model

    script:
    template "encode.py"
}