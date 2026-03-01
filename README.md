[![GitHub Actions CI Status](https://github.com/samirelanduk/imggen/actions/workflows/nf-test.yml/badge.svg)](https://github.com/samirelanduk/imggen/actions/workflows/nf-test.yml)
[![nf-test](https://img.shields.io/badge/unit_tests-nf--test-337ab7.svg)](https://www.nf-test.com)

# imggen

imggen is a generative AI pipeline which uses Stable Diffusion to generate images.

## Usage

Create a prompt describing your image, and start the pipeline:

```bash
echo "A beautiful sunset over a mountain lake" > prompt.txt
nextflow run main.nf --positive prompt.txt
```
