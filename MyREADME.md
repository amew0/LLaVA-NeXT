# Project forked from llava-next authors i.e llava-vl

## For Training
- `scripts/train/ft-s*.sh` where `s*` is `s1`, `s2`, `s3`, `s` refers to stage
  - `s1` and `s2` are P (Perception)
  - `s3` is R (Recommendation)
- Configuration (like deepspeed) and other training-related files can be found at `scripts/*`

### Example: `ft_s2.sh`
- Sets environment variables for training
- Defines model versions and paths
- Configures training parameters such as number of GPUs, nodes, and ports
- Launches training using `torchrun` with specified configurations

## For Evaluation
- `eval` folder contains evaluation scripts
  - `eval.py` for models trained from the base model `llava-vl`
  - `eval_hf.py` for models trained on the Hugging Face version of the model

### Example: `eval.py`
- Loads libraries and models
- Defines functions to extract frames from video and compute embeddings
- Compares expected and generated outputs
- Evaluates examples and saves results

## For Inference (Concurrent Requests)
- Uses Ray for handling concurrent requests
- Everything about inference is inside the `ray` folder
- Supports `vllm`, models trained from here, and Hugging Face-based models

### Example: `serve_hf.py`
- Initializes Ray and Ray Serve
- Defines a deployment class `LlavaQHF` for handling inference requests
- Loads and processes video, generates responses using the model, and returns the output

## Folder `_not_very_important`
- Contains files from the fork that weren't used or used to a lesser extent

> README generated with the assist of GPT 4o