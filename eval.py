import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model_simplified
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

import numpy as np
import copy
import warnings
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer


import json
import sys

warnings.filterwarnings("ignore")


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0)) if isinstance(video_path, str) else VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Function to compute embeddings
def compute_embeddings(paragraph, tokenizer, model):
    input_ids = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True).input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    return hidden_states[-1][:, -1, :]  # Get the last token's embedding


# Function to compare expected and generated outputs
def compare_expected_and_generated(expected_paragraph, generated_paragraph, tokenizer, model):
    expected_embeddings = compute_embeddings(expected_paragraph, tokenizer, model)
    generated_embeddings = compute_embeddings(generated_paragraph, tokenizer, model)
    cosine_similarity = torch.nn.functional.cosine_similarity(expected_embeddings, generated_embeddings)
    loss = cosine_similarity.mean()
    return loss.item()


for_s3 = False
already_generated = False

# jtokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
# jmodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto")

example_paths = ["data/s2/s2_test_v2.json"]
model_path = "/dpc/kunf0097/.cache/huggingface/hub/v2-llava-qwen-ov-s1s2-1128_232222"
for examples_path in example_paths:
    save_file = f"out/Qwen2-7B-Instruct/{examples_path.split('/')[-1].split('.')[0]}_{model_path.split('/')[-1]}.json"
    print(f"\n\nExamples to evaluate: {examples_path}")
    print(f"Model path: {model_path}")
    print(f"Save path: {save_file}\n\n")

    with open(examples_path) as f:
        examples = json.load(f)
        print(f"Examples loaded: {len(examples)}")

    if not already_generated:
        model_base = None
        model_name = "llava_qwen"
        # device_map = "auto"
        device_map = {"": 0}
        tokenizer, model, image_processor, max_length = load_pretrained_model_simplified(model_path, model_base, model_name, device_map=device_map, attn_implementation=None)

    losses = []
    json_data = []

    for i, ex in tqdm(enumerate(examples), total=len(examples), ncols=100):
        if not already_generated:
            if not for_s3:
                # Load and process video
                video_path = ex["video"]
                video_frames = load_video(video_path, 16)

                # Prepare the frames for the model
                frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
                if device_map != "auto" or True:
                    frames = frames.half().cuda()
                image_tensors = [frames]

            # Prepare conversation input
            conv_template = "qwen_1_5"
            instruction = ex["conversations"][0]["value"]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], instruction)
            # if for_s3:
            #     context = ex["conversations"][1]["value"]
            #     conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

            if not for_s3:
                image_sizes = [frame.size for frame in video_frames]

            # Generate response
            if for_s3:
                cont = model.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                )
            else:
                cont = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                    modalities=["video"],
                )

            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            generated_text = text_outputs[0]  # Holds the model's prediction

            # Compare expected and generated outputs
            # if for_s3:
            #     expected_text = ex["conversations"][2]["value"]
            # else:
            expected_text = ex["conversations"][1]["value"]

        else:
            expected_text = ex["expected"]
            generated_text = ex["generated"]
        # loss = compare_expected_and_generated(expected_text, generated_text, jtokenizer, jmodel)
        # losses.append(loss)

        if i == 0:
            print(f"## Expected:\n{expected_text}")
            print(f"## Generated:\n{generated_text}")

        json_data.append(
            {
                "id": ex["id"],
                "expected": expected_text,
                "generated": generated_text,
                # "loss": loss
            }
        )

        with open(save_file, "w") as f:
            json.dump(json_data, f, indent=4)

    # Compute average and standard deviation of losses
    # average_loss = np.mean(losses)
    # std_loss = np.std(losses)

    # print(f'Average Loss, Standard Deviation: ({average_loss:.4f},  {std_loss:.4f})')
    print(f"Results saved to {save_file}")
