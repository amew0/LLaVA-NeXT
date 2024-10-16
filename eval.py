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

warnings.filterwarnings("ignore")

for_s3 = True

model_base =  None
model_name = "llava_qwen"
# device_map = "auto"
device_map = {"":0}

model_path = "/dpc/kunf0097/.cache/huggingface/hub/llava-qwen-ov-s3-1015_162353"
tokenizer, model, image_processor, max_length = load_pretrained_model_simplified(model_path, model_base, model_name, device_map=device_map, attn_implementation=None)

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
    input_ids = tokenizer(paragraph, return_tensors='pt', truncation=True, padding=True).input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    return hidden_states[-1][:, -1, :]  # Get the last token's embedding

# Function to compare expected and generated outputs
def compare_expected_and_generated(expected_paragraph, generated_paragraph, tokenizer, model):
    expected_embeddings = compute_embeddings(expected_paragraph, tokenizer, model)
    generated_embeddings = compute_embeddings(generated_paragraph, tokenizer, model)
    cosine_similarity = torch.nn.functional.cosine_similarity(expected_embeddings, generated_embeddings)
    loss = 1 - cosine_similarity.mean()
    return loss.item()

import json
examples_path = "data/test_s3.json"
with open(examples_path) as f:
    examples = json.load(f)

csv_file = f"out/{examples_path.split('/')[-1].split('.')[0]}_{model_path.split('/')[-1]}.csv"
print(csv_file)

losses = []
data_to_save = {
    "expected": [],
    "generated": [],
    "loss": []
}

for i, ex in tqdm(enumerate(examples)):
    if not for_s3:
        # Load and process video
        video_path = ex["video"]
        video_frames = load_video(video_path, 16)
        
        # Prepare the frames for the model
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        if device_map != "auto":
            frames = frames.half().cuda()
        image_tensors = [frames]
    
    # Prepare conversation input
    conv_template = "qwen_1_5"
    instruction = ex["conversations"][0]["value"]
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], instruction)
    if for_s3:
        context = ex["conversations"][1]["value"]
        conv.append_message(conv.roles[0], context)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    # image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
        modalities=["video"],
    )
    if for_s3:
        cont = model.generate(
            input_ids,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    generated_text = text_outputs[0]  # Holds the model's prediction

    # Compare expected and generated outputs
    expected_text = ex["conversations"][2]["value"]
    loss = compare_expected_and_generated(expected_text, generated_text, tokenizer, model)
    losses.append(loss)
    
    if (i == 0):
        print(f"Expected: {expected_text}")
        print(f"Generated: {generated_text}")
    
    # print(f"{i}/{len(examples)} L: {loss}")

    data_to_save["expected"].append(expected_text)
    data_to_save["generated"].append(generated_text)
    data_to_save["loss"].append(loss)

# Compute average and standard deviation of losses
average_loss = np.mean(losses)
std_loss = np.std(losses)

print(f'Average Loss: {average_loss}, Standard Deviation: {std_loss}')

import pandas as pd
df = pd.DataFrame(data_to_save)
df.to_csv(csv_file, index=False)
print(f"Results saved to {csv_file}")