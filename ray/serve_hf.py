import ray
from ray import serve
from starlette.requests import Request
from typing import Dict

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

import numpy as np
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")

device_map = "auto"
torch_dtype = torch.float16
model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

ray.shutdown()
ray.init(address="auto",ignore_reinit_error=True)  # Connect to Ray or start locally
serve.start()


def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

# 2: Define the Sentiment Analysis Deployment
NUM_REPLICAS = 6
NUM_GPUS = 4.0
@serve.deployment(ray_actor_options={"num_gpus": NUM_GPUS/NUM_REPLICAS},num_replicas=NUM_REPLICAS)
class LlavaQHF:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            model_path, device=device_map, dtype=torch_dtype
        )   
        self.model = AutoModelForVision2Seq.from_pretrained(
           model_path, device_map=device_map, torch_dtype=torch_dtype
        )
    async def __call__(self, request: Request) -> Dict:
        # Extract text from JSON payload
        try:
            example = await request.json()
            # Load and process video
            
            text = self.processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
            video = load_video(example["video"], 16) 
            inputs = self.processor(text=text, images=[v for v in video], return_tensors="pt", padding=True).to(self.model.device)

            cont = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values.to(self.model.dtype),
                image_sizes=inputs.image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=300
            )

            text_output = self.processor.decode(cont[0], skip_special_tokens=False)
            # print(text_output)
            return text_output
        except Exception as e:
            return {"error": str(e)}

# llava_q = LlavaQ.bind()
serve.run(LlavaQHF.bind())

print("LlavaQHF is running at http://localhost:8000/")