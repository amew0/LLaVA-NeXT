import ray
from ray import serve
from starlette.requests import Request
from typing import Dict
from transformers import pipeline
import uvicorn

from llava.model.builder import load_pretrained_model, load_pretrained_model_simplified, connect_parent_lm_head
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

import numpy as np
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")

model_base =  None 
model_name = "llava_qwen"
device_map = {"":0}
model_path = "/dpc/kunf0097/.cache/huggingface/hub/llava-qwen-ov-s2-1016_100248"
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
@serve.deployment(ray_actor_options={"num_gpus": 0.22},num_replicas=9)
class LlavaQ:
    def __init__(self):
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model_simplified(model_path, model_base, model_name, device_map=device_map, attn_implementation=None)

    async def __call__(self, request: Request) -> Dict:
        # Extract text from JSON payload
        try:
            ex = await request.json()
            # Load and process video
            
            video_path = ex["video"]
            video_frames = load_video(video_path, 16)
            print(video_frames.shape) # (16, 1024, 576, 3)
            image_tensors = [] 
            frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
            image_tensors.append(frames)

            # Prepare conversation input
            conv_template = "qwen_1_5"
            instruction = ex["conversations"][0]["value"]
            context = ex["conversations"][1]["value"]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            # print(prompt_question)

            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
            image_sizes = [frame.size for frame in video_frames]

            # Generate response
            cont = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
                modalities=["video"],
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            return text_outputs[0]
        except Exception as e:
            return {"error": str(e)}

# llava_q = LlavaQ.bind()
serve.run(LlavaQ.bind())

print("Sentiment analysis service is running at http://localhost:8000/")

# # 4: Keep the service alive
# if __name__ == "__main__":
#     try:
#         while True:
#             pass  # Keeps the process alive indefinitely
#     except KeyboardInterrupt:
#         print("Shutting down service...")
#         ray.shutdown()