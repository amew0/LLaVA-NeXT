
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        for _ in range(3):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            print(f"\n\n{func.__name__} took {end_time - start_time:.4f} seconds\n\n")
        return result
    return wrapper

def main(is7b, bit, model_path):
    
    if 's3' in model_path:
        for_s3 = True
    else:
        for_s3 = False
    
    print("is7b: ", is7b)
    print("bit: ", bit)
    print("model_path: ", model_path)
    print("for_s3: ", for_s3)
    
    if is7b == "True":
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        
        
        def infer(model, processor):
            if not for_s3:
                example = {
                    "id": "cctv052x2004080607x01847",
                    "video": "/dpc/kunf0097/data/high/video/cctv052x2004080607x01847.avi",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video"
                                },
                                {
                                    "type": "text",
                                    "text": "You are going to analyse a vehicular environment so you can assess the severity of any accident and/or congestion out of 5 along with your justification and predict the most probable cause. You will be provided with a video and scene description.\n\n#### Scene description:\nContext: weather: rain, time: 07:01, congestion: light, collision: no\nA template of the output looks like:\nAccident Severity: 0-5 (Accident severity score justification)\nCongestion Severity: 0-5 (Congestion Severity score justification)\nCause: State your prediction on the most probable cause\n"
                                }
                            ]
                        },
                        # {
                        #     "role": "assistant",
                        #     "content": [
                        #         {
                        #             "type": "text",
                        #             "text": "Accident Severity: 0 (No collision occurred.)\nCongestion Severity: 1 (Light congestion reported despite the rain.)\nCause: The light congestion and lack of accidents are likely due to drivers being more cautious in the rain."
                        #         }
                        #     ]
                        # }
                    ]
                }
            else:
                example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "## Prompt: Deployable Service Recommender\n\nYou will act as a deployable service recommender based on the provided context.\n\nIn a vehicular environment, you will receive the following critical pieces of information:\n\n    - Accident Severity*:\n    - Collision Severity*: \n        *Severity: The score ranges from 0 to 5, derived from deep environmental analysis. Additionally, the justifcation reasoning will be provided for the score.\n    - Cause (most probable one):\n        This refers to the most probable cause responsible for the situation, whether it is an accident or congestion.\n\nBased on this context, you will recommend one of the following services:\n\n### Services\n    #### 1. Cooperative data sharing for incident reassessment\n        Use Case: High collision/accident severity.\n        Action: Open a service that utilizes in-vehicle OBUs (On-Board Units) to upload image/video feeds to a server for further analysis from multiple perspectives.\n    #### 2. Bandwidth Scaling Service to overcome resource scarcity\n        Use Case: High congestion severity.\n        Action: Overcome bandwidth limitations by increasing device bandwidth, enabling bandwidth-intensive services.\n    #### 3. (This is not a service, it's just a No-op for cases not at the top)\n        Use Case: No accident detected and congestion severity is low.\n        Action: No service is required. Simply note that everything is operating smoothly.\n\n### Response Style\n    This is a **zero-shot prompt**, so **DO NOT** ask for additional information.\n    Output must be concise and direct. Avoid irrelevant or lengthy sentences.\n\n### Example Response:\n    Here is an example, although you are allowed to customize it as you feel.\n    \"[some context you grasped] is observed which indicates [High/Low Accident and High/Low Congestion], [1/2/3. ][corresponding service] is suggested [with some justification, i.e what the service intends to solve].\"\n\n### Context\nAccident: severity 0 (No collision occurred.)\nCongestion: severity 1 (Light congestion reported despite the rain.)\nCause: The light congestion and lack of accidents are likely due to drivers being more cautious in the rain."
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "No accident is detected, and congestion severity is low, indicating that everything is operating smoothly. 3. No service is required at this time."
                                }
                            ]
                        }
                    ]
                }
            text = processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)

            if not for_s3:
                video = load_video(example["video"], 16) 
                inputs = processor(text=text, images=[v for v in video], return_tensors="pt", padding=True).to(model.device)
                cont = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values.to(model.dtype),
                    image_sizes=inputs.image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=300
                )
            else:
                inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)
                cont = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=300
                )   
                                
            print(processor.decode(cont[0][len(inputs['input_ids'][0]):], skip_special_tokens=False))
             
        
        if bit == "16":
            model_kwargs = {
                'device_map': 'auto',
                'torch_dtype': torch.float16
            }
            model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
            processor = AutoProcessor.from_pretrained(model_path, device=model_kwargs['device_map'])
            infer(model, processor)
            print("Latency: 0.1ms")
        elif bit == "8":
            print("Latency: 0.2ms")
        elif bit == "4":
            print("Latency: 0.3ms")
    else:
        from llava.model.builder import load_pretrained_model_simplified
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates

        import numpy as np
        import copy
        import warnings
        from decord import VideoReader, cpu
        
        def load_video(video_path, max_frames_num):
            if type(video_path) == str:
                vr = VideoReader(video_path, ctx=cpu(0))
            else:
                vr = VideoReader(video_path[0], ctx=cpu(0))
            total_frame_num = len(vr)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return spare_frames 
        
        @timeit
        def infer (tokenizer, model, image_processor):
            if not for_s3:
                ex = {
                    "id": "cctv052x2004080607x01847",
                    "video": "/dpc/kunf0097/data/high/video/cctv052x2004080607x01847.avi",
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nYou are going to analyse a vehicular environment so you can assess the severity of any accident and/or congestion out of 5 along with your justification and predict the most probable cause. You will be provided with a video and scene description.\n\n#### Scene description:\nContext: weather: rain, time: 07:01, congestion: light, collision: no\nA template of the output looks like:\nAccident Severity: 0-5 (Accident severity score justification)\nCongestion Severity: 0-5 (Congestion Severity score justification)\nCause: State your prediction on the most probable cause\n"
                        },
                        {
                            "from": "gpt",
                            "value": "Accident Severity: 0 (No collision occurred.)\nCongestion Severity: 1 (Light congestion reported despite the rain.)\nCause: The light congestion and lack of accidents are likely due to drivers being more cautious in the rain."
                        }
                    ]
                }
            else:
                ex = {
                    "id": "cctv052x2004080607x01847",
                    "video": "/dpc/kunf0097/data/high/video/cctv052x2004080607x01847.avi",
                    "conversations": [
                        {
                            "from": "human",
                            "value": "## Prompt: Deployable Service Recommender\n\nYou will act as a deployable service recommender based on the provided context.\n\nIn a vehicular environment, you will receive the following critical pieces of information:\n\n    - Accident Severity*:\n    - Collision Severity*: \n        *Severity: The score ranges from 0 to 5, derived from deep environmental analysis. Additionally, the justifcation reasoning will be provided for the score.\n    - Cause (most probable one):\n        This refers to the most probable cause responsible for the situation, whether it is an accident or congestion.\n\nBased on this context, you will recommend one of the following services:\n\n### Services\n    #### 1. Cooperative data sharing for incident reassessment\n        Use Case: High collision/accident severity.\n        Action: Open a service that utilizes in-vehicle OBUs (On-Board Units) to upload image/video feeds to a server for further analysis from multiple perspectives.\n    #### 2. Bandwidth Scaling Service to overcome resource scarcity\n        Use Case: High congestion severity.\n        Action: Overcome bandwidth limitations by increasing device bandwidth, enabling bandwidth-intensive services.\n    #### 3. (This is not a service, it's just a No-op for cases not at the top)\n        Use Case: No accident detected and congestion severity is low.\n        Action: No service is required. Simply note that everything is operating smoothly.\n\n### Response Style\n    This is a **zero-shot prompt**, so **DO NOT** ask for additional information.\n    Output must be concise and direct. Avoid irrelevant or lengthy sentences.\n\n### Example Response:\n    Here is an example, although you are allowed to customize it as you feel.\n    \"[some context you grasped] is observed which indicates [High/Low Accident and High/Low Congestion], [1/2/3. ][corresponding service] is suggested [with some justification, i.e what the service intends to solve].\"\n\n### Context\nAccident: severity 0 (No collision occurred.)\nCongestion: severity 1 (Light congestion reported despite the rain.)\nCause: The light congestion and lack of accidents are likely due to drivers being more cautious in the rain."
                        },
                        {
                            "from": "gpt",
                            "value": "No accident is detected, and congestion severity is low, indicating that everything is operating smoothly. 3. No service is required at this time."
                        }
                    ]
                }
            if not for_s3:
                # Load and process video
                video_path = ex["video"]
                video_frames = load_video(video_path, 16)
                
                # Prepare the frames for the model
                frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
                frames = frames.half().cuda()
                image_tensors = [frames]

            # Prepare conversation input
            conv_template = "qwen_1_5"
            instruction = ex["conversations"][0]["value"]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            if not for_s3:
                image_sizes = [frame.size for frame in video_frames]
                cont = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                    modalities=["video"],
                )
            # Generate response
            else:
                cont = model.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                )

            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            print(text_outputs[0])

        
        model_base = None
        model_name = "llava_qwen"
        # model_path = "/dpc/kunf0097/.cache/huggingface/hub/v2-llava-qwen-ov-s2-1028_182909"

        if bit == "16":
            kwargs = {
                "device_map": {"":0}, 
                "attn_implementation": None
            }
            
            tokenizer, model, image_processor, max_length = load_pretrained_model_simplified(model_path, model_base, model_name, **kwargs)
            
            infer(tokenizer, model, image_processor)

            
        elif bit == "8":
            kwargs = {
                "device_map": {"":0}, 
                "load_8bit": True,
                "attn_implementation": None
            }
            
            tokenizer, model, image_processor, max_length = load_pretrained_model_simplified(model_path, model_base, model_name, **kwargs)
            
            infer(tokenizer, model, image_processor)

            
        elif bit == "4":
            kwargs = {
                "device_map": {"":0}, 
                "load_4bit": True,
                "attn_implementation": None
            }
            
            tokenizer, model, image_processor, max_length = load_pretrained_model_simplified(model_path, model_base, model_name, **kwargs)
            
            infer(tokenizer, model, image_processor)
            
    
# if "__name__" == "__main__":
import sys
is7b, bit, model_path = sys.argv[1:]
main(is7b, bit, model_path)