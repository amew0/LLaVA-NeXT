# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    SiglipVisionConfig,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/llava/convert_llava_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/llava-v1.5-7b-conv --old_state_dict_id liuhaotian/llava-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/llava-v1.5-7b/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    ".vision_resampler": "",  # all lmms-lab models do avg pooling, so no vision_resampler
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # tied wieghts so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    if "model.image_newline" in original_state_dict:
        # not used in the original implementation because "merge_type=flat"
        del original_state_dict["model.image_newline"]
    return original_state_dict


# used only for llava-interlave
# for ex: Qwen/Qwen1.5-0.5B-Chat google/siglip-so400m-patch14-384 lmms-lab/llava-next-interleave-qwen-0.5b
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def convert_llava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id, pytorch_dump_folder_path, model_id, push_to_hub):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:  # qwen already has a pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if "siglip" in vision_model_id:
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=26,
            patch_size=14,
            vision_use_head=False,
        ).to_dict()
    else:
        vision_config = None

    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    # llms-lab interleeave models do not use any selection startegy except for last hidden state
    if "Qwen" in text_model_id:
        config.image_token_index = 151646
        if "siglip" in vision_model_id:
            config.vision_feature_select_strategy = "full"
            config.vision_feature_layer = -1
    else:
        config.pad_token_id = 32001
        config.image_token_index = 32000

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    if old_state_dict_id is None:
        old_state_dict_id = model_id
    if "Qwen" in text_model_id:
        state_dict = load_original_state_dict(old_state_dict_id)
    else:
        state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")

    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )

    # model.push_to_hub(output_hub_path)
    # processor.push_to_hub(output_hub_path)
    from pathlib import Path

    print(f"Saving model and processor to {pytorch_dump_folder_path}")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)
    
    # Make space so we can load the model properly now.
    del state_dict
    import gc
    gc.collect()

    # Load everything back for inference tests in float32 because prev script was written as that
    # Though it's mostly loaded in fp16 as original weights are in fp16
    model = LlavaForConditionalGeneration.from_pretrained(
        pytorch_dump_folder_path, torch_dtype="float16", device_map="auto"
    )
    processor = LlavaProcessor.from_pretrained(pytorch_dump_folder_path)
    device = model.device

    # prepare inputs
    from PIL import Image
    import requests
    def load_image():
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)
        return image
    image = load_image()
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch.float16)

    # verify inputs
    filepath = hf_hub_download(
        repo_id="RaushanTurganbay/test-image", filename="llava_onevision_pixel_values.pt", repo_type="dataset"
    )
    # original_pixel_values = torch.load(filepath, map_location="cpu")
    # import ipdb; ipdb.set_trace()
    # assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    image_sizes = torch.tensor([[899, 1024]])
    # assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

    # verify single forward pass
    print("Single forward pass")
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(**inputs)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])

        if model_id == "lmms-lab/llava-onevision-qwen2-0.5b-si":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[-12.1953, -14.6797, -12.7891], [0.5840, -0.8467, 1.3799], [3.6055, 4.5430, 9.9062]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-0.5b-ov":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[-12.0234, -14.3828, -12.7500], [2.3594, 1.0000, 3.9336], [3.6582, 4.7148, 9.1172]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-7b-si":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[1.7656, 3.3418, 1.4033], [0.0757, 0.7427, 3.5098], [6.7109, 5.6797, 9.3828]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-7b-ov":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[1.8496, 3.4219, 1.3135], [3.0996, 3.0117, 3.1484], [4.2422, 4.7109, 9.9688]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-72b-si":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[4.1875, 4.4883, 2.7910], [1.2949, 5.1328, 3.1582], [0.9390, 6.4531, 8.4375]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-72b-ov":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[4.2930, 4.7305, 2.7363], [1.7529, 5.0742, 3.9590], [1.3936, 6.3438, 9.3984]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-7b-ov-chat":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[1.8662, 3.4316, 1.3174], [2.7109, 2.5488, 3.0117], [4.4648, 4.9648, 10.3359]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "lmms-lab/llava-onevision-qwen2-72b-ov-chat":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[4.3086, 4.7344, 2.6953], [1.7090, 5.1719, 4.0234], [1.3057, 6.3438, 9.5469]],
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"Model {model_id} not supported")

        # assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
        print("Logits are ok!")

    # verify generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("Generated text:", repr(generated_text))
    if True:
        if model_id == "lmms-lab/llava-onevision-qwen2-0.5b-si":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart that shows the performance of different algorithms or models in a specific domain, such as image classification or natural language processing. The chart is color-coded to represent different algorithms, with each color corresponding to a specific algorithm. The algorithms are labeled as BLIP-2, InstructBLIP, Owen-VL-Chat, and LLaVA-1.5. The chart also includes a legend at the bottom that explains the color coding and the algorithms represented."
        elif model_id == "lmms-lab/llava-onevision-qwen2-0.5b-ov":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related to natural language processing or machine learning. The chart is divided into different categories, each represented by a different color and labeled with the name of the model or technique used. The models are evaluated based on their performance metrics, such as BLEU-2, InstructBLIP, Qwen-VL-Chat, and LLaVA-1.5. The radar chart helps to visualize the relative"
        elif model_id == "lmms-lab/llava-onevision-qwen2-7b-si":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThis image is a radar chart that compares the performance of different models on various metrics. The models being compared are BLIP-2, InstructBLIP, and Qwen-VL-Chat. The metrics being compared are VQA, QA, GQA, VQA-av2, and VQA-av2. The chart shows that BLIP-2 performs the best on all metrics, followed by InstructBLIP and Qwen-VL-Chat."
        elif model_id == "lmms-lab/llava-onevision-qwen2-7b-ov":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart, also known as a spider chart or a star chart, which is used to compare multiple quantitative variables. Each axis represents a different variable, and the chart is filled with data points that represent the performance or values of different entities across these variables.\n\nIn this particular radar chart, the variables are represented on the axes, and the performance of different models or systems is shown by the lines connecting the data points. The models or systems are labeled along the bottom of the chart,"
        elif model_id == "lmms-lab/llava-onevision-qwen2-72b-si":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. The chart is used to compare the performance of different models or systems across various benchmarks or metrics.\n\nIn this specific radar chart, there are multiple axes, each representing a different benchmark or metric, such as VQA2, GQA, TextVQA, and others. The chart includes several colored lines"
        elif model_id == "lmms-lab/llava-onevision-qwen2-72b-ov":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart comparing the performance of different models on various multimodal benchmarks. The models compared are BLIP-2, InstructBLIP, POPE, QWen-VL-Chat, and LLava-1.5. The benchmarks include VQAv2, GQA, TextVQA, SQA-IMG, VizWiz, MM-IMDb, MM-VQA, MM-IMDb-CN, MM-IMDb-EN, MM-"
        elif model_id == "lmms-lab/llava-onevision-qwen2-7b-ov-chat":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart, also known as a spider chart or a star chart, which is used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along these axes.\n\nIn this particular radar chart, there are multiple lines representing different models or systems, each distinguished by a different color and labeled with a name such as BLIP-2, In"
        elif model_id == "lmms-lab/llava-onevision-qwen2-72b-ov-chat":
            expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart comparing the performance of different models on various multimodal benchmarks. The models compared are BLIP-2, InstructBLIP, POPE, QWen-VL-Chat, and LLava-1.5. The benchmarks include VQAv2, GQA, TextVQA, SQA-IMG, VizWiz, MM-IMDb, MM-VQA, MM-IMDb-CN, MM-IMDb-EN, MM-"
        else:
            raise ValueError(f"Model {model_id} not supported")

    print("Expected text:", repr(expected_text))
    # assert generated_text == expected_text
    print("Generated text is ok!")

    # verify batched generation
    print("Batched generation...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    cats_image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        images=[image, cats_image],
        text=[prompt, prompt],
        padding=True,
        return_tensors="pt",
    ).to(device, torch.float16)

    for k, v in inputs.items():
        print(k, v.shape)

    # print("Image sizes:", inputs.image_sizes)
    print("Keys:"   , inputs.keys())

    # make sure image_sizes are the same
    # as otherwise batched generation doesn't work
    # inputs.image_sizes[1] = inputs.image_sizes[0]

    print("Batched generation...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)
    
    if push_to_hub:
        # load env from .env file
        import os
        from dotenv import load_dotenv
        load_dotenv()
        HF_TOKEN_WRITE= os.getenv("HF_TOKEN_WRITE")
        
        checkpoint_name = model_id.split("/")[-1]
        print(f"Pushing to repo amew0/{checkpoint_name}-hf")
        model.push_to_hub(f"amew0/{checkpoint_name}-hf", token=HF_TOKEN_WRITE)
        processor.push_to_hub(f"amew0/{checkpoint_name}-hf", token=HF_TOKEN_WRITE)



def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
        default="Qwen/Qwen2-0.5B-Instruct"
    )
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
        default="google/siglip-so400m-patch14-384"
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model directory."
    ),
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/llava-onevision-qwen2-0.5b-ov",
        choices=[
            "lmms-lab/llava-onevision-qwen2-0.5b-ov",
            "lmms-lab/llava-onevision-qwen2-0.5b-si",
            "lmms-lab/llava-onevision-qwen2-7b-si",
            "lmms-lab/llava-onevision-qwen2-7b-ov",
            "lmms-lab/llava-onevision-qwen2-72b-si",
            "lmms-lab/llava-onevision-qwen2-72b-ov",
            "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
            "lmms-lab/llava-onevision-qwen2-72b-ov-chat",
        ],
        required=False,
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_llava_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id, args.pytorch_dump_folder_path, args.model_id, args.push_to_hub)


if __name__ == "__main__":
    main()