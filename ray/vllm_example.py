"""
RUN WITH conda vllmm
"""

"""An example showing how to use vLLM to serve multimodal models 
and run online inference with OpenAI client.

Launch the vLLM server with the following command:

(single image inference with Llava)
vllm serve llava-hf/llava-1.5-7b-hf --chat-template template_llava.jinja

(multi-image inference with Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --max-model-len 4096 \
    --trust-remote-code --limit-mm-per-prompt image=2

(audio inference with Ultravox)
vllm serve fixie-ai/ultravox-v0_3 --max-model-len 4096
"""

from openai import OpenAI

# from vllm.assets.audio import AudioAsset

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "rpa_8KVH83Q5DWNXFOGCWN5BKEGEEOYLBG0SMHGCMIPPf78zvv"
# openai_api_base = "http://localhost:8000/v1"
openai_api_base = "https://api.runpod.ai/v2/2vufx7y6uc7pb9/run"

# defaults to os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
)

models = client.models.list()
model = models.data[0].id
print(model)
# import sys
# sys.exit(0) 

{
  "input": 
    { 
        "text": "Hello how are you?",
        "image": "https://www.example.com/image.jpg" 
    },
}

import requests

def encode_base64_content_from_url(content_url: str) -> str:
    import base64
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


# Text-only inference
def run_text_only() -> None:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": "What's the capital of France?"
        }],
        model=model,
        max_tokens=64,
    )

    result = chat_completion.choices[0].message.content
    print("Chat completion output:", result)


# Single-image input inference
def run_single_image() -> None:

    ## Use image url in the payload
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    chat_completion_from_url = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
            ],
        }],
        model=model,
        max_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:", result)

    ## Use base64 encoded image in the payload
    image_base64 = encode_base64_content_from_url(image_url)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        }],
        model=model,
        max_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded image:", result)


# Multi-image input inference
def run_multi_image() -> None:
    image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
    image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"
    chat_completion_from_url = client.chat.completions.create(
       messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the animals in these images?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.youtube.com/watch?v=2ZIpFytCSVc"
                    }
                }
            ]
        }
        ],
        model=model,
        max_tokens=64,
    )
    print('-'*100)
    print(chat_completion_from_url)
    print('-'*100)

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output:", result)


# Audio input inference
def run_audio() -> None:
    # Any format supported by librosa is supported
    audio_url = AudioAsset("winning_call").url

    # Use audio url in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this audio?"
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": audio_url
                    },
                },
            ],
        }],
        model=model,
        max_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from audio url:", result)

    audio_base64 = encode_base64_content_from_url(audio_url)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this audio?"
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        # Any format supported by librosa is supported
                        "url": f"data:audio/ogg;base64,{audio_base64}"
                    },
                },
            ],
        }],
        model=model,
        max_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded audio:", result)


example_function_map = {
    "text-only": run_text_only,
    "single-image": run_single_image,
    "multi-image": run_multi_image,
    "audio": run_audio,
}


def main(args) -> None:
    chat_type = args.chat_type
    example_function_map[chat_type]()


if __name__ == "__main__":
    # from vllm.utils import FlexibleArgumentParser

    # parser = FlexibleArgumentParser(
    #     description='Demo on using OpenAI client for online inference with '
    #     'multimodal language models served with vLLM.')
    
    
    # import parser
    import argparse
    parser = argparse.ArgumentParser(description='Demo on using OpenAI client for online inference with multimodal language models served with vLLM.')
    
    parser.add_argument(
        '--chat-type',
        '-c',
        type=str,
        default="single-image",
        choices=["text-only", "single-image", "multi-image", "audio"],
        help='Conversation type with multimodal data.')
    args = parser.parse_args()
    main(args)