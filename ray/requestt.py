import asyncio
import httpx
import time
import json

URL = "http://localhost:8000"  # Update with your endpoint URL
# REQUEST_PAYLOAD = {
#         "id": "a64f802f-479b-4f6f-a507-389c21e14685",
#         "video": "/dpc/kunf0097/data/hwid12/a64f802f-479b-4f6f-a507-389c21e14685.mp4",
#         "conversations": [
#             {
#                 "from": "human",
#                 "value": "## Prompt: Deployable Service Recommender\n\nYou will act as a deployable service recommender based on the provided context.\n\nIn a vehicular environment, you will receive two critical pieces of information:\n\n    - Severity:\n        If a collision/accident is detected, this score reflects its severity. If not, it relates to the congestion level. The score ranges from 0 to 5, derived from deep environmental analysis. Additionally, the primary factor influencing this score will be provided.\n    - Cause (most probable one):\n        This refers to the most probable cause responsible for the situation, whether it is an accident or congestion.\n\nBased on this context, you will recommend one of the following services:\n\n### Services\n    #### Cooperative data sharing for incident reassessment\n        Use Case: High collision/accident severity.\n        Action: Open a service that utilizes in-vehicle OBUs (On-Board Units) to upload image/video feeds to a server for further analysis from multiple perspectives.\n    #### Bandwidth Scaling Service to overcome resource scarcity\n        Use Case: High congestion severity.\n        Action: Overcome bandwidth limitations by increasing device bandwidth, enabling bandwidth-intensive services.\n    #### (This is not a service, its just a No-op for cases not at the top)\n        Use Case: No accident detected and congestion severity is low.\n        Action: No service is required. Simply note that everything is operating smoothly.\n\n### Response Style\n    This is a **zero-shot prompt**, so **DO NOT** ask for additional information.\n    Output must be concise and direct. Avoid irrelevant or lengthy sentences.\n\n### Example Response:\n    Here is an example, although you are allowed to customize it as you feel.\n    \"[some context you grasped] is observed and the severity score is [the score], [corresponding service] is suggested [with some justification, i.e what the service intends to solve].\""
#             },
#             {
#                 "from": "human",
#                 "value": "<image>Severity: 2 (The collision with a stationary object often results in damage primarily to the vehicle, rather than widespread disruption or severe injuries. The score may increase if the object was significant or caused a hazardous blockage). The most probable cause of the scene might be driver inattention or misjudgment, leading to the collision with a stationary object."
#             },
#             {
#                 "from": "gpt",
#                 "value": "A collision with a stationary object due to driver inattention or misjudgment is observed and the severity score is 2. No specific deployable service is suggested as the scenario does not indicate high severity or significant disruption."
#             }
#         ]
#     }
REQUEST_PAYLOAD = {
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
REQUEST_PAYLOAD_HF = {
        "id": "7517ee86-e83d-4a2f-9004-49d20a85396a",
        "video": "/dpc/kunf0097/data/hwid12/7517ee86-e83d-4a2f-9004-49d20a85396a.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nYou are going to analyse a vehicular environment so you can assess the severity of any accident and/or congestion out of 5 along with your justification and predict the most probable cause. You will be provided with a video, scene description, scene overview, and any observed features of the scene.\n\n#### Scene description:\nCollision: Yes, Involved parties: A vehicle with a stationary object\n\n#### Scene overview:\nA head-on collision between a vehicle and a stationary object was observed with minor damages and minimal congestion impact.\n\n#### Observed features:\ncollision_type: head-on\npedestrian_involvement: no\nroad_type: unknown\nweather_conditions: unknown\ntime_of_day: unknown\nvisibility: likely clear\nvehicle_density: low\nroad_blockages: none observed\nadditional_observation: {'number_of_vehicles_involved': 'one', 'damage_to_infrastructure': 'likely, as a stationary object was hit', 'fire': 'no', 'vehicle_rollover': 'no', 'hazardous_materials': 'no'}\n\nA template of the output looks like:\nAccident Severity: 0-5 (Accident severity score justification)\nCongestion Severity: 0-5 (Congestion Severity score justification)\nCause: State your prediction on the most probable cause\n"
            },
            {
                "from": "gpt",
                "value": "Accident Severity: 2 (A single vehicle collided with a stationary object, causing damage but no pedestrian involvement.)\nCongestion Severity: 1 (Considering the low vehicle density and no significant road blockage, slight congestion may occur.)\nCause: The driver likely misjudged or lost control of the vehicle leading to the collision with a stationary object."
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video"
                    },
                    {
                        "type": "text",
                        "text": "You are going to analyse a vehicular environment so you can assess the severity of any accident and/or congestion out of 5 along with your justification and predict the most probable cause. You will be provided with a video, scene description, scene overview, and any observed features of the scene.\n\n#### Scene description:\nCollision: Yes, Involved parties: A vehicle with a stationary object\n\n#### Scene overview:\nA head-on collision between a vehicle and a stationary object was observed with minor damages and minimal congestion impact.\n\n#### Observed features:\ncollision_type: head-on\npedestrian_involvement: no\nroad_type: unknown\nweather_conditions: unknown\ntime_of_day: unknown\nvisibility: likely clear\nvehicle_density: low\nroad_blockages: none observed\nadditional_observation: {'number_of_vehicles_involved': 'one', 'damage_to_infrastructure': 'likely, as a stationary object was hit', 'fire': 'no', 'vehicle_rollover': 'no', 'hazardous_materials': 'no'}\n\nA template of the output looks like:\nAccident Severity: 0-5 (Accident severity score justification)\nCongestion Severity: 0-5 (Congestion Severity score justification)\nCause: State your prediction on the most probable cause\n"
                    }
                ]
            }
            # {
            #     "role": "assistant",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": "Accident Severity: 2 (A single vehicle collided with a stationary object, causing damage but no pedestrian involvement.)\nCongestion Severity: 1 (Considering the low vehicle density and no significant road blockage, slight congestion may occur.)\nCause: The driver likely misjudged or lost control of the vehicle leading to the collision with a stationary object."
            #         }
            #     ]
            # }
        ]
    }
async def send_request(client: httpx.AsyncClient):
    """Send a single request and return the response time."""
    start_time = time.perf_counter()
    try:
        response = await client.post(URL, json=REQUEST_PAYLOAD, timeout=1000) 
        latency = time.perf_counter() - start_time
        return latency, response.status_code, response.text
    except httpx.ReadTimeout:
        return float('inf'), 408, {"error": "Request timed out"}

async def benchmark(num_requests: int):
    """Send multiple concurrent requests and measure average latency."""
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    # Calculate metrics
    latencies = [res[0] for res in results if res[1] == 200]  # Filter out timeouts
    success_count = sum(1 for _, status, _ in results if status == 200)
    timeout_count = num_requests - success_count

    responses = [res[2] for res in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5 if latencies else None

    d = {
        "total_requests": num_requests,
        "successful_requests": success_count,
        "timed_out_requests": timeout_count,
        "average_latency": avg_latency,
        "std_latency": std_latency,
        "max_latency": max(latencies) if latencies else None,
        "min_latency": min(latencies) if latencies else None,
        "responses": responses
    }
    return d

if __name__ == "__main__":
    # Test different loads (10, 50, 100 concurrent requests)
    import sys
    if len(sys.argv) != 2:
        print("Usage: python request.py <output_json_path>")
        sys.exit(1)
    JSON_PATH = sys.argv[1]
    
    ress = []
    for load in [1,5,25,50,100,75,15,85,35,65,95,45,55,40,60,30,70,20,80,10,90]:
        print(f"\n--- Benchmarking with {load} concurrent requests ---")
        res = asyncio.run(benchmark(load))
        ress.append(res)
        
        with open(JSON_PATH, "w") as f:
            json.dump(ress, f, indent=4)
                # ray start --address='10.126.11.2:6379'
