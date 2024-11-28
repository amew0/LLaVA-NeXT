import evaluate
from datetime import datetime
from tqdm import tqdm
import json

bert = evaluate.load("bertscore")
gleu = evaluate.load("google_bleu")

class MetricTracker:
    def __init__(self):
        self.bert_p = []
        self.bert_r = []
        self.bert_f1 = []
        self.gleu = []

    def update(self, predicted_deto, label_deto):
        # Calculate scores for this batch
        bert_score = bert.compute(predictions=predicted_deto, references=label_deto, model_type="microsoft/deberta-v3-small")
        gleu_score = gleu.compute(predictions=predicted_deto, references=label_deto)

        self.bert_p.extend(bert_score["precision"])
        self.bert_r.extend(bert_score["recall"])
        self.bert_f1.extend(bert_score["f1"])
        self.gleu.append(gleu_score["google_bleu"])  # gleu avg batch single score returned

    def compute(self, log_for="eval"):
        # Aggregate scores across batches
        mean = lambda l: "{:.3f}".format(sum(l) / len(l)) if l else "0.0"
        result = {
            f"{log_for}/bert_p": mean(self.bert_p),
            f"{log_for}/bert_r": mean(self.bert_r),
            f"{log_for}/bert_f1": mean(self.bert_f1),
            f"{log_for}/gleu": mean(self.gleu),
        }

        # Reset batch statistics
        self.bert_p = []
        self.bert_r = []
        self.bert_f1 = []
        self.gleu = []

        return result


metric_tracker = MetricTracker()

files_to_evaluate = [
    # # 0.5b
    # # base
    # "out/Qwen2-7B-Instruct/s1_test_v2_llava-onevision-qwen2-0.5b-ov.json",
    # "out/Qwen2-7B-Instruct/s2_test_v2_llava-onevision-qwen2-0.5b-ov.json",
    # "out/Qwen2-7B-Instruct/s3_test_v2_llava-onevision-qwen2-0.5b-ov.json",
    # # s1
    # "out/Qwen2-7B-Instruct/s1_test_v2_v2-llava-qwen-ov-s1-1028_125343.json",
    # "out/Qwen2-7B-Instruct/s2_test_v2_v2-llava-qwen-ov-s1-1028_125343.json",
    # # s2
    # "out/Qwen2-7B-Instruct/s2_test_v2_v2-llava-qwen-ov-s2-1028_182909.json",
    # # direct
    # "out/Qwen2-7B-Instruct/s2_test_v2_v2-llava-qwen-ov-direct-1111_185928.json",
    # # s3
    # "out/Qwen2-7B-Instruct/s3_test_v2_v2-llava-qwen-ov-s3-1030_002733.json",
    # # 7b
    # # base
    # "out/Qwen2-7B-Instruct/s1_test_v2_hf-llava-onevision-qwen2-7b-ov-hf.json",
    # "out/Qwen2-7B-Instruct/s2_test_v2_hf-llava-onevision-qwen2-7b-ov-hf.json",
    # "out/Qwen2-7B-Instruct/s3_test_v2_hf-llava-onevision-qwen2-7b-ov-hf.json",
    # # s1
    # "out/Qwen2-7B-Instruct/s1_test_v2_hf_responses_llava-qwen-ov-s1-1027_174437.json",
    # "out/Qwen2-7B-Instruct/s2_test_v2_model_s1_7b-v2-hf-llava-qwen-ov-s1-1027_174437.json",
    # # s2
    # # "out/Qwen2-7B-Instruct/s2_test_v2_7b-v2-hf-llava-qwen-ov-s2-1028_235232.json",
    # "out/Qwen2-7B-Instruct/s2_test_v2_hf_7b-v2-hf-llava-qwen-ov-s2-1126_143318.json", 
    # # direct
    # "/home/kunet.ae/ku5001069/hff/out/s2_test_v2_direct_1125_151101.json",
    # # s3
    # "out/Qwen2-7B-Instruct/s3_test_v2_7b-v2-hf-llava-qwen-ov-s3-1030_122125.json",
    # "out/Qwen2-7B-Instruct/s2_test_v2_v2-llava-qwen-ov-s2-1127_162428.json",
    "out/Qwen2-7B-Instruct/s2_test_v2_v2-llava-qwen-ov-direct-1128_092613.json" # 16 epochs
]


save_path = f"out/out-{datetime.now().strftime('%m%d_%H%M%S')}.json"
with open(save_path, "w") as f:
    json.dump([], f)

results = []

for i, file in tqdm(enumerate(files_to_evaluate), total=len(files_to_evaluate)):
    print(f"File {i+1}/{len(files_to_evaluate)}: {file}")
    with open(file) as f:
        examples = json.load(f)
    preds = [ex["generated"] for ex in examples]
    labels = [ex["expected"] for ex in examples]

    metric_tracker.update(preds, labels)

    result = metric_tracker.compute(log_for="eval")
    print(result)
    results.append({"file": file, **result})

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

print(f"Results saved to {save_path}")
