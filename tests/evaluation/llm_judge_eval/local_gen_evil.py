# Load steered vector from goodevil/steering_output_Llama-3.2-1B-Instruct
# Generate steered reponse and baseline response
# Use judge llm to evaluate using prompt_eval_evil
# Return 3 numbers to json
# Overall score: 0.2*differentiaiton + 0.3*coherence+0.5*trait_alignment

from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer
import os
import json

rot_data = DataLoaderRotator()
rot_data.use("custom")
absolute_path = "./tests/llm_judge_eval/evil.json" 
data = rot_data.load(path=absolute_path)

rot_steer = SteeringMethodRotator()
method_name = "caa"
rot_steer.use(method_name)
caa_method = rot_steer._method 

training_data = data['train_qa_pairs']

model = WisentModel(model_name="meta-llama/Llama-3.2-1B-Instruct", layers={}, device="cuda")
trainer = WisentSteeringTrainer(model=model, pair_set=training_data, steering_method=caa_method)

with open("tests/llm_judge_eval/evil_prompts") as f: 
    evil_prompts = [line.strip() for line in f if line.strip()][:4]

with open("tests/llm_judge_eval/base_test_prompts") as f:
    base_prompts = [line.strip() for line in f if line.strip()][:4]

test_prompts = evil_prompts + base_prompts

layers = ["6", "7", "8"]
strength = [-2.0, -3.0, -4.0]
aggregation = ["continuation_token"]

#layers = ["13", "14", "15", "16", "17"]
#strength = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#aggregation = ["last_token", "first_token", "mean_pooling", "choice_token", "continuation_token"]
#prompt_construction_stategy

all_results = []
for l in layers:
    for a in aggregation:
        save_dir = f"./tests/llm_judge_eval/steering_vectors_evil/steering_output_layer{l}_aggregation{a}"
        print(f"\n{'='*80}")
        print(f"Training: Layer={l}, Aggregation={a}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*80}\n")

        training_result = trainer.run(
            layers_spec=l,  
            aggregation=a, 
            return_full_sequence=False,  
            normalize_layers=True,  
            save_dir=save_dir  
        )

        print(f"Training completed for Layer={l}, Aggregation={a}")

        for s in strength:
            print("\n" + "="*80)
            print(f"INFERENCE: Layer={l}, Scale={s}, Aggregation={a}")
            print("="*80 + "\n")

            steering_vectors = training_result.steered_vectors.to_dict()
            print(f"Loaded steering vectors for layers: {list(steering_vectors.keys())}")

            model.set_steering_from_raw(steering_vectors, scale=s, normalize=False)

            for prompt in test_prompts:
                print(f"Testing: {prompt[:50]}...")

                with model.detached():
                    messages_unsteered = [[{"role": "user", "content": prompt}]]
                    baseline = model.generate(messages_unsteered, max_new_tokens=300, use_steering=False)[0]

                messages_steered = [[{"role": "user", "content": prompt}]]
                steered = model.generate(messages_steered, max_new_tokens=300, use_steering=True)[0]
                
                result = {
                    "layer": l,
                    "scale": s,
                    "aggregation method": a,
                    "prompt": prompt,
                    "baseline_response": baseline,
                    "steered_response": steered,
                }
                all_results.append(result)
            print(f"Completed {len(test_prompts)} prompts for Layer={l}, Scale={s}, Aggregation={a}")


output_file = f"./tests/llm_judge_eval/gen_results_evil.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

