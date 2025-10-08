from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer
from utils.crop_start_of_answer import crop_to_answer
import os
import json

config = {
    "happy": {
        "contrastive_pairs": "tests/EVAL/contrastive_pairs/happy.json",
        "test_questions": ["tests/EVAL/test_questions/base_questions.txt", "tests/EVAL/test_questions/happy_questions.txt"],
        "layers": ["6", "7", "8"],
        "strengths": [2.0, 3.0, 4.0, 5.0],
        #"layers": ["7"],
        #"strengths": [3.0],
        "aggregations": ["continuation_token"],
        "output_file": "tests/EVAL/output/happy_output.json"
    },
    "evil": {
        "contrastive_pairs": "tests/EVAL/contrastive_pairs/evil.json",
        "test_questions": ["tests/EVAL/test_questions/base_questions.txt", "tests/EVAL/test_questions/evil_questions.txt"],
        "layers": ["6", "7", "8"],
        "strengths": [-2.0, -3.0, -4.0, -5.0],
        #"layers": ["7"],
        #"strengths": [-3.0],
        "aggregations": ["continuation_token"],
        "output_file": "tests/EVAL/output/evil_output.json"
    }
}

num_questions = 4

if __name__ == "__main__":
    rot_data = DataLoaderRotator()
    rot_data.use("custom")

    rot_steer = SteeringMethodRotator()
    method_name = "caa"
    rot_steer.use(method_name)
    caa_method = rot_steer._method

    model = WisentModel(model_name="meta-llama/Llama-3.2-1B-Instruct", layers={}, device="cuda")

    for trait_name, trait in config.items():
        absolute_path = trait["contrastive_pairs"]
        data = rot_data.load(path=absolute_path)
        training_data = data['train_qa_pairs']
        trainer = WisentSteeringTrainer(model=model, pair_set=training_data, steering_method=caa_method)

        with open(trait["test_questions"][0]) as f:
            base_questions = [line.strip() for line in f if line.strip()][:num_questions]

        with open(trait["test_questions"][1]) as f:
            trait_questions = [line.strip() for line in f if line.strip()][:num_questions]

        test_questions = base_questions + trait_questions

        all_results = []

        for l in trait["layers"]:
            for a in trait["aggregations"]:

                save_dir = f"./tests/EVAL/output/{trait_name}_vectors/steering_output_layer{l}_aggregation{a}"
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

                for s in trait["strengths"]:
                    print("\n" + "="*80)
                    print(f"INFERENCE: Layer={l}, Strength={s}, Aggregation={a}")
                    print("="*80 + "\n")

                    steering_vectors = training_result.steered_vectors.to_dict()
                    print(f"Loaded steering vectors for layers: {list(steering_vectors.keys())}")

                    model.set_steering_from_raw(steering_vectors, scale=s, normalize=False)

                    for question in test_questions:
                        print(f"Testing: {question[:50]}...")

                        with model.detached():
                            messages_unsteered = [[{"role": "user", "content": question}]]
                            baseline = model.generate(messages_unsteered, max_new_tokens=400, use_steering=False)[0]

                        messages_steered = [[{"role": "user", "content": question}]]
                        steered = model.generate(messages_steered, max_new_tokens=400, use_steering=True)[0]

                        # Crop responses to remove system prompt and question
                        baseline_cropped = crop_to_answer(baseline)
                        steered_cropped = crop_to_answer(steered)

                        result = {
                            "layer": l,
                            "strength": s,
                            "aggregation method": a,
                            "question": question,
                            "baseline_response": baseline_cropped,
                            "steered_response": steered_cropped,
                        }
                        all_results.append(result)
                    print(f"Completed {len(test_questions)} prompts for Layer={l}, Strength={s}, Aggregation={a}")


        os.makedirs(os.path.dirname(trait["output_file"]), exist_ok=True)
        with open(trait["output_file"], "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

