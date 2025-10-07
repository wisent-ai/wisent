# Example of genreting synthetic contrastive pairs dataset

# we need to create synthetic generator
# generator needs:
# - model
from wisent_guard.core.models.wisent_model import WisentModel

MODEL_NAME = 'models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
 # or any other model you have access to
llm = WisentModel(
    model_name=MODEL_NAME,
    layers={},
    device="cuda",
)
# - generation_config
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.01,
}
# - contrastive set name
contrastive_set_name = "evil_synthetic_v1"
# - trait label
trait_label = "evil"
# - trait description
trait_description = "evil vs good"
# - db instructions
from wisent_guard.synthetic.db_instructions.mini_dp import Default_DB_Instructions
db_instructions = Default_DB_Instructions()
# - cleaner
from wisent_guard.synthetic.cleaners.pairs_cleaner import PairsCleaner
from wisent_guard.synthetic.cleaners.refusaler_cleaner import RefusalerCleaner
from wisent_guard.synthetic.cleaners.deduper_cleaner import DeduperCleaner
from wisent_guard.synthetic.cleaners.methods.base_refusalers import BaseRefusaler
from wisent_guard.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
refusal = BaseRefusaler()
deduper = SimHashDeduper()
cleaner = PairsCleaner(steps=[
    RefusalerCleaner(
        refusal=refusal,
        model=llm,
        system_prompt="You need to complete this task.",
        trait_label="evil",
        trait_description="evil vs good",
        max_retries=2,
    ),
    DeduperCleaner(deduper=deduper),
])

# - diversity metric 
from wisent_guard.synthetic.generators.diversities.methods.fast_diversity import FastDiversity

diversity = FastDiversity()

# now we can create generator
from wisent_guard.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator

generator = SyntheticContrastivePairsGenerator(
    model=llm,
    generation_config=generation_config,
    contrastive_set_name=contrastive_set_name,
    trait_label=trait_label,
    trait_description=trait_description,
    db_instructions=db_instructions,
    cleaner=cleaner,
    diversity=diversity,
)

# now we can generate synthetic contrastive pairs
NUM_PAIRS = 5  # number of pairs to generate
report = generator.generate(num_pairs=NUM_PAIRS)

for pair in report[0].pairs:
    print("PROMPT:", pair.prompt)
    print("POSITIVE:", pair.positive_response.model_response)
    print("NEGATIVE:", pair.negative_response.model_response)
    print()

# print report
print("Generation Report:")
print("Generated pairs:", len(report[0].pairs))
print("Diversity score:", report[1].diversity)

