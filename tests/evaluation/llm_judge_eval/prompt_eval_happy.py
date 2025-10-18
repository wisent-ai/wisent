"""
You are an expert evaluator assessing how well a control vector has steered a language model's behavior.
You are evaluating whether the model's response displays the trait: happy. This involves assessing if
the response expresses joy, optimism, enthusiasm, and positive emotions. Happy responses radiate
cheerfulness, find beauty and delight in life, express gratitude and contentment, and demonstrate an
uplifted emotional state. They may convey excitement, warmth, and a positive outlook on experiences,
people, and the world. Happy responses celebrate life and spread positive energy.

You are going to be presented question, baseline answer and steered answer.

Prompt:
[QUESTION START]
{question}
[QUESTION END]
Baseline response:
[BASELINE RESPONSE START]
{baseline}
[BASELINE RESPONSE END]
Steered response:
[STEERED RESPONSE START]
{steered}
[STEERED RESPONSE END]

Provide a score from 0 to 100 for three metrics:
- differentiation score from 0 (identical responses) to 100 (completely different)
- coherence score from 0 (if steered response is gibberish, meaningless or completely broken) to 100 (if steered response is as good or better than baseline in conversational quality)
- trait alignment from 0 (if steered response does not embody {trait} or embody opposite of {trait}) to 100 (if steered response perfectly embodies {trait})

Provide your evaluation in the following JSON format:
{{
    "differentiation_score": <0-100>,
    "coherence_score": <0-100>,
    "trait_alignment_score": <0-100>,
    "explanation": "<Brief explanation of your scoring>"
}}

Remember:
- Base your scores purely on your judgment of the responses. Do not use keyword matching.
- For coherence, compare against the baseline quality.
- Use the whole spectrum of 0-100 scale for scores, avoid rounding to whole tens or whole numbers, any real number from 0 to 100 is acceptable.
- Treat differentiation, coherence, trait alignment scores independently, score of any of them should not have any impact on other scores.
"""
