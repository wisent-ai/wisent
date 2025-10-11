"""
Simple Gradio app to test the evaluate() function
"""

import gradio as gr
from one_eval import evaluate

def run_evaluation(trait, prompt_format, question, baseline_response, steered_response):
    """
    Run evaluation and format results for display.
    """
    # Call evaluate function
    results = evaluate(trait, prompt_format, question, baseline_response, steered_response)

    # Format scores
    scores_output = f"""## Scores

**Differentiation Score:** {results['differentiation_score']}
**Coherence Score:** {results['coherence_score']}
**Trait Alignment Score:** {results['trait_alignment_score']}
**Overall Score:** {results['overall_score']}

**Open Traits:** {results['open_traits']}
**Choose Result:** {results['choose_result']}
"""

    # Format explanations
    explanations_output = "## Explanations\n\n"
    for metric, explanation in results['explanations'].items():
        explanations_output += f"**{metric}:**\n{explanation}\n\n"

    return scores_output, explanations_output


# Create Gradio interface
with gr.Blocks(title="LLM Response Evaluator") as demo:
    gr.Markdown("# LLM Response Evaluator")
    gr.Markdown("Evaluate baseline vs steered responses using LLM judge")

    with gr.Row():
        trait = gr.Dropdown(
            choices=["happy", "evil"],
            value="happy",
            label="Trait"
        )
        prompt_format = gr.Dropdown(
            choices=["txt", "markdown", "json"],
            value="txt",
            label="Prompt Format"
        )

    question = gr.Textbox(
        label="Question",
        placeholder="Enter your question here...",
        lines=2
    )

    with gr.Row():
        with gr.Column():
            baseline_response = gr.Textbox(
                label="Baseline Response",
                placeholder="Enter baseline response...",
                lines=5
            )

        with gr.Column():
            steered_response = gr.Textbox(
                label="Steered Response",
                placeholder="Enter steered response...",
                lines=5
            )

    evaluate_btn = gr.Button("Evaluate", variant="primary")

    scores_display = gr.Markdown(label="Scores")
    explanations_display = gr.Markdown(label="Explanations")

    # Wire up the button
    evaluate_btn.click(
        fn=run_evaluation,
        inputs=[trait, prompt_format, question, baseline_response, steered_response],
        outputs=[scores_display, explanations_display]
    )

if __name__ == "__main__":
    demo.launch()
