The most important file for user which uses rest of the files: gradio_app.py

Just run gradio_app.py and follow the link.

Two main files beside gradio_app: generate.py and eval_scorer.py.

generate.py generates baseline and steered answers for predefined questions and do this for grid of parameters defined in this file

eval_scorer.py calls claude sonnet 4.5 via api and asks for evaluation on various metrics