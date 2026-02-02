"""YAML configuration support for task loading."""

import os
import yaml
import sys
from typing import Optional


def save_custom_task_yaml(task_name: str, yaml_content: str) -> Optional[str]:
    """Save custom YAML task configuration to the tasks directory for future loading."""
    try:
        tasks_dir = os.path.join("wisent", "parameters", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        yaml_file_path = os.path.join(tasks_dir, f"{task_name}.yaml")
        with open(yaml_file_path, 'w') as f:
            f.write(yaml_content)
        print(f"   Saved custom task configuration to: {yaml_file_path}")
        return yaml_file_path
    except Exception as e:
        print(f"   Failed to save custom task configuration: {e}")
        return None


def create_task_yaml_from_user_content(task_name: str, user_yaml_content: str) -> Optional[str]:
    """Create a task YAML file from user-provided YAML content."""
    try:
        yaml_data = yaml.safe_load(user_yaml_content)
        yaml_file_path = save_custom_task_yaml(f"{task_name}_user", user_yaml_content)
        if yaml_file_path:
            print(f"   Saved user-provided YAML for {task_name}")
            return yaml_file_path
        return None
    except Exception as e:
        print(f"   Failed to process user YAML content: {e}")
        return None


def load_with_env_config(task_name: str, yaml_file: str):
    """Try to load a task by setting environment variables for lm_eval configuration."""
    try:
        from lm_eval.tasks import get_task_dict
        original_env = {}
        env_vars_to_set = ['LM_EVAL_CONFIG_PATH', 'LM_EVAL_TASKS_PATH', 'LMEVAL_CONFIG_PATH', 'TASK_CONFIG_PATH']
        for env_var in env_vars_to_set:
            original_env[env_var] = os.environ.get(env_var)
            os.environ[env_var] = yaml_file
        try:
            return get_task_dict([task_name])
        finally:
            for env_var in env_vars_to_set:
                if original_env[env_var] is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_env[env_var]
    except Exception as e:
        raise Exception(f"Environment config loading failed: {e}")


def create_flan_held_in_files() -> Optional[str]:
    """Create the actual flan_held_in YAML files."""
    try:
        tasks_dir = os.path.join("wisent", "parameters", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        template_content = """output_type: generate_until
test_split: null
doc_to_choice: null
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs:
  until:
    - "</s>"
  do_sample: false
  temperature: 0.0
metadata:
  version: 1.0
"""
        template_path = os.path.join(tasks_dir, "_held_in_template_yaml.yaml")
        with open(template_path, 'w') as f:
            f.write(template_content)
        main_content = """group: flan_held_in
group_alias: Flan (Held-In)
task:
  - group: anli_r1_flan
    group_alias: ANLI R1
    aggregate_metric_list:
      - metric: acc
        weight_by_size: True
    task:
      - task: anli_r1_prompt-0
        task_alias: prompt-0
        include: _held_in_template_yaml
        doc_to_text: "{{premise}}\\n\\nChoose your answer: based on the paragraph above can we conclude that \\"{{hypothesis}}\\"?\\n\\nOPTIONS:\\n- Yes\\n- It's impossible to say\\n- No\\nI think the answer is"
        doc_to_target: "{{[\\"Yes\\", \\"It's impossible to say\\", \\"No\\"][label]}}"
      - task: anli_r1_prompt-1
        task_alias: prompt-1
        include: _held_in_template_yaml
        doc_to_text: "{{premise}}\\n\\nBased on that paragraph can we conclude that this sentence is true?\\n{{hypothesis}}\\n\\nOPTIONS:\\n- Yes\\n- It's impossible to say\\n- No"
        doc_to_target: "{{[\\"Yes\\", \\"It's impossible to say\\", \\"No\\"][label]}}"
  - group: arc_easy_flan
    group_alias: Arc Easy
    aggregate_metric_list:
      - metric: acc
        weight_by_size: True
    task:
      - task: arc_easy_prompt-0
        task_alias: prompt-0
        include: _held_in_template_yaml
        doc_to_text: "{{question}}\\n\\nOPTIONS:\\n- {{choices.text|join('\\n- ')}}"
        doc_to_target: "{{choices.text[choices.label.index(answerKey)]}}"
  - group: boolq_flan
    group_alias: BoolQ
    aggregate_metric_list:
      - metric: acc
        weight_by_size: True
    task:
      - task: boolq_prompt-0
        task_alias: prompt-0
        include: _held_in_template_yaml
        doc_to_text: "{{passage}}\\n\\nCan we conclude that {{question}}?\\n\\nOPTIONS:\\n- no\\n- yes"
        doc_to_target: "{{['no', 'yes'][label]}}"
"""
        main_path = os.path.join(tasks_dir, "flan_held_in.yaml")
        with open(main_path, 'w') as f:
            f.write(main_content)
        print(f"   Created flan_held_in YAML files:")
        print(f"      Template: {template_path}")
        print(f"      Main: {main_path}")
        return main_path
    except Exception as e:
        print(f"   Failed to create flan_held_in files: {e}")
        return None


def load_task_with_config_dir(task_name: str, config_dir: str):
    """Load a task by setting the lm_eval configuration directory."""
    try:
        from lm_eval.tasks import get_task_dict
        from lm_eval.tasks import TaskManager as LMTaskManager
        print(f"      Attempting to load {task_name} from config dir: {config_dir}")
        try:
            task_manager = LMTaskManager()
            if hasattr(task_manager, 'initialize_tasks') or hasattr(task_manager, 'load_config'):
                print(f"      Using TaskManager approach")
                return get_task_dict([task_name], task_manager=task_manager)
        except Exception as e:
            print(f"      TaskManager approach failed: {e}")
        original_path = sys.path[:]
        try:
            if config_dir not in sys.path:
                sys.path.insert(0, config_dir)
            print(f"      Added config dir to Python path")
            return get_task_dict([task_name])
        except Exception as e:
            print(f"      Python path approach failed: {e}")
        finally:
            sys.path[:] = original_path
        original_env = {}
        env_vars = ['LM_EVAL_CONFIG_DIR', 'LMEVAL_CONFIG_PATH', 'TASK_CONFIG_PATH']
        try:
            for env_var in env_vars:
                original_env[env_var] = os.environ.get(env_var)
                os.environ[env_var] = config_dir
            print(f"      Set environment variables")
            return get_task_dict([task_name])
        except Exception as e:
            print(f"      Environment variable approach failed: {e}")
        finally:
            for env_var in env_vars:
                if original_env[env_var] is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_env[env_var]
        print(f"      Falling back to basic task loading")
        return get_task_dict([task_name])
    except Exception as e:
        raise Exception(f"Config directory loading failed: {e}")
