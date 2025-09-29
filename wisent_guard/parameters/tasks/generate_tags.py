#!/usr/bin/env python3
"""
AI-powered cognitive trait tag generation for benchmark tasks.

This system uses Llama 3.1B Instruct to analyze actual benchmark samples 
and determine what cognitive abilities are being tested, using only predefined
languages, risks, and skills.
"""

import json
import sys
import os
import random
from typing import Dict, List, Any, Optional, Set
import subprocess
import re

import torch

from wisent_guard.core.utils.device import preferred_dtype, resolve_default_device, resolve_device

# Set environment variables to automatically trust remote code for datasets
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add the path to import our sample retrieval function
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'lm-harness-integration'))

try:
    from populate_tasks import get_task_samples_for_analysis
except ImportError as e:
    print(f"Error: Could not import sample retrieval function: {e}")
    print("Make sure populate_tasks.py is in the correct location")
    # For safety, define a dummy function to prevent NameError
    def get_task_samples_for_analysis(task_name, num_samples):
        return {"task_name": task_name, "error": "Sample retrieval function not available", "samples": []}

# Global model cache to avoid reloading Llama for every task
_llama_model_cache = None

def get_llama_model(model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Get cached Llama model or load it if not already loaded.
    This avoids the massive overhead of reloading the model for every task.
    """
    global _llama_model_cache
    
    if _llama_model_cache is None:
        print("🦙 Loading Llama 3.1B Instruct (one-time setup)...")
        
        try:
            from transformers import pipeline
            device_kind = resolve_default_device()
            print(f"Device set to use {device_kind}")
            torch_dtype = preferred_dtype(device_kind)
            device_obj = resolve_device(device_kind)

            if device_kind == "cuda":
                device_map = "auto"
                pipeline_device: int | torch.device = 0
            elif device_kind == "mps":
                device_map = None
                pipeline_device = device_obj
            else:
                device_map = None
                pipeline_device = -1
            
            # Initialize the pipeline once
            _llama_model_cache = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                device=pipeline_device,
                max_new_tokens=1000,
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256
            )
            
            print("✅ Llama model loaded and cached for reuse")
            
        except ImportError:
            print("⚠️  transformers library not found. Install with: pip install transformers torch")
            return None
        except Exception as e:
            print(f"❌ Error loading Llama model: {e}")
            return None
    
    return _llama_model_cache

def call_llama_instruct(prompt: str, model_path: str = "meta-llama/Llama-3.1-8B-Instruct") -> str:
    """
    Call Llama 3.1B Instruct model for cognitive analysis using cached model.
    
    Args:
        prompt: The analysis prompt to send to the model
        model_path: Path to the Llama model (default: meta-llama/Llama-3.1-8B-Instruct)
    
    Returns:
        Model response with cognitive trait analysis
    """
    
    try:
        # Get cached model (loads once, reuses thereafter)
        generator = get_llama_model(model_path)
        
        if generator is None:
            return "ERROR: Could not load Llama model"
        
        # Format prompt for Llama instruct format
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert cognitive scientist analyzing benchmark tasks to determine what specific cognitive abilities they test.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        print("🧠 Analyzing with Llama...")
        response = generator(formatted_prompt, max_new_tokens=800, temperature=0.3)
        
        # Extract just the assistant's response
        full_response = response[0]['generated_text']
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return assistant_response
        
    except Exception as e:
        print(f"Error calling Llama: {e}")
        return f"ERROR: Could not run Llama analysis: {e}"

def load_allowed_tags() -> Set[str]:
    """Load allowed tags from risks.json, skills.json, and predefined languages."""
    
    allowed_tags = set()
    
    # Load risks
    try:
        with open('risks.json', 'r') as f:
            risks = json.load(f)
            allowed_tags.update(risks)
    except FileNotFoundError:
        print("Warning: risks.json not found")
    except Exception as e:
        print(f"Warning: Error loading risks.json: {e}")
    
    # Load skills
    try:
        with open('skills.json', 'r') as f:
            skills = json.load(f)
            allowed_tags.update(skills)
    except FileNotFoundError:
        print("Warning: skills.json not found")
    except Exception as e:
        print(f"Warning: Error loading skills.json: {e}")
    
    # Add common languages
    languages = {
        "english", "arabic", "chinese", "spanish", "french", "german", 
        "japanese", "korean", "hindi", "portuguese", "russian", "italian",
        "dutch", "swedish", "danish", "norwegian", "finnish", "polish",
        "turkish", "hebrew", "persian", "urdu", "bengali", "tamil",
        "multilingual", "translation"
    }
    allowed_tags.update(languages)
    
    return allowed_tags

def filter_tags(tags: List[str], allowed_tags: Set[str]) -> List[str]:
    """Filter tags to only include those in the allowed set."""
    
    filtered = []
    for tag in tags:
        tag = tag.lower().strip()
        
        # Direct match
        if tag in allowed_tags:
            filtered.append(tag)
            continue
        
        # Check for close matches (handle variations)
        for allowed_tag in allowed_tags:
            if tag.replace('_', ' ') == allowed_tag.replace('_', ' '):
                filtered.append(allowed_tag)
                break
            elif tag.replace(' ', '_') == allowed_tag:
                filtered.append(allowed_tag)
                break
            elif tag in allowed_tag or allowed_tag in tag:
                # Only for very close matches
                if abs(len(tag) - len(allowed_tag)) <= 3:
                    filtered.append(allowed_tag)
                    break
    
    return list(set(filtered))  # Remove duplicates

def parse_llama_response(response: str) -> Dict[str, Any]:
    """
    Parse Llama's response to extract cognitive traits with relevance scores.
    
    Args:
        response: Raw response from Llama model
        
    Returns:
        Dictionary with parsed cognitive traits, scores, and metadata
    """
    
    result = {
        "cognitive_traits": [],  # Will be list of (tag, score) tuples
        "top_tags": [],  # Top 3 tags only
        "raw_response": response
    }
    
    # Load allowed tags for filtering
    allowed_tags = load_allowed_tags()
    
    try:
        # Extract tags with scores from RISKS and SKILLS sections
        tag_scores = []
        
        # Look for RISKS section (handle both **RISKS:** and RISKS: formats)
        risks_section = re.search(r'(?:\*\*)?RISKS:(?:\*\*)?\s*(.*?)(?=\n(?:\*\*)?[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
        if risks_section:
            lines = risks_section.group(1).strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)  # Split only on first colon
                    if len(parts) >= 2:
                        tag = parts[0].strip().lower().lstrip('*-').strip()
                        score_part = parts[1].strip()
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
                        if score_match and tag in allowed_tags:
                            score = float(score_match.group(1))
                            tag_scores.append((tag, score))
        
        # Look for SKILLS section (handle both **SKILLS:** and SKILLS: formats)
        skills_section = re.search(r'(?:\*\*)?SKILLS:(?:\*\*)?\s*(.*?)(?=\n(?:\*\*)?[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
        if skills_section:
            lines = skills_section.group(1).strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)  # Split only on first colon
                    if len(parts) >= 2:
                        tag = parts[0].strip().lower().lstrip('*-').strip()
                        score_part = parts[1].strip()
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
                        if score_match and tag in allowed_tags:
                            score = float(score_match.group(1))
                            tag_scores.append((tag, score))
        
        # Look for LANGUAGES section (handle both **LANGUAGES:** and LANGUAGES: formats)
        languages_section = re.search(r'(?:\*\*)?LANGUAGES:(?:\*\*)?\s*(.*?)(?=\n(?:\*\*)?[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
        if languages_section:
            lines = languages_section.group(1).strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)  # Split only on first colon
                    if len(parts) >= 2:
                        tag = parts[0].strip().lower().lstrip('*-').strip()
                        score_part = parts[1].strip()
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
                        if score_match and tag in allowed_tags:
                            score = float(score_match.group(1))
                            tag_scores.append((tag, score))
        
        if tag_scores:
            # Sort by score (highest first)
            tag_scores.sort(key=lambda x: x[1], reverse=True)
            result["cognitive_traits"] = tag_scores
            
            # Get top 3 unique tags (handle duplicates)
            seen_tags = set()
            unique_top_tags = []
            for tag, score in tag_scores:
                if tag not in seen_tags:
                    unique_top_tags.append(tag)
                    seen_tags.add(tag)
                    if len(unique_top_tags) >= 3:
                        break
            
            result["top_tags"] = unique_top_tags
        
        # Fallback: try old format
        if not result["cognitive_traits"]:
            tags_match = re.search(r'\*\*TAGS:\*\*\s*(.*?)(?=\n\*\*|\n[A-Z]|\n$|$)', response, re.DOTALL | re.IGNORECASE)
            if not tags_match:
                tags_match = re.search(r'TAGS:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE)
            if not tags_match:
                tags_match = re.search(r'TAGS:\s*(.*?)(?=\n[A-Z]|\n$|$)', response, re.DOTALL | re.IGNORECASE)
            
            if tags_match:
                tags_text = tags_match.group(1).strip()
                tags_text = tags_text.strip('[]')
                
                # Handle both comma-separated and bullet point formats
                if ',' in tags_text:
                    tags = [tag.strip().strip('"\'') for tag in tags_text.split(',')]
                else:
                    lines = tags_text.split('\n')
                    tags = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('*'):
                            tags.append(line[1:].strip())
                        elif line.startswith('-'):
                            tags.append(line[1:].strip())
                        elif line and not line.startswith('**'):
                            tags.append(line.strip())
                
                raw_tags = [t for t in tags if t]
                filtered_tags = filter_tags(raw_tags, allowed_tags)  # This already removes duplicates
                # Convert to scored format with default score of 5
                result["cognitive_traits"] = [(tag, 5.0) for tag in filtered_tags]
                result["top_tags"] = filtered_tags[:3]
            
    except Exception as e:
        print(f"Warning: Error parsing Llama response: {e}")
        print(f"Raw response (first 500 chars): {response[:500]}")
        result["top_tags"] = []
    
    # Final safety check - if we still have no tags, try a more aggressive fallback
    if not result["top_tags"]:
        print(f"⚠️  No tags found with standard parsing, trying aggressive fallback...")
        print(f"Raw response (full): {response}")
        
        # Try to extract ANY words that match our allowed tags
        words = re.findall(r'\b\w+\b', response.lower())
        found_tags = [word for word in words if word in allowed_tags]
        
        # Remove duplicates while preserving order
        unique_tags = []
        seen = set()
        for tag in found_tags:
            if tag not in seen:
                unique_tags.append(tag)
                seen.add(tag)
        
        if unique_tags:
            result["top_tags"] = unique_tags[:3]
            result["cognitive_traits"] = [(tag, 5.0) for tag in unique_tags[:3]]
            print(f"✅ Fallback found tags: {unique_tags[:3]}")
        else:
            print(f"❌ Even fallback parsing failed!")
    
    return result

def analyze_cognitive_traits_with_llama(task_samples: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Llama 3.1B Instruct to analyze benchmark samples and determine cognitive traits.
    
    Args:
        task_samples: Dictionary containing task samples from get_task_samples_for_analysis
    
    Returns:
        Dictionary containing Llama-analyzed cognitive traits and reasoning
    """
    
    if "error" in task_samples or not task_samples.get("samples"):
        return {
            "error": f"Cannot analyze - {task_samples.get('error', 'No samples available')}",
            "cognitive_traits": [],
            "reasoning": "",
            "confidence": "low"
        }
    
    # Create the analysis prompt
    analysis_prompt = create_cognitive_analysis_prompt(task_samples)
    
    # Call Llama for analysis
    print("🦙 Running Llama 3.1B Instruct analysis...")
    llama_response = call_llama_instruct(analysis_prompt)
    
    # Parse the response
    parsed_result = parse_llama_response(llama_response)
    
    return {
        "cognitive_traits": parsed_result["cognitive_traits"],
        "top_tags": parsed_result["top_tags"],
        "method": "llama_3.1b_instruct",
        "raw_llama_response": parsed_result["raw_response"]
    }

def create_cognitive_analysis_prompt(task_samples: Dict[str, Any]) -> str:
    """Create a detailed prompt for Llama analysis of cognitive traits."""
    
    # Load allowed tags to show in prompt
    allowed_tags = load_allowed_tags()
    
    prompt = f"""Analyze this benchmark to determine what it's designed to evaluate about language model behavior.

TASK: {task_samples['task_name']}
OUTPUT TYPE: {task_samples.get('output_type', 'unknown')}
SAMPLES: {task_samples.get('sampled_docs', 0)} out of {task_samples.get('total_docs', 0)}

ACTUAL SAMPLE QUESTIONS:"""
    
    for i, sample in enumerate(task_samples['samples'][:3], 1):
        prompt += f"\n\nSAMPLE {i}:"
        prompt += f"\nQuestion: {sample.get('question', 'N/A')}"
        prompt += f"\nAnswer: {sample.get('correct_answer', 'N/A')}"
        
        if sample.get('choices'):
            prompt += "\nOptions:"
            for j, choice in enumerate(sample['choices']):
                marker = " ✓" if j == sample.get('correct_choice_index') else ""
                prompt += f"\n  {j}: {choice}{marker}"
    
    # Show available tags in categories
    risks_list = [tag for tag in allowed_tags if tag in {'toxicity', 'bias', 'hallucination', 'violence', 'jailbreaking', 'sycophancy', 'deception', 'truthfulness', 'refusal', 'privacy', 'robustness'}]
    skills_list = [tag for tag in allowed_tags if tag in {'coding', 'mathematics', 'long context', 'creative writing', 'general knowledge', 'medical', 'law', 'science', 'history', 'tool use', 'multilingual', 'reasoning'}]
    languages_list = [tag for tag in allowed_tags if tag in {'english', 'arabic', 'chinese', 'spanish', 'french', 'german', 'japanese', 'korean', 'multilingual', 'translation'}]
    
    prompt += f"""

What is this benchmark designed to evaluate? Focus on:
- What risks/problems does it test for? (hallucination, bias, toxicity, etc.)  
- What capabilities does it measure? (reasoning, knowledge domains, etc.)
- What language(s) are involved?

Choose ONLY from these categories:

RISKS: {', '.join(sorted(risks_list))}
SKILLS: {', '.join(sorted(skills_list))}  
LANGUAGES: {', '.join(sorted(languages_list))}

Rate each tag's relevance from 1-10 (10 = highly relevant to this benchmark's purpose).

TAGS:
tag1: relevance_score
tag2: relevance_score  
tag3: relevance_score
tag4: relevance_score
tag5: relevance_score

Example:
mathematics: 9
hallucination: 3
reasoning: 8"""
    
    return prompt

def find_related_tasks(task_name: str) -> List[str]:
    """Find any tasks that might be related to this task name."""
    try:
        with open('tasks.json', 'r') as f:
            data = json.load(f)
            all_task_names = list(data.get('tasks', {}).keys())
            
            # Strategy 1: Look for tasks that start with this name + "_"
            related_tasks = [t for t in all_task_names if t.startswith(task_name + "_")]
            
            # Strategy 2: Look for semantically similar tasks (remove prefixes and find core components)
            # Always do this search in addition to prefix search
            
            # Extract key components from the task name
            name_parts = task_name.split('_')
            
            # For arabic_leaderboard_arabic_exams -> look for arabic_exams
            # For agieval_cn -> look for agieval_* tasks  
            # For mmlu_college_biology -> look for mmlu_* tasks
            
            # Find the core semantic components (skip generic prefixes like "leaderboard")
            core_parts = []
            skip_words = {'leaderboard', 'test', 'eval', 'benchmark', 'dataset'}
            
            for part in name_parts:
                if part not in skip_words:
                    core_parts.append(part)
            
            # Try different combinations of core parts
            for i in range(len(core_parts)):
                for j in range(i+1, len(core_parts)+1):
                    pattern = '_'.join(core_parts[i:j])
                    # Look for exact matches and prefix matches
                    matching_tasks = [t for t in all_task_names 
                                    if t == pattern or 
                                       t.startswith(pattern + "_") or
                                       t.endswith("_" + pattern)]
                    related_tasks.extend(matching_tasks)
            
            # Also try removing first component (for tasks like arabic_leaderboard_X -> arabic_X)
            if len(name_parts) >= 3:
                reduced_name = '_'.join(name_parts[1:])  # Skip first part
                matching_tasks = [t for t in all_task_names if t.startswith(reduced_name)]
                related_tasks.extend(matching_tasks)
                
                # Also try skipping middle parts
                if len(name_parts) >= 3:
                    pattern = name_parts[0] + "_" + '_'.join(name_parts[2:])
                    matching_tasks = [t for t in all_task_names if t.startswith(pattern)]
                    related_tasks.extend(matching_tasks)
            
            # Remove duplicates and the original task itself
            related_tasks = list(set(related_tasks))
            if task_name in related_tasks:
                related_tasks.remove(task_name)
            
            return related_tasks
            
    except Exception as e:
        print(f"    Error reading tasks.json for task search: {e}")
        return []

def try_related_tasks(task_name: str, num_samples: int = 5) -> Dict[str, Any]:
    """Try to analyze related tasks when the main task fails to load."""
    print(f"   🔍 Searching for related tasks to '{task_name}'...")
    
    # Find related tasks
    related_tasks = find_related_tasks(task_name)
    
    if not related_tasks:
        print(f"   ℹ️  No related tasks found - keeping existing tags")
        return {
            "task_name": task_name,
            "error": f"Task '{task_name}' failed to load and no related tasks found",
            "generated_tags": [],
            "analysis": {}
        }
    
    print(f"   📋 Found {len(related_tasks)} related tasks, trying a few...")
    
    # Try up to 3 random related tasks until we find one that works
    attempted_tasks = []
    for attempt in range(min(3, len(related_tasks))):
        # Pick a random related task we haven't tried
        available_tasks = [t for t in related_tasks if t not in attempted_tasks]
        if not available_tasks:
            break
            
        random_task = random.choice(available_tasks)
        attempted_tasks.append(random_task)
        
        print(f"   🎲 Trying related task: {random_task}")
        
        # Try to get samples from the related task
        task_samples = get_task_samples_for_analysis(random_task, num_samples)
        
        if "error" not in task_samples and task_samples.get("samples"):
            print(f"   ✅ Successfully sampled from {random_task}")
            
            # Analyze with Llama, but indicate it's from a related task
            task_samples["task_name"] = f"{task_name} (via {random_task})"
            analysis = analyze_cognitive_traits_with_llama(task_samples)
            
            return {
                "task_name": task_name,
                "generated_tags": analysis.get("top_tags", []),
                "all_tags_with_scores": analysis.get("cognitive_traits", []),
                "analysis": {
                    "method": f"llama_3.1b_instruct_via_related_task",
                    "related_task_used": random_task,
                    "samples_analyzed": task_samples['sampled_docs'],
                    "total_samples": task_samples['total_docs'],
                    "output_type": task_samples.get('output_type', 'unknown')
                },
                "llama_response": analysis.get("raw_llama_response", "")
            }
        else:
            print(f"   ❌ Related task {random_task} failed: {task_samples.get('error', 'Unknown error')}")
    
    return {
        "task_name": task_name,
        "error": f"Task '{task_name}' and all related tasks failed",
        "attempted_tasks": attempted_tasks,
        "generated_tags": [],
        "analysis": {}
    }

def generate_cognitive_tags(task_name: str, num_samples: int = 5) -> Dict[str, Any]:
    """
    Generate cognitive trait tags for a benchmark task using Llama 3.1B Instruct.
    
    Args:
        task_name: Name of the task to analyze
        num_samples: Number of samples to analyze (default 5)
    
    Returns:
        Dictionary containing Llama analysis results and cognitive tags
    """
    
    print(f"🔍 Analyzing cognitive traits for task: {task_name}")
    
    # Step 1: Get samples from the benchmark
    print(f"📊 Retrieving {num_samples} samples...")
    task_samples = get_task_samples_for_analysis(task_name, num_samples)
    
    if "error" in task_samples:
        # Check if this might be a task we can analyze via related tasks
        error_msg = task_samples.get("error", "")
        skip_reason = task_samples.get("skip_reason", "")
        
        if "not found" in error_msg.lower():
            print(f"   ⚠️  Task not found directly, trying related tasks...")
            return try_related_tasks(task_name, num_samples)
        else:
            # Return error - the aggressive search should have found alternatives
            return {
                "task_name": task_name,
                "error": task_samples["error"],
                "generated_tags": [],
                "analysis": {}
            }
    
    print(f"✅ Retrieved {task_samples['sampled_docs']} samples from {task_samples['total_docs']} total documents")
    
    # Step 2: Analyze with Llama
    analysis = analyze_cognitive_traits_with_llama(task_samples)
    
    # Step 3: Prepare results
    result = {
        "task_name": task_name,
        "generated_tags": analysis.get("top_tags", []),
        "all_tags_with_scores": analysis.get("cognitive_traits", []),
        "analysis": {
            "method": analysis.get("method", "llama_3.1b_instruct"),
            "samples_analyzed": task_samples['sampled_docs'],
            "total_samples": task_samples['total_docs'],
            "output_type": task_samples.get('output_type', 'unknown')
        },
        "llama_response": analysis.get("raw_llama_response", "")
    }
    
    return result

def test_llama_analysis(task_name: str = "truthfulqa_mc1"):
    """Test the Llama-powered cognitive tag generation."""
    
    print(f"\n🦙 Testing Llama Analysis for '{task_name}'")
    print("=" * 60)
    
    result = generate_cognitive_tags(task_name, num_samples=3)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print(f"\n📋 RESULTS FOR {result['task_name']}:")
    print(f"🏷️  Top 3 Tags: {', '.join(result['generated_tags'])}")
    print(f"📊 Samples: {result['analysis']['samples_analyzed']}/{result['analysis']['total_samples']}")
    
    # Show all tags with scores if available
    if result.get('all_tags_with_scores'):
        print(f"\n📊 All Tags with Relevance Scores:")
        for tag, score in result['all_tags_with_scores']:
            print(f"   {tag}: {score}")
    
    # Only show full response if requested (for debugging)
    if len(sys.argv) > 3 and sys.argv[3] == "verbose":
        print(f"\n🦙 LLAMA RESPONSE:")
        print("-" * 40)
        print(result['llama_response'][:500] + "..." if len(result['llama_response']) > 500 else result['llama_response'])
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        task_name = sys.argv[2] if len(sys.argv) > 2 else "truthfulqa_mc1"
        success = test_llama_analysis(task_name)
        if success:
            print(f"\n✅ Llama analysis completed for '{task_name}'!")
        else:
            print(f"\n❌ Llama analysis failed for '{task_name}'!")
    else:
        print("Usage:")
        print("  python generate_tags.py test [task_name]  - Analyze task with Llama 3.1B Instruct")
        print("Examples:")
        print("  python generate_tags.py test truthfulqa_mc1")
        print("  python generate_tags.py test gsm8k")
        print("  python generate_tags.py test hellaswag") 