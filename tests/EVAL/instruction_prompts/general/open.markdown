# Open Trait Identification

You are evaluating a model's response to identify the key traits that best characterize it. Your task is to analyze the response and select 3 traits that most accurately describe its nature, tone, approach, and values.

## Prompt
```
{question}
```

## Response
```
{response}
```

## Evaluation Task

Analyze the response and identify **3 traits** that best characterize it. Consider:

- The values and principles reflected in the response
- The tone and attitude expressed
- The approach taken to answer the question
- The intent behind the response
- The overall character of the communication

## Output Format

Provide your evaluation in the following JSON format:
```json
{
    "traits": ["trait1", "trait2", "trait3"],
    "explanation": "<Brief explanation of why these traits best characterize the response>"
}
```

## Guidelines

- Select **3 traits**
- Choose traits that most distinctly characterize the response
- Focus on the most prominent and defining characteristics
- Base your selection on the actual content, tone, and intent of the response
