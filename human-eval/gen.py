import sys
import os
import re
from openai import OpenAI
from human_eval.data import write_jsonl, read_problems

# Direct copy of get_response from CodeElo/llm_client.py
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_response(prompt, model):
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stop=["<|eot_id|>", "\n\n---", "\n\n\n", "\n---", "---", '<|im_end|>'],
        max_tokens=4096,
        top_p=0.8,
        temperature=0.7,
        extra_body={
            'repetition_penalty': 1.1,
            'top_k': 20,
        }
    )
    response = chat_response.choices[0].message.content
    return response

def extract_completion(response: str, prompt: str) -> str:
    """Extract just the completion code from the model response."""
    # Try to extract code from markdown code blocks
    code_block_pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        completion = matches[0].strip()
        if prompt.strip() in completion:
            completion = completion.replace(prompt.strip(), "").strip()
        return completion
    
    # If no code block, try to find code after the prompt
    if prompt.strip() in response:
        completion = response.split(prompt.strip())[-1].strip()
    else:
        completion = response.strip()
    
    # Remove common prefixes
    prefixes = ["Here's the solution:", "Solution:", "Here's the code:", "Code:"]
    for prefix in prefixes:
        if completion.startswith(prefix):
            completion = completion[len(prefix):].strip()
    
    return completion

def generate_one_completion(prompt: str, model: str) -> str:
    """Generate a single completion for a given prompt"""
    response = get_response(prompt, model)
    completion = extract_completion(response, prompt)
    return completion

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model name to use")
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples per problem (for pass@k evaluation)")
    parser.add_argument("--output", type=str, default="samples.jsonl",
                       help="Output file for samples")
    args = parser.parse_args()
    
    print(f"Loading HumanEval problems...")
    problems = read_problems()
    print(f"Loaded {len(problems)} problems")
    
    print(f"Generating {args.num_samples} sample(s) per problem using model: {args.model}")
    samples = []
    
    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        print(f"Processing {task_id}...")
        
        for i in range(args.num_samples):
            try:
                completion = generate_one_completion(prompt, args.model)
                samples.append(dict(task_id=task_id, completion=completion))
                print(f"  Sample {i+1}/{args.num_samples} generated")
            except Exception as e:
                print(f"  Error generating sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Add empty completion on error
                samples.append(dict(task_id=task_id, completion=""))
    
    print(f"\nWriting {len(samples)} samples to {args.output}...")
    write_jsonl(args.output, samples)
    print("Done!")

