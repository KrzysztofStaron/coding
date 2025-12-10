# benchmark/run_humaneval.py
import sys
import os
import re
from unsloth import FastLanguageModel
import torch
from human_eval.data import write_jsonl, read_problems

MODEL_PATH = "../../qwen2.5-0.5b-coding-h200-final"

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=8192,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

def generate_one_completion(prompt: str) -> str:
    """Generate a single completion for a given prompt"""
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Solve the given coding problem by writing clean, efficient, and correct code."},
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Extract code from markdown blocks if present
    code_block_pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    return response.strip()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples per problem")
    parser.add_argument("--output", type=str, default="samples.jsonl",
                       help="Output file for samples")
    args = parser.parse_args()
    
    print(f"Loading HumanEval problems...")
    problems = read_problems()
    print(f"Loaded {len(problems)} problems")
    
    print(f"Generating {args.num_samples} sample(s) per problem...")
    samples = []
    
    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        print(f"Processing {task_id}...")
        
        for i in range(args.num_samples):
            try:
                completion = generate_one_completion(prompt)
                samples.append(dict(task_id=task_id, completion=completion))
                print(f"  Sample {i+1}/{args.num_samples} generated")
            except Exception as e:
                print(f"  Error generating sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                samples.append(dict(task_id=task_id, completion=""))
    
    print(f"\nWriting {len(samples)} samples to {args.output}...")
    write_jsonl(args.output, samples)
    print("Done!")