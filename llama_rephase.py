#!/usr/bin/env python3
import torch
from transformers import pipeline
from tqdm import tqdm
import json
import argparse
import os
from pathlib import Path


def clean_text(text):
    """
    Clean text by removing content before 'ASSISTANT:' marker.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Find the position of 'ASSISTANT:'
    pos = text.find('ASSISTANT:')
    if pos != -1:
        # Remove the content before 'ASSISTANT:' and strip any leading/trailing whitespace
        text = text[pos + len('ASSISTANT:'):].strip()
    return text


def load_annotations(input_file):
    """
    Load annotations from JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        
    Returns:
        dict: Loaded annotations
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} annotations from {input_file}")
    return data


def setup_model(model_id, device_map="auto"):
    """
    Setup the text generation pipeline.
    
    Args:
        model_id (str): Model identifier
        device_map (str): Device mapping strategy
        
    Returns:
        pipeline: Configured text generation pipeline
    """
    print(f"Loading model: {model_id}")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    print(f"Model loaded on device: {pipe.device}")
    return pipe


def process_annotations(data, pipe, max_tokens=256, target_tokens=77):
    """
    Process annotations by cleaning and rephrasing them.
    
    Args:
        data (dict): Input annotations
        pipe: Text generation pipeline
        max_tokens (int): Maximum tokens for generation
        target_tokens (int): Target tokens for summary
        
    Returns:
        dict: Processed annotations
    """
    annotations_processed = {}
    
    system_prompt = f"you are a text processing assistant, and you help summarize the input text into a single sentence within {target_tokens} tokens while keeping necessary keywords and elements."
    
    for key, value in tqdm(data.items(), desc="Processing annotations"):
        # Clean the text
        cleaned_value = clean_text(value)
        
        # Skip if text is empty after cleaning
        if not cleaned_value.strip():
            annotations_processed[key] = ""
            continue
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cleaned_value},
        ]
        
        try:
            # Generate rephrased text
            outputs = pipe(messages, max_new_tokens=max_tokens)[0]["generated_text"][-1]
            annotations_processed[key] = outputs
        except Exception as e:
            print(f"Error processing {key}: {e}")
            annotations_processed[key] = cleaned_value  # Fallback to original
    
    return annotations_processed


def save_annotations(annotations, output_file):
    """
    Save processed annotations to JSON file.
    
    Args:
        annotations (dict): Processed annotations
        output_file (str): Output file path
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(annotations)} processed annotations to {output_file}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Rephrase annotation text using Llama 3.2 Instruct model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python llama_rephase.py --input annotations_raw.json --output annotations_processed.json
  
  # Custom model and parameters
  python llama_rephase.py --input raw.json --output processed.json --model meta-llama/Llama-3.2-1B-Instruct --max_tokens 200
  
  # Process with different target length
  python llama_rephase.py --input raw.json --output processed.json --target_tokens 50
        """
    )
    
    parser.add_argument("--input", "-i", type=str, default="annotations.json",
                        help="Input JSON file containing annotations")
    parser.add_argument("--output", "-o", type=str, default="annotations_processed.json",
                        help="Output JSON file for processed annotations")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model identifier to use")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--target_tokens", type=int, default=77,
                        help="Target tokens for summary length")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device mapping strategy")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Configuration:")
        print(f"  Input file: {args.input}")
        print(f"  Output file: {args.output}")
        print(f"  Model: {args.model}")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Target tokens: {args.target_tokens}")
        print(f"  Device map: {args.device_map}")
        print()
    
    try:
        # Load input data
        data = load_annotations(args.input)
        
        # Setup model
        pipe = setup_model(args.model, args.device_map)
        
        # Process annotations
        processed_annotations = process_annotations(
            data, pipe, args.max_tokens, args.target_tokens
        )
        
        # Save results
        save_annotations(processed_annotations, args.output)
        
        print(f"\nSuccessfully processed {len(processed_annotations)} annotations!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
