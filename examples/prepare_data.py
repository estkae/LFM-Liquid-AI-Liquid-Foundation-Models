"""
Prepare data for fine-tuning LFM model
"""
import json
from pathlib import Path
import argparse
from typing import List, Dict
from loguru import logger


def create_sample_data(output_path: str, num_samples: int = 1000):
    """Create sample training data in JSONL format"""
    
    # Example prompts and completions for different tasks
    examples = [
        # Instruction following
        {
            "instruction": "Explain what machine learning is in simple terms.",
            "response": "Machine learning is a way for computers to learn from data without being explicitly programmed. Instead of following fixed rules, the computer identifies patterns in data and uses them to make predictions or decisions."
        },
        {
            "instruction": "Write a Python function to calculate factorial.",
            "response": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
        },
        {
            "instruction": "Translate 'Hello, how are you?' to German.",
            "response": "Hallo, wie geht es dir?"
        },
        # Q&A format
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "How do neural networks work?",
            "answer": "Neural networks work by processing information through layers of interconnected nodes (neurons). Each connection has a weight that gets adjusted during training to minimize prediction errors."
        },
        # Completion format
        {
            "prompt": "The benefits of regular exercise include",
            "completion": "improved cardiovascular health, stronger muscles and bones, better mental health, increased energy levels, and improved sleep quality."
        },
        # Chat format
        {
            "messages": [
                {"role": "user", "content": "Can you help me debug this Python code?"},
                {"role": "assistant", "content": "Of course! Please share the code you're having trouble with, and describe what error you're encountering or what behavior you're expecting."}
            ]
        }
    ]
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    training_data = []
    
    for i in range(num_samples):
        # Cycle through different example types
        example = examples[i % len(examples)].copy()
        
        # Convert to unified format
        if "instruction" in example:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        elif "question" in example:
            text = f"Q: {example['question']}\nA: {example['answer']}"
        elif "prompt" in example:
            text = f"{example['prompt']} {example['completion']}"
        elif "messages" in example:
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in example['messages']])
        else:
            text = str(example)
        
        training_data.append({"text": text, "id": f"example_{i}"})
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Created {num_samples} training examples in {output_file}")
    return output_file


def prepare_custom_dataset(input_file: str, output_file: str, format_type: str = "instruction"):
    """
    Convert custom dataset to training format
    
    Args:
        input_file: Path to input data (JSON or JSONL)
        output_file: Path to output JSONL file
        format_type: Type of formatting to apply ('instruction', 'chat', 'completion')
    """
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    data = []
    if input_path.suffix == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:  # JSONL
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    
    # Process and format data
    processed_data = []
    
    for item in data:
        if format_type == "instruction":
            # Expect 'instruction' and 'output' fields
            if 'instruction' in item and 'output' in item:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                if 'input' in item and item['input']:
                    text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                logger.warning(f"Skipping item without instruction/output fields: {item}")
                continue
                
        elif format_type == "chat":
            # Expect 'messages' field with list of {role, content}
            if 'messages' in item:
                text = ""
                for msg in item['messages']:
                    text += f"{msg['role']}: {msg['content']}\n"
            else:
                logger.warning(f"Skipping item without messages field: {item}")
                continue
                
        elif format_type == "completion":
            # Expect 'prompt' and 'completion' fields
            if 'prompt' in item and 'completion' in item:
                text = f"{item['prompt']} {item['completion']}"
            else:
                logger.warning(f"Skipping item without prompt/completion fields: {item}")
                continue
        else:
            # Default: use 'text' field or convert to string
            text = item.get('text', str(item))
        
        processed_data.append({
            "text": text,
            "id": item.get('id', f"item_{len(processed_data)}")
        })
    
    # Write processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Processed {len(processed_data)} examples to {output_path}")
    return output_path


def split_dataset(input_file: str, train_ratio: float = 0.9):
    """Split dataset into train and validation sets"""
    
    input_path = Path(input_file)
    
    # Load data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    
    # Shuffle data
    import random
    random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save splits
    train_path = input_path.parent / f"train_{input_path.name}"
    val_path = input_path.parent / f"val_{input_path.name}"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for line in val_data:
            f.write(line + '\n')
    
    logger.info(f"Split dataset into {len(train_data)} train and {len(val_data)} validation examples")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LFM fine-tuning")
    parser.add_argument("--mode", choices=["sample", "custom", "split"], default="sample", 
                        help="Mode: create sample data, process custom data, or split dataset")
    parser.add_argument("--input", type=str, help="Input file for custom mode")
    parser.add_argument("--output", type=str, default="./data/train.jsonl", help="Output file path")
    parser.add_argument("--format", choices=["instruction", "chat", "completion", "text"], 
                        default="instruction", help="Format type for custom mode")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for sample mode")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train ratio for split mode")
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        # Create sample data
        create_sample_data(args.output, args.num_samples)
        
    elif args.mode == "custom":
        # Process custom dataset
        if not args.input:
            raise ValueError("--input required for custom mode")
        prepare_custom_dataset(args.input, args.output, args.format)
        
    elif args.mode == "split":
        # Split dataset
        if not args.input:
            raise ValueError("--input required for split mode")
        split_dataset(args.input, args.train_ratio)
    
    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()