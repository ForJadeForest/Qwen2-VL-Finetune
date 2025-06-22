import argparse
import torch
import json
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
import gc
import time
from torch.multiprocessing import Process, Queue


def wa5(logits, token_ids):
    keys = [" five", " four", " three", " two", " one"]
    token_ids = [token_ids[k] for k in keys]
    logits_level = logits[:, token_ids]
    dtype = logits.dtype
    # Only softmax the level logits
    probs_level = torch.softmax(logits_level, dim=-1).to(logits.device)
    probs_all = torch.softmax(logits, dim=-1).to(logits.device)
    print(
        "probs_all[:, token_ids]: ",
        probs_all[:, token_ids],
        probs_all[:, token_ids].sum(dim=-1),
    )
    print("probs_level: ", probs_level, probs_level.sum(dim=-1))

    weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=logits.device, dtype=dtype)
    score_target = (probs_level * weights).sum(dim=-1)
    print("score_target: ", score_target)
    return score_target


def process_batch(batch_items, processor, prompt_template):
    """Process a single batch of items"""
    try:
        # Create batch messages
        messages = []
        for item in batch_items:
            msg = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data/DIQA/val/res/{item['image']}",
                        },
                        {"type": "text", "text": prompt_template},
                    ],
                },
                {"role": "assistant", "content": "The quality of the image is"},
            ]
            messages.append(msg)

        # Process batch
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages
        ]
        texts = [
            text[:-11] if text.endswith("<|im_end|>\n") else text for text in texts
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return None


def process_gpu_chunk(gpu_id, data_chunk, result_queue):
    """Process a chunk of data on a specific GPU"""
    try:
        # Set device
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # Initialize model and processor for this GPU
        processor = AutoProcessor.from_pretrained(
            "output/deqa/checkpoint-110", trust_remote_code=True
        )
        processor.tokenizer.padding_side = 'left'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "output/deqa/checkpoint-110",
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        
        # Token IDs for scoring
        token_ids = {
            k: processor.tokenizer.encode(k)[-1]
            for k in [" five", " four", " three", " two", " one"]
        }
        
        # Process batches
        batch_size = 2
        results = []
        num_batches = (len(data_chunk) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"GPU {gpu_id} processing"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data_chunk))
            batch_items = data_chunk[start_idx:end_idx]
            
            # Process current batch
            inputs = process_batch(batch_items, processor, "Could you evaluate the quality of this image?")
            if inputs is None:
                continue
                
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                    attention_mask=inputs.attention_mask,
                )
                logits = outputs.logits[:, -1]
                scores = wa5(logits, token_ids)
                
                # Store results for this batch
                for i, score in enumerate(scores):
                    results.append(
                        {
                            "image": batch_items[i]["image"],
                            "overall": score.item(),
                        }
                    )
            
            # Clear memory
            del inputs, outputs, logits, scores
            torch.cuda.empty_cache()
            gc.collect()
            
        # Put results in queue
        result_queue.put(results)
        
    except Exception as e:
        print(f"Error in GPU {gpu_id}: {str(e)}")
        result_queue.put([])


def main(args):
    # Initialize timing
    start_time = time.time()
    
    # Load data
    with open("data/val_file/val.json") as f:
        data = json.load(f)
    
    # Split data for 8 GPUs
    num_gpus = 8
    chunk_size = len(data) // num_gpus
    data_chunks = [
        data[i * chunk_size : (i + 1) * chunk_size if i < num_gpus - 1 else len(data)]
        for i in range(num_gpus)
    ]
    
    # Create result queue and processes
    result_queue = Queue()
    processes = []
    
    # Start processes for each GPU
    for gpu_id in range(num_gpus):
        p = Process(
            target=process_gpu_chunk,
            args=(gpu_id, data_chunks[gpu_id], result_queue)
        )
        processes.append(p)
        p.start()
    
    # Collect results
    all_results = []
    for _ in range(num_gpus):
        all_results.extend(result_queue.get())
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Save results
    with open("./data/2epoch_qwen_deqa.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Evaluation completed in {(time.time()-start_time)/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # Set multiprocessing start method
    torch.multiprocessing.set_start_method('spawn')
    main(args)
