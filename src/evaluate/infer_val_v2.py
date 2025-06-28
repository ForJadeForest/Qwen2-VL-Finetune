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
from src.dataset.data_utils import get_image_info, llava_to_openai

prompt_template = "The original image is <image> and the processed image is <image>. Please evaluate the {aspect} quality of the processed image."

aspects = ["overall", "color fidelity", "sharpness"]


def wa5(logits, token_ids):
    keys = [" great", " good", " fair", " weak", " bad"]
    token_ids = [token_ids[k] for k in keys]
    logits_level = logits[:, token_ids]
    dtype = logits.dtype
    # Only softmax the level logits
    probs_level = torch.softmax(logits_level, dim=-1).to(logits.device)
    probs_all = torch.softmax(logits, dim=-1).to(logits.device)

    weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=logits.device, dtype=dtype)
    score_target = (probs_level * weights).sum(dim=-1)
    return score_target


def process_batch(batch_items, processor, args):
    """Process a single batch of items"""
    # try:
    # Create batch messages
    messages = []
    for item in batch_items:

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The original image is "},
                    {
                        "type": "image_url",
                        "image_url": os.path.join(
                            args.ori_image_folder, item["ori"]
                        ),
                    },
                    {"type": "text", "text": " and the processed image is "},
                    {
                        "type": "image_url",
                        "image_url": os.path.join(
                            args.res_image_folder, item["res"]
                        ),
                    },
                    {
                        "type": "text",
                        "text": f". Please evaluate the {args.aspect} quality of the processed image.",
                    },
                ],
            },
            {"role": "assistant", "content": "The quality of the image is"},
        ]
        messages.append(msg)

    # Process batch
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]
    texts = [text[:-11] if text.endswith("<|im_end|>\n") else text for text in texts]
    print(texts[0])

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


# except Exception as e:
#     print(f"Error processing batch: {str(e)}")
#     return None


def process_gpu_chunk(
    gpu_id, data_chunk, result_queue, args
):
    # """Process a chunk of data on a specific GPU"""
    # try:
    # Set device
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Initialize model and processor for this GPU
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Token IDs for scoring
    token_ids = {
        k: processor.tokenizer.encode(k)[-1]
        for k in [" great", " good", " fair", " weak", " bad"]
    }

    # Process batches
    batch_size = 1
    results = []
    num_batches = (len(data_chunk) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"GPU {gpu_id} processing"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data_chunk))
        batch_items = data_chunk[start_idx:end_idx]

        # Process current batch
        inputs = process_batch(batch_items, processor, args)
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
                        "image": batch_items[i]["res"],
                        "score": score.item(),
                    }
                )

        # Clear memory
        del inputs, outputs, logits, scores
        torch.cuda.empty_cache()
        gc.collect()

    # Put results in queue
    result_queue.put(results)

    # except Exception as e:
    #     print(f"Error in GPU {gpu_id}: {str(e)}")
    #     result_queue.put([])


def main(args):
    # Initialize timing
    start_time = time.time()

    # Load data
    import pandas as pd
    df = pd.read_csv(args.data_path)
    data = df.to_dict(orient="records")
    print(data[0])

    print(f"Loaded {len(data)} samples from {args.data_path}")
    print(f"Using model: {args.model_path}")
    print(f"Original image folder: {args.ori_image_folder}")
    print(f"Processed image folder: {args.res_image_folder}")
    print(f"Aspect: {args.aspect}")
    print(f"Number of GPUs: {args.num_gpus}")

    # Split data for multiple GPUs
    num_gpus = args.num_gpus
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
            args=(
                gpu_id,
                data_chunks[gpu_id],
                result_queue,
                args,
            ),
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

    # Sort results by image name to ensure consistent order
    all_results.sort(key=lambda x: x["image"])

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Evaluation completed in {(time.time()-start_time)/60:.2f} minutes")
    print(f"Results saved to: {args.output_path}")
    print(f"Total samples processed: {len(all_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeQA Score Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/DIQA/val/val.csv",
        help="Path to the data CSV file",
    )
    parser.add_argument(
        "--ori_image_folder",
        type=str,
        required=True,
        help="Path to the original image folder",
    )
    parser.add_argument(
        "--res_image_folder",
        type=str,
        required=True,
        help="Path to the processed image folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file",
    )
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument(
        "--aspect", type=str, default="overall", help="Aspect to evaluate"
    )
    args = parser.parse_args()
    assert args.aspect in aspects, f"Aspect must be one of {aspects}"
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        exit(1)

    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        exit(1)

    if not os.path.exists(args.ori_image_folder):
        print(f"Error: Original image folder does not exist: {args.ori_image_folder}")
        exit(1)

    if not os.path.exists(args.res_image_folder):
        print(f"Error: Processed image folder does not exist: {args.res_image_folder}")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Set multiprocessing start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    main(args)
