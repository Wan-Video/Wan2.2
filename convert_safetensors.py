# convert high_noise_model and low_noise_model to bfloat16 to fit one block in 8GB VRAM

import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import os


def convert_file(input_path: str, output_path: str, target_dtype_str: str):
    """
    Loads a safetensors file, converts all floating-point tensors to a new
    data type, and saves the result to a new file.

    Args:
        input_path (str): Path to the source .safetensors file.
        output_path (str): Path to save the converted .safetensors file.
        target_dtype_str (str): The target dtype as a string (e.g., 'bfloat16').
    """
    # Mapping from string representation to torch.dtype object
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }

    target_dtype = dtype_map.get(target_dtype_str)
    if target_dtype is None:
        raise ValueError(
            f"Unsupported dtype '{target_dtype_str}'. "
            f"Supported dtypes are: {list(dtype_map.keys())}"
        )

    print(f"Loading safetensors file from: {input_path}")
    print(f"Converting floating-point tensors to: {target_dtype_str}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converted_tensors = {}

    # Use safe_open for memory-efficient loading
    with safe_open(input_path, framework="pt", device="cpu") as f:
        # Get all tensor keys and wrap with tqdm for a progress bar
        tensor_keys = f.keys()
        for key in tqdm(tensor_keys, desc="Converting tensors"):
            tensor = f.get_tensor(key)

            # Check if the tensor's dtype is a floating point type
            if tensor.dtype.is_floating_point:
                # Convert the tensor to the target dtype
                converted_tensors[key] = tensor.to(target_dtype)
            else:
                # If not a float, keep the original tensor
                converted_tensors[key] = tensor

    print("Saving converted tensors...")
    # Save the new dictionary of tensors to the output file
    save_file(converted_tensors, output_path)

    print(f"\nSuccessfully saved converted file to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a .safetensors file to a new data type, "
                    "only affecting floating-point tensors."
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input .safetensors file.'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the new converted .safetensors file.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        required=True,
        choices=['float32', 'float16', 'bfloat16'],
        help="The target data type for floating-point tensors."
    )

    args = parser.parse_args()

    convert_file(args.input, args.output, args.dtype)


# python convert_safetensors.py --input "./low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors" --output "./low_noise_model/diffusion_pytorch_model-00001-of-00006_.safetensors" --dtype "bfloat16"