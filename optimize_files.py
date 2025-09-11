# split safetensors files by modules (run after convert_safetensors.py)

# --- Configuration ---
MODEL_DIR = "./high_noise_model" # The folder with the 6 safetensors files
import json
import os
import safetensors.torch
from collections import defaultdict

# --- Configuration ---
# The folder with the 6 safetensors files and the index.json
MODEL_SOURCE_DIR = "./low_noise_model"
# A new folder where the small, optimized files will be saved
MODEL_OUTPUT_DIR = "./low_noise_model_"
# ---------------------

if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)
    print(f"Created output directory: {MODEL_OUTPUT_DIR}")

index_path = os.path.join(MODEL_SOURCE_DIR, "diffusion_pytorch_model.safetensors.index.json")
print(f"Loading original index from: {index_path}")
with open(index_path, 'r') as f:
    weight_map = json.load(f)['weight_map']

part_to_keys = defaultdict(list)
for tensor_key in weight_map.keys():
    part_name = tensor_key.split('.')[0]
    if part_name == "blocks":
        part_name = ".".join(tensor_key.split('.')[:2])
    part_to_keys[part_name].append(tensor_key)
print("Grouped all tensors by their respective model parts.")

for part_name, all_keys_for_part in part_to_keys.items():
    print(f"\n--- Consolidating part: {part_name} ---")

    final_state_dict = {}

    files_needed = defaultdict(list)
    for key in all_keys_for_part:
        filename = weight_map[key]
        files_needed[filename].append(key)

    for filename, keys_in_this_file in files_needed.items():
        source_filepath = os.path.join(MODEL_SOURCE_DIR, filename)
        print(f"  -> Reading {len(keys_in_this_file)} tensor(s) from {filename}")

        with safetensors.safe_open(source_filepath, framework="pt", device="cpu") as f:
            for key in keys_in_this_file:
                final_state_dict[key] = f.get_tensor(key)

    # --- RENAME KEYS BEFORE SAVING ---
    prefix = f"{part_name}."
    renamed_state_dict = {
        key.removeprefix(prefix): tensor
        for key, tensor in final_state_dict.items()
    }
    print(f"  -> Renamed {len(renamed_state_dict)} keys by stripping prefix '{prefix}'")
    # -----------------------------------------------

    output_path = os.path.join(MODEL_OUTPUT_DIR, f"{part_name}.safetensors")

    # Save the RENAMED state dict
    safetensors.torch.save_file(renamed_state_dict, output_path)
    print(f"  => Saved consolidated part to {output_path}")

print("\nPreprocessing complete. All parts are consolidated with corrected keys.")
