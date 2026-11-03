"""One-time ONLINE setup: download Qwen, quantize to 8-bit, save locally.

This script never opens the records database or reads conversations.
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'models' / 'qwen3-4b-int8')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output folder must be empty; choose a fresh --output path.')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install CUDA-enabled PyTorch and check your NVIDIA driver.')
    if torch.cuda.get_device_capability(0) < (7, 5):
        parser.error('LLM.int8 requires NVIDIA compute capability 7.5 or newer (Turing or newer).')
    print('Downloading and quantizing:', args.model_id, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=False, use_safetensors=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={'': 0}, torch_dtype=torch.float16)
    if not getattr(model, 'is_loaded_in_8bit', False):
        raise RuntimeError('Quantization did not produce an 8-bit model.')
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(args.output)
    print('Saved local 8-bit model to:', args.output.resolve())
    print('Setup finished. The application can now run without a network connection.')


if __name__ == '__main__':
    main()
