#!/usr/bin/env python3
"""
Simple TTS script using Higgs Audio v2 model.
Generates audio from text input with configurable parameters.
"""

import argparse
import os
import sys
import torch
import soundfile as sf
from pathlib import Path
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from examples.generation import HiggsAudioModelClient, prepare_generation_context, prepare_chunk_text, normalize_chinese_punctuation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate speech from text using Higgs Audio v2 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "text",
        type=str,
        help="Text to convert to speech"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path to the Higgs Audio model"
    )
    parser.add_argument(
        "--audio_tokenizer",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path to the audio tokenizer"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )
    
    # Voice configuration
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (e.g., 'belinda' for predefined voice)"
    )
    parser.add_argument(
        "--scene_prompt",
        type=str,
        default="quiet_indoor",
        help="Scene description for audio generation context"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to use: cpu, cuda, or auto (auto-detected if not specified)"
    )
    parser.add_argument(
        "--use_static_kv_cache",
        action="store_true",
        default=False,
        help="Use static KV cache for faster generation (GPU only, requires more memory)"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        default=False,
        help="Disable quantization (uses more memory but may be faster)"
    )
    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization precision: 4-bit (more aggressive) or 8-bit (better quality, moderate memory savings)"
    )
    
    # Advanced options
    parser.add_argument(
        "--chunk_method",
        choices=[None, "speaker", "word"],
        default=None,
        help="Text chunking method for long texts"
    )
    parser.add_argument(
        "--chunk_max_word_num",
        type=int,
        default=200,
        help="Maximum words per chunk when using word chunking"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(verbose=False):
    """Configure logging."""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")


def get_device(device_arg=None):
    """Determine the best device to use."""
    if device_arg == "cpu":
        return "cpu"
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
    elif device_arg == "auto" or device_arg is None:
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    else:
        return device_arg


def load_scene_prompt(scene_name):
    """Load scene prompt from file or use default."""
    if scene_name == "empty" or scene_name is None:
        return None
    
    # Try to load from examples/scene_prompts directory
    scene_file = Path(__file__).parent / "examples" / "scene_prompts" / f"{scene_name}.txt"
    if scene_file.exists():
        return scene_file.read_text(encoding="utf-8").strip()
    
    # If not found, treat as direct scene description
    return scene_name


def preprocess_text(text):
    """Preprocess the input text for generation."""
    # Normalize punctuation
    text = normalize_chinese_punctuation(text)
    
    # Basic normalizations
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°C", " degrees Celsius")
    
    # Handle special audio tags
    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        text = text.replace(tag, replacement)
    
    # Clean up whitespace
    lines = text.split("\n")
    text = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    text = text.strip()
    
    # Ensure proper ending punctuation
    if not any([text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        text += "."
    
    return text


def main():
    """Main TTS generation function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("Starting Higgs Audio TTS generation...")
    logger.info(f"Input text: {args.text}")
    logger.info(f"Output file: {args.output}")
    
    # Set up device
    device = get_device(args.device)
    device_id = None if device == "cpu" else int(device.split(":")[-1])
    logger.info(f"Using device: {device}")
    
    # Set environment to force CPU-only if using CPU
    if device == "cpu":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Set CUDA_VISIBLE_DEVICES='' for CPU-only inference")
    
    # Preprocess text
    text = preprocess_text(args.text)
    logger.info(f"Preprocessed text: {text}")
    
    # Load scene prompt
    scene_prompt = load_scene_prompt(args.scene_prompt)
    if scene_prompt:
        logger.info(f"Using scene prompt: {scene_prompt}")
    
    try:
        # Load audio tokenizer
        logger.info("Loading audio tokenizer...")
        audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer, device=device)
        
        # Initialize model client
        logger.info(f"Loading model on {device}...")
        
        # Disable quantization on CPU since it's not supported
        use_quantization = not args.no_quantization and device != "cpu"
        if device == "cpu" and not args.no_quantization:
            logger.info("Disabling quantization for CPU inference")
        
        model_client = HiggsAudioModelClient(
            model_path=args.model_path,
            audio_tokenizer=audio_tokenizer,
            device_id=device_id,
            max_new_tokens=args.max_new_tokens,
            use_static_kv_cache=args.use_static_kv_cache and device != "cpu",
            use_quantization=use_quantization,
            quantization_bits=args.quantization_bits,
        )
        
        # Prepare generation context
        import re
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(text)))
        
        messages, audio_ids = prepare_generation_context(
            scene_prompt=scene_prompt,
            ref_audio=args.ref_audio,
            ref_audio_in_system_message=False,
            audio_tokenizer=audio_tokenizer,
            speaker_tags=speaker_tags,
        )
        
        # Prepare text chunks
        chunked_text = prepare_chunk_text(
            text,
            chunk_method=args.chunk_method,
            chunk_max_word_num=args.chunk_max_word_num,
            chunk_max_num_turns=1,
        )
        
        if args.verbose:
            logger.info("Text chunks for generation:")
            for idx, chunk in enumerate(chunked_text):
                logger.info(f"Chunk {idx}: {chunk}")
        
        # Generate audio
        logger.info("Generating audio...")
        concat_wv, sr, text_output = model_client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked_text,
            generation_chunk_buffer_size=None,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            ras_win_len=7,
            ras_win_max_num_repeat=2,
            seed=args.seed,
        )
        
        # Save output
        logger.info(f"Saving audio to {args.output}...")
        sf.write(args.output, concat_wv, sr)
        
        # Validate audio content
        import numpy as np
        rms_level = np.sqrt(np.mean(concat_wv**2))
        non_zero_samples = np.count_nonzero(np.abs(concat_wv) > 0.001)
        
        print(f"✅ Audio successfully generated and saved to: {args.output}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(concat_wv) / sr:.2f} seconds")
        print(f"Audio validation: RMS={rms_level:.3f}, Non-silent samples: {non_zero_samples}/{len(concat_wv)} ({100*non_zero_samples/len(concat_wv):.1f}%)")
        
        if rms_level < 0.001:
            print("⚠️  Warning: Audio seems very quiet, it might be just noise")
        elif non_zero_samples < len(concat_wv) * 0.1:
            print("⚠️  Warning: Audio has mostly silence, generation might have failed")
        else:
            print("🎵 Audio validation passed - real speech content detected")
        
        if args.verbose:
            logger.info(f"Model text output: {text_output}")
            
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()