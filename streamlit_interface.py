#!/usr/bin/env python3
"""
Enhanced Streamlit interface for Higgs Audio v2 with comprehensive parameter control,
quantization options, and persistent model loading for faster generation.
"""

import streamlit as st
import requests
import base64
import io
import os
import time
import json
import gc
import torch
import re
import copy
from pathlib import Path
from openai import OpenAI
import soundfile as sf
import numpy as np
from typing import Optional, Dict, List
from loguru import logger
try:
    from scipy import signal
    from scipy.ndimage import binary_dilation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, using numpy-only audio processing")

# Add project root to path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.data_types import Message, ChatMLSample
    from examples.generation import HiggsAudioModelClient, prepare_generation_context, prepare_chunk_text
    LOCAL_MODE_AVAILABLE = True
except ImportError as e:
    LOCAL_MODE_AVAILABLE = False
    logger.warning(f"Local mode dependencies not available: {e}")
    print(f"⚠️ Local mode disabled due to import error: {e}")

# Page config
st.set_page_config(
    page_title="⚡ Faster Higgs Audio v2",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dark mode state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Multi-speaker system message (from examples/generation.py)
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

def normalize_chinese_punctuation(text):
    """Convert Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        """: '"',  # left double quotation
        """: '"',  # right double quotation
        "'": "'",  # left single quotation
        "'": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }
    
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def normalize_transcript_text(text):
    """Apply all text normalizations like in examples/generation.py"""
    # Basic Chinese punctuation normalization
    text = normalize_chinese_punctuation(text)
    
    # Other normalizations
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°C", " degrees Celsius")
    
    # Sound effect tag replacements
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
    
    # Line cleanup
    lines = text.split("\n")
    text = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    text = text.strip()
    
    # Add final punctuation if missing
    if not any([text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        text += "."
    
    return text

def remove_long_pauses(audio_data: np.ndarray, sample_rate: int, 
                      silence_threshold: float = 0.01, 
                      max_pause_duration: float = 1.0,
                      fade_duration: float = 0.05) -> np.ndarray:
    """
    Remove long pauses from audio while preserving natural speech rhythm.
    
    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        silence_threshold: RMS threshold for silence detection (0.01 = -40dB relative)
        max_pause_duration: Maximum allowed pause duration in seconds
        fade_duration: Duration of fade in/out when cutting pauses
    
    Returns:
        Processed audio with reduced pauses
    """
    if len(audio_data.shape) > 1:
        # Convert stereo to mono if needed
        audio_data = np.mean(audio_data, axis=1)
    
    # Parameters
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)   # 10ms hop
    max_pause_samples = int(max_pause_duration * sample_rate)
    fade_samples = int(fade_duration * sample_rate)
    
    # Calculate RMS energy for each frame
    num_frames = (len(audio_data) - frame_length) // hop_length + 1
    rms_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio_data[start:end]
        rms_values[i] = np.sqrt(np.mean(frame**2))
    
    # Detect silence frames
    silence_frames = rms_values < silence_threshold
    
    # Dilate silence regions to avoid cutting too aggressively
    kernel_size = max(3, int(0.1 * sample_rate / hop_length))  # 100ms dilation
    if SCIPY_AVAILABLE:
        silence_frames = binary_dilation(silence_frames, iterations=kernel_size//2)
    else:
        # Numpy-only dilation using convolution
        kernel = np.ones(kernel_size, dtype=bool)
        silence_frames = np.convolve(silence_frames.astype(int), kernel, mode='same') > 0
    
    # Find continuous silence regions
    silence_regions = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            silence_end = i
            silence_duration = (silence_end - silence_start) * hop_length
            if silence_duration > max_pause_samples:
                # Convert frame indices to sample indices
                start_sample = silence_start * hop_length
                end_sample = silence_end * hop_length
                # Keep only max_pause_duration worth of silence
                keep_samples = max_pause_samples
                remove_start = start_sample + keep_samples // 2
                remove_end = end_sample - keep_samples // 2
                silence_regions.append((remove_start, remove_end))
            in_silence = False
    
    # Handle silence at the end
    if in_silence:
        silence_end = len(silence_frames)
        silence_duration = (silence_end - silence_start) * hop_length
        if silence_duration > max_pause_samples:
            start_sample = silence_start * hop_length
            end_sample = min(len(audio_data), silence_end * hop_length)
            keep_samples = max_pause_samples
            remove_start = start_sample + keep_samples // 2
            remove_end = end_sample - keep_samples // 2
            silence_regions.append((remove_start, remove_end))
    
    # Remove long pauses (work backwards to maintain indices)
    processed_audio = audio_data.copy()
    for remove_start, remove_end in reversed(silence_regions):
        if remove_end > remove_start and remove_start >= 0 and remove_end <= len(processed_audio):
            # Apply fade out before cut
            fade_start = max(0, remove_start - fade_samples)
            if fade_start < remove_start:
                fade_length = remove_start - fade_start
                fade_out = np.linspace(1, 0, fade_length)
                processed_audio[fade_start:remove_start] *= fade_out
            
            # Apply fade in after cut
            fade_end = min(len(processed_audio), remove_end + fade_samples)
            if remove_end < fade_end:
                fade_length = fade_end - remove_end
                fade_in = np.linspace(0, 1, fade_length)
                processed_audio[remove_end:fade_end] *= fade_in
            
            # Remove the silence region
            processed_audio = np.concatenate([
                processed_audio[:remove_start],
                processed_audio[remove_end:]
            ])
    
    return processed_audio

def apply_audio_postprocessing(audio_bytes: bytes, 
                             silence_threshold: float = 0.01,
                             max_pause_duration: float = 1.0,
                             fade_duration: float = 0.05) -> bytes:
    """
    Apply post-processing to remove long pauses and clean up audio.
    
    Args:
        audio_bytes: Input audio as bytes
        silence_threshold: RMS threshold for silence detection
        max_pause_duration: Maximum allowed pause in seconds
        fade_duration: Fade duration for smooth cuts
    
    Returns:
        Processed audio as bytes
    """
    try:
        # Load audio
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer)
        
        # Apply pause removal
        processed_audio = remove_long_pauses(
            audio_data, sample_rate, 
            silence_threshold, max_pause_duration, fade_duration
        )
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio * (0.95 / max_val)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        sf.write(output_buffer, processed_audio, sample_rate, format='WAV')
        return output_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Audio post-processing failed: {e}")
        return audio_bytes  # Return original if processing fails

# Dynamic CSS based on dark mode state
def get_css_theme():
    if st.session_state.dark_mode:
        return """
        <style>
            .main-header {
                text-align: center;
                padding: 1rem 0;
                background: linear-gradient(90deg, #4a5568 0%, #2d3748 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: #2d3748;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                color: white;
            }
            .parameter-section {
                background: #2d3748;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #4a5568;
                margin: 1rem 0;
            }
            .stButton > button {
                width: 100%;
                height: 3rem;
                font-size: 1.1rem;
                font-weight: bold;
            }
            .generation-status {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                background: #2d3748;
            }
            /* Dark mode specific styling */
            .stApp {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .stSidebar {
                background-color: #2d3748;
            }
        </style>
        """
    else:
        return """
        <style>
            .main-header {
                text-align: center;
                padding: 1rem 0;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: #f0f2f6;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            .parameter-section {
                background: #ffffff;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                margin: 1rem 0;
            }
            .stButton > button {
                width: 100%;
                height: 3rem;
                font-size: 1.1rem;
                font-weight: bold;
            }
            .generation-status {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
        """

# Apply CSS theme
st.markdown(get_css_theme(), unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class="main-header">
    <h1>⚡ Faster Higgs Audio v2 - Enhanced Interface</h1>
    <p>Generate expressive speech with quantization, voice cloning, and comprehensive parameter control</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_client' not in st.session_state:
    st.session_state.model_client = None
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'api_client' not in st.session_state:
    st.session_state.api_client = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Dark mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("🌙 Dark Mode")
    with col2:
        if st.button("🔄", help="Toggle dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.markdown("---")
    
    # Mode selection
    if LOCAL_MODE_AVAILABLE:
        mode = st.selectbox(
            "Inference Mode",
            ["Local Model", "API Server"],
            help="Local mode loads model directly, API mode uses remote server"
        )
    else:
        mode = "API Server"
        st.info("Local mode not available - using API server mode")
    
    if mode == "API Server":
        # API Configuration
        st.subheader("📡 API Settings")
        api_base = st.text_input("API Base URL", value="http://localhost:8000/v1")
        model_name = st.text_input("Model Name", value="higgs-audio-v2-generation-3B-base")
        
        # Test API connection
        if st.button("🔍 Test API Connection"):
            try:
                client = OpenAI(api_key="EMPTY", base_url=api_base)
                models = client.models.list()
                st.success(f"✅ Connected! Available models: {[m.id for m in models.data]}")
                st.session_state.api_client = client
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                st.session_state.api_client = None
    
    else:
        # Local Model Configuration
        st.subheader("🔧 Local Model Settings")
        
        model_path = st.text_input(
            "Model Path",
            value="bosonai/higgs-audio-v2-generation-3B-base",
            help="HuggingFace model path or local directory"
        )
        
        tokenizer_path = st.text_input(
            "Audio Tokenizer Path",
            value="bosonai/higgs-audio-v2-tokenizer",
            help="HuggingFace tokenizer path or local directory"
        )
        
        # Hardware Configuration
        st.subheader("💻 Hardware Settings")
        
        device = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda:0", "cuda:1"],
            help="Device for model inference"
        )
        
        use_quantization = st.checkbox(
            "Enable Quantization",
            value=True,
            help="Use quantization to reduce memory usage"
        )
        
        if use_quantization:
            quantization_bits = st.selectbox(
                "Quantization Precision",
                [4, 8],
                index=0,
                help="4-bit: most memory efficient, 8-bit: better quality"
            )
        else:
            quantization_bits = None
        
        use_static_kv_cache = st.checkbox(
            "Static KV Cache",
            value=False,
            help="Faster generation but uses more memory. Disable for multi-speaker if getting OOM."
        )
        
        if use_static_kv_cache:
            st.warning("⚠️ Static KV Cache uses significant memory. Disable if you get CUDA OOM errors.")
        
        max_new_tokens = st.number_input(
            "Max New Tokens",
            min_value=100,
            max_value=4096,
            value=2048,
            step=100,
            help="Maximum tokens to generate"
        )
        
        # Model Loading/Unloading
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    try:
                        # Clear any existing model
                        if st.session_state.model_client is not None:
                            del st.session_state.model_client
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Determine device
                        if device == "auto":
                            actual_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                        else:
                            actual_device = device
                        
                        # Load model client
                        st.session_state.model_client = HiggsAudioModelClient(
                            model_path=model_path,
                            audio_tokenizer=load_higgs_audio_tokenizer(tokenizer_path),
                            device_id=actual_device.split(":")[-1] if "cuda" in actual_device else None,
                            max_new_tokens=max_new_tokens,
                            use_static_kv_cache=use_static_kv_cache,
                            use_quantization=use_quantization,
                            quantization_bits=quantization_bits or 4,
                        )
                        
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded successfully!")
                        
                        # Display model info
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3
                            st.info(f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {str(e)}")
                        st.session_state.model_loaded = False
        
        with col2:
            if st.button("🗑️ Unload Model"):
                if st.session_state.model_client is not None:
                    del st.session_state.model_client
                    st.session_state.model_client = None
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    st.session_state.model_loaded = False
                    st.success("✅ Model unloaded")
    
    # Model Status
    if mode == "Local Model":
        if st.session_state.model_loaded:
            st.success("🟢 Model Loaded")
        else:
            st.warning("🟡 Model Not Loaded")
    elif st.session_state.api_client:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 Not Connected")
    
    # System Info
    st.subheader("📊 System Info")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.info(f"GPU: {gpu_name}\nVRAM: {total_memory:.1f}GB")
    else:
        st.info("CPU Only Mode")

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Smart Voice", "👥 Voice Clone", "🎭 Multi-Speaker", "📊 History"])

# Voice presets
VOICE_PRESETS = {
    "belinda": "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year.",
    "broom_salesman": "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic.",
    "en_woman": "Hello, I'm a female English speaker with a clear and pleasant voice.",
    "en_man": "Hello, I'm a male English speaker with a deep and resonant voice.",
    "alice": "Hi there! I'm Alice, your friendly AI assistant. How can I help you today?",
    "bob": "Hey, this is Bob. I have a warm, conversational speaking style."
}

# Scene prompts
SCENE_PROMPTS = {
    "quiet_indoor": "Audio is recorded from a quiet room.",
    "outdoor": "Audio recorded outdoors with natural ambient sounds.",
    "studio": "Professional studio recording with pristine audio quality.",
    "phone_call": "Audio recorded during a phone call with slight compression.",
    "conference": "Audio from a conference room with slight echo.",
    "custom": "Enter your own scene description..."
}

def create_parameter_section(tab_name: str):
    """Create a collapsible parameter section"""
    with st.expander(f"🎛️ {tab_name} Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Generation")
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                                   help="Higher = more creative/random", key=f"temp_{tab_name}")
            top_k = st.number_input("Top-K", 1, 100, 50, 
                                  help="Limits vocabulary for sampling", key=f"topk_{tab_name}")
            top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05, 
                            help="Nucleus sampling threshold", key=f"topp_{tab_name}")
        
        with col2:
            st.subheader("🎵 Audio")
            scene_type = st.selectbox("Scene Type", list(SCENE_PROMPTS.keys()), key=f"scene_{tab_name}")
            if scene_type == "custom":
                scene_prompt = st.text_area("Custom Scene", 
                                          placeholder="Describe the recording environment...", key=f"custom_scene_{tab_name}")
            else:
                scene_prompt = SCENE_PROMPTS[scene_type]
                st.text_area("Scene Description", scene_prompt, disabled=True, height=60, key=f"scene_desc_{tab_name}")
            
            seed = st.number_input("Random Seed", 0, 999999, 42, 
                                 help="For reproducible generation", key=f"seed_{tab_name}")
        
        with col3:
            st.subheader("📝 Text Processing")
            chunk_method = st.selectbox("Chunking", [None, "word", "speaker"], 
                                      help="How to split long text", key=f"chunk_{tab_name}")
            if chunk_method == "word":
                chunk_max_words = st.number_input("Max Words/Chunk", 50, 500, 200, key=f"chunk_words_{tab_name}")
            else:
                chunk_max_words = 200
            
            # Memory Management
            st.subheader("🧠 Memory Management")
            chunk_buffer_size = st.number_input("Chunk Buffer Size", 0, 10, 0, 
                                              help="Limit chunks in memory (0=auto, good for long multi-speaker)", key=f"chunk_buffer_{tab_name}")
            if chunk_buffer_size == 0:
                chunk_buffer_size = None
            
            # Memory usage info
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    st.info(f"📊 GPU: {allocated:.1f}GB / {gpu_memory_gb:.1f}GB used")
                except:
                    pass
            
            # RAS sampling parameters (from examples)
            st.subheader("🎯 Advanced Sampling")
            ras_win_len = st.number_input("RAS Window Length", 0, 20, 7, 
                                        help="Window length for RAS sampling (0 to disable)", key=f"ras_win_{tab_name}")
            if ras_win_len > 0:
                ras_win_max_repeat = st.number_input("RAS Max Repeats", 1, 5, 2, 
                                                   help="Max times to repeat RAS window", key=f"ras_repeat_{tab_name}")
            else:
                ras_win_max_repeat = 2
        
        return {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "scene_prompt": scene_prompt,
            "seed": seed,
            "chunk_method": chunk_method,
            "chunk_max_words": chunk_max_words,
            "chunk_buffer_size": chunk_buffer_size,
            "ras_win_len": ras_win_len if ras_win_len > 0 else None,
            "ras_win_max_repeat": ras_win_max_repeat
        }

def generate_audio_local(text_input: str, params: Dict, ref_audio: str = None):
    """Generate audio using local model"""
    if not st.session_state.model_loaded or st.session_state.model_client is None:
        st.error("❌ Model not loaded. Please load the model first.")
        return None, None, None
    
    try:
        with st.spinner("🎵 Generating audio locally..."):
            start_time = time.time()
            
            # Apply text normalization like in examples
            normalized_text = normalize_transcript_text(text_input)
            
            # Extract speaker tags like in examples/generation.py
            pattern = re.compile(r"\[(SPEAKER\d+)\]")
            speaker_tags = sorted(set(pattern.findall(normalized_text)))
            
            # Prepare generation context with proper speaker tags
            messages, audio_ids = prepare_generation_context(
                scene_prompt=params.get("scene_prompt", "Audio is recorded from a quiet room."),
                ref_audio=ref_audio,
                ref_audio_in_system_message=False,
                audio_tokenizer=st.session_state.model_client._audio_tokenizer,
                speaker_tags=speaker_tags  # Now passing the extracted speaker tags!
            )
            
            # Prepare chunked text
            chunked_text = prepare_chunk_text(
                normalized_text,  # Use normalized text
                chunk_method=params.get("chunk_method"),
                chunk_max_word_num=params.get("chunk_max_words", 200),
                chunk_max_num_turns=1
            )
            
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Memory optimization: Use chunk buffer size for multi-speaker
            chunk_buffer_size = params.get("chunk_buffer_size")
            if chunk_buffer_size is None:
                # Auto-determine based on number of chunks and available memory
                num_chunks = len(chunked_text)
                if num_chunks > 4:  # For long multi-speaker content
                    chunk_buffer_size = min(3, num_chunks // 2)
                    logger.info(f"Using chunk buffer size: {chunk_buffer_size} for {num_chunks} chunks")
            
            # Generate audio with RAS parameters and memory optimization
            concat_wv, sr, text_output = st.session_state.model_client.generate(
                messages=messages,
                audio_ids=audio_ids,
                chunked_text=chunked_text,
                generation_chunk_buffer_size=chunk_buffer_size,
                temperature=params.get("temperature", 0.7),
                top_k=params.get("top_k", 50),
                top_p=params.get("top_p", 0.95),
                ras_win_len=params.get("ras_win_len", 7),
                ras_win_max_num_repeat=params.get("ras_win_max_repeat", 2),
                seed=params.get("seed", 42) if params.get("seed", 42) > 0 else None,
            )
            
            generation_time = time.time() - start_time
            
            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Save to memory buffer
            buffer = io.BytesIO()
            sf.write(buffer, concat_wv, sr, format='WAV')
            audio_bytes = buffer.getvalue()
            
            return audio_bytes, text_output, generation_time
            
    except Exception as e:
        error_msg = str(e)
        if "CUDA out of memory" in error_msg:
            st.error(f"❌ CUDA out of memory. Try reducing chunk size or enabling chunk buffer limit.")
            st.info("💡 **Memory Optimization Tips:**\n"
                   "- Set 'Chunk Buffer Size' to 2-3 for long multi-speaker content\n"
                   "- Use 'speaker' chunk method for multi-speaker dialogues\n"
                   "- Reduce 'Max New Tokens' to 1024 or lower\n"
                   "- Disable 'Static KV Cache' checkbox\n"
                   "- Try 4-bit quantization if using 8-bit\n"
                   "- Reduce chunk max words to 100-150")
            
            # Show current memory usage if available
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    cached = torch.cuda.memory_reserved(0) / (1024**3)
                    st.error(f"📊 Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
                except:
                    pass
        else:
            st.error(f"❌ Local generation failed: {error_msg}")
        
        # Clear memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return None, None, None

def generate_audio_api(messages: List[Dict], params: Dict):
    """Generate audio using API"""
    if not st.session_state.api_client:
        st.error("❌ API not connected. Please test API connection first.")
        return None, None, None
    
    try:
        with st.spinner("🎵 Generating audio via API..."):
            start_time = time.time()
            
            response = st.session_state.api_client.chat.completions.create(
                messages=messages,
                model=model_name,
                modalities=["text", "audio"],
                audio={"format": "wav"},
                temperature=params.get("temperature", 0.7),
                max_completion_tokens=params.get("max_tokens", 500),
                top_p=params.get("top_p", 0.95),
                extra_body={"top_k": params.get("top_k", 50)},
                stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"]
            )
            
            generation_time = time.time() - start_time
            
            text_output = response.choices[0].message.content
            audio_data = response.choices[0].message.audio.data
            audio_bytes = base64.b64decode(audio_data)
            
            return audio_bytes, text_output, generation_time
            
    except Exception as e:
        st.error(f"❌ API generation failed: {str(e)}")
        return None, None, None

def validate_audio(audio_bytes: bytes) -> Dict:
    """Validate generated audio and return metrics"""
    try:
        # Load audio for analysis
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer)
        
        # Calculate metrics
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data**2))
        non_silent_samples = np.sum(np.abs(audio_data) > 0.01)
        non_silent_percentage = (non_silent_samples / len(audio_data)) * 100
        
        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "rms": rms,
            "non_silent_samples": non_silent_samples,
            "total_samples": len(audio_data),
            "non_silent_percentage": non_silent_percentage,
            "is_valid": rms > 0.01 and non_silent_percentage > 20
        }
    except Exception as e:
        return {"error": str(e), "is_valid": False}

def display_generation_result(audio_bytes, text_output, gen_time, params, input_text):
    """Display generation results with validation"""
    if audio_bytes:
        # Validate audio
        validation = validate_audio(audio_bytes)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Generation Time", f"{gen_time:.2f}s")
        with col2:
            st.metric("🎵 Duration", f"{validation.get('duration', 0):.2f}s")
        with col3:
            st.metric("📊 RMS Level", f"{validation.get('rms', 0):.3f}")
        
        # Audio validation status
        if validation.get("is_valid", False):
            st.success(f"✅ Audio validation passed - Real speech content detected "
                      f"({validation.get('non_silent_percentage', 0):.1f}% non-silent)")
        else:
            st.warning("⚠️ Audio validation warning - Check output quality")
        
        # Model text output
        if text_output and text_output.strip():
            st.text_area("🤖 Model Output", value=text_output, height=80)
        
        # Audio player
        st.audio(audio_bytes, format="audio/wav")
        
        # Post-processing section
        st.subheader("🎧 Audio Post-Processing")
        col_pp1, col_pp2, col_pp3 = st.columns(3)
        
        with col_pp1:
            silence_thresh = st.slider("Silence Threshold", 0.001, 0.1, 0.01, 
                                     help="Lower = more aggressive pause removal", format="%.3f")
        with col_pp2:
            max_pause = st.slider("Max Pause Duration", 0.2, 3.0, 1.0, step=0.1, 
                                help="Maximum allowed pause in seconds")
        with col_pp3:
            fade_duration = st.slider("Fade Duration", 0.01, 0.2, 0.05, step=0.01,
                                    help="Fade duration for smooth cuts")
        
        # Post-processing button
        if st.button("🎧 Remove Long Pauses", help="Process audio to remove long pauses and empty sounds"):
            with st.spinner("🎧 Processing audio..."):
                try:
                    start_time = time.time()
                    processed_audio = apply_audio_postprocessing(
                        audio_bytes, 
                        silence_threshold=silence_thresh,
                        max_pause_duration=max_pause,
                        fade_duration=fade_duration
                    )
                    processing_time = time.time() - start_time
                    st.success(f"✅ Audio processed in {processing_time:.2f}s")
                    
                    # Validate processed audio
                    processed_validation = validate_audio(processed_audio)
                    original_duration = validation.get('duration', 0)
                    processed_duration = processed_validation.get('duration', 0)
                    time_saved = original_duration - processed_duration
                    
                    # Show comparison
                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    with col_comp1:
                        st.metric("⏱️ Original Duration", f"{original_duration:.2f}s")
                    with col_comp2:
                        st.metric("⏰ Processed Duration", f"{processed_duration:.2f}s")
                    with col_comp3:
                        st.metric("💾 Time Saved", f"{time_saved:.2f}s")
                    
                    # Play processed audio
                    st.subheader("🎧 Processed Audio")
                    st.audio(processed_audio, format="audio/wav")
                    
                    # Download processed audio
                    processed_filename = f"processed_higgs_audio_{int(time.time())}.wav"
                    st.download_button(
                        label="📥 Download Processed Audio",
                        data=processed_audio,
                        file_name=processed_filename,
                        mime="audio/wav",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Post-processing failed: {str(e)}")
        
        # Original download button
        filename = f"higgs_audio_{int(time.time())}.wav"
        st.download_button(
            label="📥 Download Original Audio",
            data=audio_bytes,
            file_name=filename,
            mime="audio/wav"
        )
        
        # Add to history
        st.session_state.generation_history.append({
            "timestamp": time.time(),
            "input_text": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "duration": validation.get('duration', 0),
            "generation_time": gen_time,
            "parameters": params,
            "rms": validation.get('rms', 0),
            "is_valid": validation.get("is_valid", False)
        })

# Tab 1: Smart Voice Generation
with tab1:
    st.header("🎙️ Smart Voice Generation")
    st.markdown("The model automatically selects an appropriate voice based on your text.")
    
    params = create_parameter_section("Smart Voice")
    
    text_input = st.text_area(
        "Enter text to generate speech:",
        value="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
        height=120,
        help="Enter any text you want to convert to speech"
    )
    
    if st.button("🎵 Generate Smart Voice", key="smart_voice", type="primary"):
        if text_input.strip():
            # Generate audio
            if mode == "Local Model":
                audio_bytes, text_output, gen_time = generate_audio_local(text_input, params)
            else:
                # Prepare system prompt for API
                system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{params['scene_prompt']}\n<|scene_desc_end|>"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input}
                ]
                audio_bytes, text_output, gen_time = generate_audio_api(messages, params)
            
            # Display results
            if audio_bytes:
                display_generation_result(audio_bytes, text_output, gen_time, params, text_input)
        else:
            st.warning("Please enter some text to generate speech.")

# Tab 2: Voice Cloning
with tab2:
    st.header("👥 Voice Cloning")
    st.markdown("Clone a voice by providing reference audio and transcript.")
    
    params = create_parameter_section("Voice Clone")
    
    # Voice selection
    col1, col2 = st.columns([1, 2])
    with col1:
        use_preset = st.checkbox("Use preset voice", value=True)
    
    if use_preset:
        with col2:
            selected_voice = st.selectbox("Select preset voice:", list(VOICE_PRESETS.keys()))
        
        ref_transcript = st.text_area(
            "Reference transcript:",
            value=VOICE_PRESETS[selected_voice],
            height=80,
            disabled=True
        )
        
        # Check if preset audio exists
        ref_audio_path = Path(f"examples/voice_prompts/{selected_voice}.wav")
        if ref_audio_path.exists():
            st.success(f"✅ Using preset audio: {selected_voice}.wav")
            # Display reference audio
            with open(ref_audio_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        else:
            st.error(f"❌ Preset audio file not found: {ref_audio_path}")
    else:
        uploaded_file = st.file_uploader(
            "Upload reference audio",
            type=['wav', 'mp3', 'flac'],
            help="Upload an audio file to clone its voice"
        )
        ref_transcript = st.text_area(
            "Reference transcript:",
            placeholder="Enter the exact text spoken in the reference audio...",
            height=80
        )
        
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
    
    # Target text
    target_text = st.text_area(
        "Enter text to generate with cloned voice:",
        value="Hey there! I'm your friendly voice twin in the making. Let's clone some vocals and bring your voice to life!",
        height=120
    )
    
    if st.button("🎭 Clone Voice", key="voice_clone", type="primary"):
        if target_text.strip() and (use_preset or (uploaded_file and ref_transcript.strip())):
            try:
                # Prepare audio data
                if use_preset:
                    with open(ref_audio_path, "rb") as f:
                        audio_data = f.read()
                else:
                    audio_data = uploaded_file.read()
                
                if mode == "API Server":
                    # API mode - use base64 encoding
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    
                    messages = [
                        {"role": "user", "content": ref_transcript},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_base64,
                                        "format": "wav",
                                    },
                                }
                            ],
                        },
                        {"role": "user", "content": target_text}
                    ]
                    
                    audio_bytes, text_output, gen_time = generate_audio_api(messages, params)
                else:
                    # Local mode - save temp audio file
                    temp_audio_path = Path("temp_ref_audio.wav")
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_data)
                    
                    # Use local generation with reference audio
                    ref_audio_name = selected_voice if use_preset else None
                    audio_bytes, text_output, gen_time = generate_audio_local(target_text, params, ref_audio_name)
                    
                    # Clean up temp file
                    temp_audio_path.unlink(missing_ok=True)
                
                # Display results
                if audio_bytes:
                    st.success("🎭 Voice cloning completed!")
                    display_generation_result(audio_bytes, text_output, gen_time, params, target_text)
                    
            except Exception as e:
                st.error(f"❌ Voice cloning failed: {str(e)}")
        else:
            st.warning("Please provide target text and reference audio with transcript.")

# Tab 3: Multi-Speaker Generation
with tab3:
    st.header("🎭 Multi-Speaker Generation")
    st.markdown("""
    Generate conversations with multiple speakers using [SPEAKER0], [SPEAKER1] tags.
    
    **🎵 Sound Effects Support:** Use tags like [laugh], [music], [applause], [cough] for realistic audio effects.
    **🌡️ Temperature Support:** °F and °C are automatically converted to spoken form.
    **🌍 Multi-language:** Chinese punctuation is automatically normalized.
    """)
    
    params = create_parameter_section("Multi-Speaker")
    
    # Multi-speaker configuration
    col1, col2 = st.columns(2)
    with col1:
        num_speakers = st.number_input("Number of Speakers", 2, 4, 2)
    
    with col2:
        assign_voices = st.checkbox("Assign specific voices", value=False)
    
    if assign_voices:
        st.subheader("🎤 Voice Assignment")
        voice_assignments = {}
        for i in range(num_speakers):
            voice_assignments[f"SPEAKER{i}"] = st.selectbox(
                f"Voice for SPEAKER{i}:",
                ["auto"] + list(VOICE_PRESETS.keys()),
                key=f"voice_speaker_{i}"
            )
    
    # Sample transcript with enhanced features
    sample_transcript = """[SPEAKER0] I can't believe you forgot our anniversary again! [cough]

[SPEAKER1] I'm sorry, honey. I've been so swamped with work lately. The office was 85°F today and the air conditioning broke.

[SPEAKER0] That's what you said last year! And the year before that! [laugh] At least I find it amusing now.

[SPEAKER1] You're right, I have no excuse. How can I make it up to you?

[SPEAKER0] Well, you could start by taking me to that new restaurant downtown. The one with live [music] every Friday night.

[SPEAKER1] Consider it done. I'll make a reservation right now. [applause] You deserve the best!"""
    
    transcript_input = st.text_area(
        "Enter multi-speaker dialogue (use [SPEAKER0], [SPEAKER1] tags):",
        value=sample_transcript,
        height=250,
        help="Use [SPEAKER0], [SPEAKER1], etc. to indicate different speakers"
    )
    
    if st.button("🎭 Generate Multi-Speaker", key="multi_speaker", type="primary"):
        if transcript_input.strip():
            # Create system prompt for multi-speaker
            speaker_descriptions = []
            if assign_voices:
                for speaker, voice in voice_assignments.items():
                    if voice != "auto":
                        speaker_descriptions.append(f"{speaker}: {voice} voice style")
                    else:
                        speaker_descriptions.append(f"{speaker}: automatically selected voice")
            else:
                for i in range(num_speakers):
                    speaker_descriptions.append(f"SPEAKER{i}: automatically selected voice")
            
            system_prompt = f"""You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own.

<|scene_desc_start|>
{params['scene_prompt']}
{chr(10).join(speaker_descriptions)}
<|scene_desc_end|>"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_input}
            ]
            
            # Generate audio
            if mode == "Local Model":
                audio_bytes, text_output, gen_time = generate_audio_local(transcript_input, params)
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript_input}
                ]
                audio_bytes, text_output, gen_time = generate_audio_api(messages, params)
            
            # Display results
            if audio_bytes:
                st.success("🎭 Multi-speaker dialogue generated!")
                display_generation_result(audio_bytes, text_output, gen_time, params, transcript_input)
        else:
            st.warning("Please enter a multi-speaker dialogue.")

# Tab 4: Generation History
with tab4:
    st.header("📊 Generation History")
    
    if st.session_state.generation_history:
        # Statistics
        total_generations = len(st.session_state.generation_history)
        avg_duration = np.mean([h["duration"] for h in st.session_state.generation_history])
        avg_gen_time = np.mean([h["generation_time"] for h in st.session_state.generation_history])
        success_rate = np.mean([h["is_valid"] for h in st.session_state.generation_history]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Generations", total_generations)
        with col2:
            st.metric("Avg Audio Duration", f"{avg_duration:.2f}s")
        with col3:
            st.metric("Avg Generation Time", f"{avg_gen_time:.2f}s")
        with col4:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # History table
        st.subheader("📋 Recent Generations")
        
        for i, entry in enumerate(reversed(st.session_state.generation_history[-10:])):
            with st.expander(f"#{total_generations - i}: {entry['input_text']} ({time.ctime(entry['timestamp'])})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Duration:** {entry['duration']:.2f}s")
                    st.write(f"**Gen Time:** {entry['generation_time']:.2f}s")
                with col2:
                    st.write(f"**RMS:** {entry['rms']:.3f}")
                    st.write(f"**Valid:** {'✅' if entry['is_valid'] else '❌'}")
                with col3:
                    st.write(f"**Temperature:** {entry['parameters'].get('temperature', 'N/A')}")
                    st.write(f"**Top-K:** {entry['parameters'].get('top_k', 'N/A')}")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.generation_history = []
            st.rerun()
    
    else:
        st.info("📝 No generations yet. Generate some audio to see history here!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🎵 **Faster Higgs Audio v2**")
with col2:
    st.markdown("⚡ Enhanced Interface with Quantization")
with col3:
    if torch.cuda.is_available():
        st.markdown(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.markdown("💻 CPU Mode")