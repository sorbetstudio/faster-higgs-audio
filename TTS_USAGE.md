# Higgs Audio v2 TTS Usage Guide

This guide shows how to use the enhanced TTS script with quantization support and uv.

## Quick Start

### Basic TTS Generation (GPU with quantization - recommended)
```bash
./run_tts.sh "Hello world, this is a test of Higgs Audio TTS." --output hello.wav
```

### CPU Inference (slower but works on any system)
```bash
./run_tts.sh "Hello world, this is CPU generated speech." --device cpu --output cpu_hello.wav
```

## Advanced Usage

### Voice Cloning
```bash
./run_tts.sh "This uses a specific voice clone." --ref_audio belinda --output voice_clone.wav
```

### Custom Temperature and Verbose Output
```bash
./run_tts.sh "This is more creative speech generation." --temperature 0.8 --verbose --output creative.wav
```

### Longer Text with Chunking
```bash
./run_tts.sh "This is a very long piece of text that will be processed in smaller chunks for better memory efficiency and performance." --chunk_method word --chunk_max_word_num 30 --output chunked.wav
```

### 8-bit Quantization (ideal for 8GB GPUs)
```bash
./run_tts.sh "Better quality with 8-bit quantization." --quantization_bits 8 --output 8bit.wav
```

### Disable Quantization (requires more GPU memory)
```bash
./run_tts.sh "This runs without quantization." --no_quantization --output no_quant.wav
```

### Enable Static KV Cache for Speed (requires more memory)
```bash
./run_tts.sh "Faster generation with static cache." --use_static_kv_cache --output faster.wav
```

## Device Options

- `--device cpu`: Force CPU inference (slower, but works without GPU)
- `--device cuda`: Use GPU with CUDA (default if available)
- `--device auto`: Auto-detect best device (default)

## Memory Considerations

- **GPU < 8GB**: Use default settings (4-bit quantization enabled)
- **GPU 8GB**: Use `--quantization_bits 8` for better quality with moderate memory usage
- **GPU >= 12GB**: You can try `--no_quantization` for potentially better quality
- **CPU only**: Use `--device cpu` (will take longer but works)
- **Out of memory**: Try `--device cpu` or ensure no other GPU processes are running

## Audio Output Validation

The generated WAV files should contain actual speech audio. You can check them with:

```bash
# Basic file info
file output.wav

# Check audio properties
python -c "
import soundfile as sf
import numpy as np
audio, sr = sf.read('output.wav')
print(f'Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz')
print(f'Audio range: {audio.min():.3f} to {audio.max():.3f}')
print(f'RMS level: {np.sqrt(np.mean(audio**2)):.3f}')
"
```

## Troubleshooting

1. **"CUDA out of memory"**: Use `--device cpu` or restart your shell to clear CUDA cache
2. **"bitsandbytes not found"**: The script will automatically fall back to non-quantized mode
3. **Silent audio**: Check that your text doesn't contain only special characters
4. **Slow generation**: This is normal for CPU inference; GPU with quantization is much faster

## Performance Comparison

- **GPU + 4-bit Quantization**: ~3-5 seconds for short text (most memory efficient)
- **GPU + 8-bit Quantization**: ~4-6 seconds for short text (better quality, moderate memory)
- **GPU without Quantization**: Requires 12+ GB GPU memory, fastest generation
- **CPU**: ~60+ seconds for short text (but works everywhere)