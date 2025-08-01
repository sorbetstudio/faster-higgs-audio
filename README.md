<h1 align="center">Faster Higgs Audio (Quantized)</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">

  <a href="#quantized-inference"><img src="https://img.shields.io/badge/4bit/8bit Quantization-32CD32" style="margin-right: 5px;"></a>
  <a href="#server-mode"><img src="https://img.shields.io/badge/Local Server + Web UI-4169E1" style="margin-right: 5px;"></a>
</div>

<p align="center">
    <img src="https://img.shields.io/badge/GPU-8GB_VRAM_(recommended)-green?style=for-the-badge" alt="8GB VRAM">
    <img src="https://img.shields.io/badge/CPU-Compatible-blue?style=for-the-badge" alt="CPU Compatible">
    <img src="https://img.shields.io/badge/Mac-Compatible-yellow?style=for-the-badge" alt="Mac Compatible">
</p>

This fork optimizes the Higgs Audio v2 model for local deployment with quantization, making it easy to run it on a regular GPU. And it adds an OpenAI-compatible API server for local deployment and a streamlit web ui to tinker with it without the need for any API server.

The goal was to build upon the original repo's examples and expand it's accessiblity.

**What this repo does:**
- **4-bit & 8-bit quantization** - 4-bit is really experimental, but it works!
- **CPU fallback** - Vanilla ram-based deployment
- **Local server mode** - OpenAI-compatible API + web interface
- **30-second setup** - One command to start generating
- **Smart device detection** - Automatically optimizes for your hardware

**Hardware Requirements:**
- **Ideal:** GPU with 8GB+ VRAM (8-bit quantization)
- **Minimum :** GPU with 4-6GB VRAM (4-bit quantization)
- **Fallback:** CPU-only (slower but universal)
- **Mac:** M1/M2/M3 (8-bit quantization) (slightly slower but works great, thanks to the parent repo's MPS implementation)
- **AMD:** Not tested yet. (help needed)

## Setup

Get it running :

```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# Create venv
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt -e . bitsandbytes

# Test
./run_tts.sh "Hello world, this is Higgs Audio!"
```

## Quantized Inference

**Scripts to run without setting up a server or webui:**

```bash
./run_tts.sh "Hey there, if you can hear me, it's working!"

# 8-bit quantization
./run_tts.sh "8-bit quants are up and running" --quantization_bits 8

# 4-bit quantization 
./run_tts.sh "4-bit quants are up and running" --quantization_bits 4

# CPU-only mode
./run_tts.sh "So, this is the CPU talking, right now." --device cpu

# Voice cloning with quantization
./run_tts.sh "Hey there, I'm belinda, nice to meet ya!" --ref_audio belinda --quantization_bits 8
```

### Performance & Memory

| Hardware      | Mode            | Memory    | Speed     | Quality      |
| ------------- | --------------- | --------- | --------- | ------------ |
| **12GB+ GPU** | No quantization | idk haha  | ?         | Perfect      |
| **8GB GPU**   | 8-bit quantized | ~7GB      | Fast      | Near Perfect |
| **6GB GPU**   | 4-bit quantized | ~5GB      | Very Fast | Fair         |
| **CPU Only**  | Any mode        | ~21GB RAM | Slow      | Perfect      |

**Real Performance (on 3070ti 8GB):**
- **8-bit quantized:** About 0.5x realtime inference
- **4-bit quantized:** Near Realtime Inference 
- **CPU mode:** ~60+ seconds per sentence

## Server Mode

**Deploy with OpenAI-compatible API + Web Interface:**

```bash
# Start local server with web UI (one command)
./run_server.sh

# Access at:
# Web Interface: http://localhost:8501
# API Server: http://localhost:8000
```

**Features:**
- **Streamlit Web UI** - Easy point-and-click interface
- **OpenAI API Compatible** - Drop-in replacement for OpenAI TTS
- **Voice Cloning Interface** - Upload audio, get cloned voice
- **Multi-Speaker Generation** - Generate conversations
- **Auto-quantization** - Optimizes for your hardware
- **Network Access** - Use from other devices on your network

---

## Docker Option (TODO)

(Not yet tested)

For isolated environments, use NVIDIA containers:

```bash
# Run in NVIDIA container (optional)
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash

# Then install normally:
git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio
uv pip install -r requirements.txt -e . bitsandbytes
./run_tts.sh "Docker this side, awake and aware!"
```

## Alternative Install Methods

<details>
<summary>Click to expand alternative installation options</summary>

### Using conda
```bash
git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio
conda create -n faster_higgs python=3.10 && conda activate faster_higgs
pip install -r requirements.txt -e . bitsandbytes
```

### Using venv
```bash
git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e . bitsandbytes
```

### Direct pip install
```bash
git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio
pip install -r requirements.txt -e . bitsandbytes
```

### Using conda with local environment
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

conda create -y --prefix ./conda_env --override-channels --strict-channel-priority --channel "conda-forge" "python==3.10.*"
conda activate ./conda_env
pip install -r requirements.txt
pip install -e .

# Uninstalling environment:
conda deactivate
conda remove -y --prefix ./conda_env --all
```

</details>

## Usage

> [!TIP]
> **Use `./run_tts.sh` for instant speech generation!** Automatically detects your hardware and applies optimal quantization. Works on 6GB GPUs or CPU-only.

> [!NOTE] 
> **For server deployment:** Use `./run_server.sh` to launch the web interface with OpenAI-compatible API.


### TTS Script Usage

The `./run_tts.sh` script provides an easy-to-use interface with automatic quantization and device detection:

#### Basic Commands

```bash
# Simple speech generation (auto-detects best device)
./run_tts.sh "Hello world, this is Higgs Audio!"

# Specify output file
./run_tts.sh "Your text here" --output my_speech.wav

# Voice cloning with reference audio
./run_tts.sh "Clone this voice style" --ref_audio belinda

# More creative/expressive generation
./run_tts.sh "Expressive speech with variation" --temperature 0.8

# CPU inference (slower but works everywhere)
./run_tts.sh "CPU generated speech" --device cpu

# Longer text with chunking for better memory usage
./run_tts.sh "This is a very long piece of text that will be processed efficiently using automatic chunking for optimal memory usage and performance." --chunk_method word --chunk_max_word_num 30
```

#### Advanced Options

```bash
# 8-bit quantization (better quality for 8GB+ GPUs)
./run_tts.sh "High quality 8-bit mode" --quantization_bits 8

# Disable quantization (requires more GPU memory)
./run_tts.sh "Without quantization" --no_quantization

# Enable static KV cache for speed (uses more memory)
./run_tts.sh "Faster generation" --use_static_kv_cache

# Verbose output with generation details
./run_tts.sh "Debug information" --verbose

# Custom scene/environment
./run_tts.sh "Outdoor speech" --scene_prompt "Audio recorded outdoors with ambient noise"

# Multiple speakers (use SPEAKER tags in text)
./run_tts.sh "[SPEAKER0] Hello there! [SPEAKER1] Hi, how are you?" --ref_audio alice,bob
```

#### Audio Validation

The script automatically validates generated audio:
- ✅ **RMS Level Check**: Ensures audio isn't just noise
- ✅ **Content Detection**: Verifies real speech content
- ✅ **Quality Metrics**: Reports duration, sample rate, and audio statistics

Example output:
```
✅ Audio successfully generated and saved to: speech.wav
Sample rate: 24000 Hz
Duration: 3.24 seconds
Audio validation: RMS=0.127, Non-silent samples: 71645/77760 (92.1%)
Audio validation passed - real speech content detected
```

### Get Started (Python API)

Here's a basic python snippet to help you get started.

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\\n\\n<|scene_desc_start|>\\nAudio is recorded from a quiet room.\\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

We also provide a list of examples under [examples](./examples). In the following we highlight a few examples to help you use Higgs Audio v2.

### Zero-Shot Voice Cloning
Generate audio that sounds similar as the provided [reference audio](./examples/voice_prompts/belinda.wav).

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation.wav
```

The generation script will automatically use `cuda:0` if it founds cuda is available. To change the device id, specify `--device_id`:

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--device_id 0 \
--out_path generation.wav
```

You can also try other voices. Check more example voices in [examples/voice_prompts](./examples/voice_prompts). You can also add your own voice to the folder.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio broom_salesman \
--temperature 0.3 \
--out_path generation.wav
```

### Single-speaker Generation with Smart Voice
If you do not specify reference voice, the model will decide the voice based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--temperature 0.3 \
--out_path generation.wav
```


### Multi-speaker Dialog with Smart Voice
Generate multi-speaker dialog. The model will decide the voices based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-speaker Dialog with Voice Clone

Generate multi-speaker dialog with the voices you picked.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```


## Technical Details

For more technical details, please refer to Boson AI's original [blog post](https://www.boson.ai/blog/higgs-audio-v2). This is just an inference repository.

## Credits

Higgs Audio v2 is developed by [Boson AI](https://www.boson.ai/). This is a fantastic model, thank you so much for open-sourcing it, guys! 

## Citation

If you feel the repository is helpful, please kindly cite them directly:

```
@misc{higgsaudio2025,
  author       = {{Boson AI}},
  title        = {{Higgs Audio V2: Redefining Expressiveness in Audio Generation}},
  year         = {2025},
  howpublished = {\url{https://github.com/boson-ai/higgs-audio}},
  note         = {GitHub repository. Release blog available at \url{https://www.boson.ai/blog/higgs-audio-v2}},
}
```

## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.