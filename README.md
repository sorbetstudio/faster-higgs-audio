<h1 align="center">⚡ Faster Higgs: Local Quantized Audio Generation</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="#-instant-setup"><img src='https://img.shields.io/badge/⚡-30 Second Setup-FFD700' style="margin-right: 5px;"></a>
  <a href="#-quantized-inference"><img src="https://img.shields.io/badge/🔧-4bit/8bit Quantization-32CD32" style="margin-right: 5px;"></a>
  <a href="#-server-mode"><img src="https://img.shields.io/badge/🌐-Local Server + Web UI-4169E1" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base"><img src="https://img.shields.io/badge/🤗-Original Model-ED5A22.svg" style="margin-right: 5px;"></a>
</div>


**Faster Higgs** optimizes the powerful Higgs Audio v2 model for local deployment with aggressive quantization, making high-quality voice generation accessible on consumer GPUs. Run the full model (3.6B + 2.2B params) on **8GB VRAM** or even **CPU-only** systems.

🎯 **What's New:**
- ⚡ **4-bit & 8-bit quantization** - Run on 8GB GPUs
- 🖥️ **CPU fallback** - Works without any GPU
- 🌐 **Local server mode** - OpenAI-compatible API + web interface
- 📦 **30-second setup** - One command to start generating
- 🔧 **Smart device detection** - Automatically optimizes for your hardware

<p align="center">
    <img src="https://img.shields.io/badge/GPU-8GB_VRAM-green?style=for-the-badge" alt="8GB VRAM">
    <img src="https://img.shields.io/badge/CPU-Compatible-blue?style=for-the-badge" alt="CPU Compatible">
    <img src="https://img.shields.io/badge/Setup-30_Seconds-yellow?style=for-the-badge" alt="30 Second Setup">
</p>

**Hardware Requirements:**
- 🟢 **Optimal:** GPU with 12GB+ VRAM (no quantization needed)
- 🟡 **Good:** GPU with 8GB VRAM (8-bit quantization)
- 🟠 **Works:** GPU with 6-8GB VRAM (4-bit quantization)
- 🔵 **Fallback:** CPU-only (slower but universal)

## ⚡ Instant Setup

Get Higgs Audio running with quantization in 30 seconds:

```bash
# Clone and setup (30 seconds)
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# One-command install with quantization support
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt -e . bitsandbytes

# Test immediately - works on any GPU or CPU!
./run_tts.sh "Hello world, this is Faster Higgs!"
```

## 🔧 Quantized Inference

**Automatically optimized for your hardware:**

```bash
# Auto-detects best settings (recommended)
./run_tts.sh "Smart optimization picks the best settings for your GPU!"

# Force 8-bit quantization (8GB+ GPUs)
./run_tts.sh "Higher quality on medium GPUs" --quantization_bits 8

# Force 4-bit quantization (6-8GB GPUs)
./run_tts.sh "Maximum compression for small GPUs" --quantization_bits 4

# CPU-only mode (universal compatibility)
./run_tts.sh "Works on any computer!" --device cpu

# Voice cloning with quantization
./run_tts.sh "Clone this voice efficiently" --ref_audio belinda --quantization_bits 8
```

### 📊 Performance & Memory

| Hardware      | Mode            | Memory         | Speed | Quality |
| ------------- | --------------- | -------------- | ----- | ------- |
| **12GB+ GPU** | No quantization | idk haha       | ?     | 🌟🌟🌟     |
| **8GB GPU**   | 8-bit quantized | ~7GB           | ⚡     | 🌟🌟      |
| **6GB GPU**   | 4-bit quantized | ~5GB           | ⚡⚡    | 🌟       |
| **CPU Only**  | Any mode        | ~21GB RAM Smth | 🐌     | 🌟       |

**Real Performance:**
- **8-bit quantized:** ~4-6 seconds per sentence (8GB GPU) (sweet)
- **4-bit quantized:** ~3-5 seconds per sentence (6GB GPU) (not the best in it's current state)
- **CPU mode:** ~60+ seconds per sentence (universal)

## 🌐 Server Mode

**Deploy with OpenAI-compatible API + Web Interface:**

```bash
# Start local server with web UI (one command)
./run_server.sh

# Access at:
# 🌐 Web Interface: http://localhost:8501
# 📡 API Server: http://localhost:8000
```

**Features:**
- 🎨 **Streamlit Web UI** - Easy point-and-click interface
- 📡 **OpenAI API Compatible** - Drop-in replacement for OpenAI TTS
- 🎭 **Voice Cloning Interface** - Upload audio, get cloned voice
- 👥 **Multi-Speaker Generation** - Generate conversations
- 🔧 **Auto-quantization** - Optimizes for your hardware
- 🌐 **Network Access** - Use from other devices on your network

---

## 🐳 Docker Option (Optional)

For isolated environments, use NVIDIA containers:

```bash
# Run in NVIDIA container (optional)
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash

# Then install normally:
git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio
uv pip install -r requirements.txt -e . bitsandbytes
./run_tts.sh "Docker deployment works too!"
```

## 📦 Alternative Install Methods

<details>
<summary>🔽 Click to expand alternative installation options</summary>

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

</details>

## 🚀 Usage

> [!TIP]
> **Use `./run_tts.sh` for instant speech generation!** Automatically detects your hardware and applies optimal quantization. Works on 6GB GPUs or CPU-only.

> [!NOTE] 
> **For server deployment:** Use `./run_server.sh` to launch the web interface with OpenAI-compatible API.

### Quantization Support

Higgs Audio v2 supports 4-bit and 8-bit quantization to reduce memory usage and improve accessibility. Quantization is enabled by default and automatically optimized for your hardware.

**Quantization Modes:**
- **4-bit:** Most aggressive compression, works on 6GB+ GPUs
- **8-bit:** Balanced quality/memory, ideal for 8GB GPUs  
- **No quantization:** Best quality, requires 12GB+ GPUs

### Enhanced TTS Script Usage

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
🎵 Audio validation passed - real speech content detected
```

#### Device and Memory Management

- **Auto-detection**: Automatically chooses the best available device
- **GPU < 8GB**: Uses 4-bit quantization by default (most memory efficient)
- **GPU 8GB**: Recommended to use 8-bit quantization with `--quantization_bits 8`
- **GPU ≥ 12GB**: Can optionally disable quantization with `--no_quantization`
- **CPU fallback**: Always available with `--device cpu`
- **Memory cleanup**: Automatically manages CUDA memory and prevents conflicts

> [!NOTE]
> For more detailed TTS usage examples, troubleshooting, and performance tips, see [TTS_USAGE.md](./TTS_USAGE.md)

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
<img src="figures/higgs_audio_v2_architecture_combined.png" width=900>


Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:
- We developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as **AudioVerse**. The in-house understanding model is finetuned on top of [Higgs Audio v1 Understanding](https://www.boson.ai/blog/higgs-audio), which adopts the "understanding variant" shown in the architecture figure.
- We trained a unified audio tokenizer from scratch that captures both semantic and acoustic features. Learn more in the [tokenizer blog](./tech_blogs/TOKENIZER_BLOG.md).
- We proposed the DualFFN architecture, which enhances the LLM's ability to model acoustics tokens with minimal computational overhead. See the [architecture blog](./tech_blogs/ARCHITECTURE_BLOG.md).

## Evaluation

Here's the performance of Higgs Audio v2 on four benchmarks,  [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval), [Emotional Speech Dataset (ESD)](https://paperswithcode.com/dataset/esd), [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), and Multi-speaker Eval:

#### Seed-TTS Eval & ESD

We prompt Higgs Audio v2 with the reference text, reference audio, and target text for zero-shot TTS. We use the standard evaluation metrics from Seed-TTS Eval and ESD.

|                            | SeedTTS-Eval |           | ESD      |                 |
| -------------------------- | ------------ | --------- | -------- | --------------- |
|                            | WER ↓        | SIM ↑     | WER ↓    | SIM (emo2vec) ↑ |
| Cosyvoice2                 | 2.28         | 65.49     | 2.71     | 80.48           |
| Qwen2.5-omni†              | 2.33         | 64.10     | -        | -               |
| ElevenLabs Multilingual V2 | **1.43**     | 50.00     | 1.66     | 65.87           |
| Higgs Audio v1             | 2.18         | 66.27     | **1.49** | 82.84           |
| Higgs Audio v2 (base)      | 2.44         | **67.70** | 1.78     | **86.13**       |


#### EmergentTTS-Eval ("Emotions" and "Questions")

Following the [EmergentTTS-Eval Paper](https://arxiv.org/abs/2505.23009), we report the win-rate over "gpt-4o-mini-tts" with the "alloy" voice. The judge model is Gemini 2.5 Pro.

| Model                                                                                      | Emotions (%) ↑ | Questions (%) ↑ |
| ------------------------------------------------------------------------------------------ | -------------- | --------------- |
| Higgs Audio v2 (base)                                                                      | **75.71%**     | **55.71%**      |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)      | 61.64%         | 47.85%          |
| [Hume.AI](https://www.hume.ai/research)                                                    | 61.60%         | 43.21%          |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)   | 50.00%         | 50.00%          |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)                                   | 41.60%         | 51.78%          |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)                         | 40.86%         | 47.32%          |
| [ElevenLabs Multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)            | 30.35%         | 39.46%          |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech) | 29.28%         | 48.21%          |
| [Sesame csm-1B](https://github.com/SesameAILabs/csm)                                       | 15.96%         | 31.78%          |

<sup><sub>'†' means using the strong-prompting method described in the paper.</sub></sup>


#### Multi-speaker Eval

We also designed a multi-speaker evaluation benchmark to evaluate the capability of Higgs Audio v2 for multi-speaker dialog generation. The benchmark contains three subsets

- `two-speaker-conversation`: 1000 synthetic dialogues involving two speakers. We fix two reference audio clips to evaluate the model's ability in double voice cloning for utterances ranging from 4 to 10 dialogues between two randomly chosen persona.
- `small talk (no ref)`: 250 synthetic dialogues curated in the same way as above, but are characterized by short utterances and a limited number of turns (4–6), we do not fix reference audios in this case and this set is designed to evaluate the model's ability to automatically assign appropriate voices to speakers.
- `small talk (ref)`: 250 synthetic dialogues similar to above, but contains even shorter utterances as this set is meant to include reference clips in it's context, similar to `two-speaker-conversation`.


We report the word-error-rate (WER) and the geometric mean between intra-speaker similarity and inter-speaker dis-similarity on these three subsets. Other than Higgs Audio v2, we also evaluated [MoonCast](https://github.com/jzq2000/MoonCast) and [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626), two of the most popular open-source models capable of multi-speaker dialog generation. Results are summarized in the following table. We are not able to run [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) on our "two-speaker-conversation" subset due to its strict limitation on the length of the utterances and output audio.

|                                                                           | two-speaker-conversation |                      | small talk |                      | small talk (no ref) |                      |
| ------------------------------------------------------------------------- | ------------------------ | -------------------- | ---------- | -------------------- | ------------------- | -------------------- |
|                                                                           | WER ↓                    | Mean Sim & Dis-sim ↑ | WER ↓      | Mean Sim & Dis-sim ↑ | WER ↓               | Mean Sim & Dis-sim ↑ |
| [MoonCast](https://github.com/jzq2000/MoonCast)                           | 38.77                    | 46.02                | **8.33**   | 63.68                | 24.65               | 53.94                |
| [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) | \\-                      | \\-                  | 17.62      | 63.15                | 19.46               | **61.14**            |
| Higgs Audio v2 (base)                                                     | **18.88**                | **51.95**            | 11.89      | **67.92**            | **14.65**           | 55.28                |


## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.