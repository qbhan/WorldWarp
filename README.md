<div align="center">
  <img src="assets/teaser.png">

<a href="https://hyokong.github.io/worldwarp-page/"><h1>🌏 WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion 🌀</h1></a>
</h2>
</div>

<h5 align="center">

[![Home Page](https://img.shields.io/badge/Project-Website-33728E.svg)](https://hyokong.github.io/worldwarp-page/) 
[![arXiv](https://img.shields.io/badge/Arxiv-2512.xxxxx-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2512.xxxxx) 
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/imsuperkong/worldwarp) [![Watch on YouTube](https://img.shields.io/badge/YouTube-Demo_Video-red?style=flat&logo=youtube)](https://www.youtube.com/watch?v=rfMHxb--cKs)


[Hanyang Kong](https://hyokong.github.io/),
[Xingyi Yang](https://adamdad.github.io/),
Xiaoxu Zheng,
[Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
</h5>

**TL;DR**: 🔭 Single-image long-range view generation via an <u>asynchronous chunk-wise autoregressive diffusion framework</u> that utilizes <u>explicit camera conditioning</u> and <u>online 3D cache</u> for geometric consistency.


## 🎬 Demo Video

▶️ **Click the GIF to watch the full video with sound.**

<p align="center">
  <a href="https://www.youtube.com/watch?v=rfMHxb--cKs">
    <img src="assets/web_teaser.gif" alt="WorldWarp Demo" width="100%">
  </a>
</p>

## 🛠️ Installation

> ⚠️ **Hardware Note:** The current implementation requires high GPU memory (~40GB VRAM). We are currently optimizing the code to reduce this footprint.

### 🧬 Cloning the Repository
The repository contains submodules, thus please check it out with
```bash
git clone https://github.com/HyoKong/WorldWarp.git --recursive
cd WorldWarp
```

### 🐍 Create environment

Create a conda environment and install dependencies:
```
conda create -n worldwarp python=3.12 -y
conda activate worldwarp
```

### 🔥 Install PyTorch
Install PyTorch with CUDA 12.6 support (or visit [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) for other CUDA configurations):
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

### 📦 Install Dependencies & Compile Extensions
These packages require compilation against the specific PyTorch version installed above.

```bash
# Core compiled dependencies
pip install flash-attn --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Local modules
pip install src/fused-ssim/ --no-build-isolation
pip install src/simple-knn/ --no-build-isolation

# Remaining python dependencies
pip install -r requirements.txt
```



### 🏗️ Build Other Extensions
```bash
cd src/ttt3r/croco/models/curope/
python setup.py build_ext --inplace
cd -  # Returns to the project root
```


## ☁️ Download checkpoints

```
mkdir ckpt
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ckpt/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ckpt/Qwen/Qwen2.5-VL-7B-Instruct
hf download imsuperkong/worldwarp --local-dir ckpt/

cd src/ttt3r/
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../..
```

## 🎨 GUI Demo

```bash
python gradio_demo.py
```

The web interface will open at `http://localhost:7890`. 

---

### 🚀 Quick start:

**1️⃣ Choose Starting Image**

- **📚 Examples Tab**: Click a pre-made example image (prompt auto-fills)
- **🎨 Generate Tab**: Click "Generate First Frame" from your prompt
- **📤 Upload Tab**: Upload your own image

**2️⃣ Select Camera Movement** (Recommended: 📹 From Video)

- **From Video** (Easiest and most reliable)
  - Click **"📹 From Video"** mode
  - Select an example video from the gallery OR upload your own
  - Click **"🎯 Load Poses"** to extract camera trajectory
  - Poses are automatically cached for reuse

- **Preset Movements**
  - Select **"🎯 Preset"** mode
  - Choose movements: `DOLLY_IN`, `PAN_LEFT`, `PAN_RIGHT`, etc.
  - Can combine: e.g., `DOLLY_IN + PAN_RIGHT`

- **Custom** (Advanced)
  - Select **"🔧 Custom"** mode
  - Manually control rotation and translation parameters

**3️⃣ Configure & Generate**

**Essential Parameters:**

- 💪 **Strength (0.5 - 0.8)**
  - **Higher (0.7-0.8)**: More generated details, richer content
    - ⚠️ May introduce content changes due to higher creative freedom
  - **Lower (0.5-0.6)**: More accurate camera control, closer to input
    - ⚠️ May produce blurry results due to limited diffusion model freedom
  - **Trade-off**: Higher strength = more details but less control; Lower strength = better control but potentially blurry

- ⚡ **Speed Multiplier**
  - **Purpose**: Adjust camera movement velocity to match your scene scale
  - **Why needed**: Reference video's camera movement scale may not match your scene (e.g., drone video moving 10 meters may be too fast for a small room)
  - **< 1.0**: Slower camera movement (e.g., 0.5 = half speed)
  - **= 1.0**: Original speed from reference
  - **> 1.0**: Faster camera movement (e.g., 2.0 = double speed)
  - **Tip**: Start with 1.0, then adjust based on whether motion feels too fast or too slow

---

#### 🌟 Best Practices

- 👁️ **Generate one chunk at a time**
  - Lets you preview each chunk's quality before continuing
  - Easier to identify issues early

- ↩️ **Use Rollback for iteration**
  - If a chunk is unsatisfactory, enter its number in **"Rollback to #"**
  - Click **"✂️ Rollback"** to remove it
  - Adjust parameters and regenerate

- 🏎️ **Adjust Speed Multiplier per scene**
  - If camera moves too fast → decrease value (e.g., 0.5-0.7)
  - If camera moves too slow → increase value (e.g., 1.5-2.0)






## 🙌 Acknowledgements
Our code is based on the following awesome repositories:

- [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer)
- [TTT3R](https://github.com/Inception3D/TTT3R)

We thank the authors for releasing their code!

## 📖 Citation

If you find our work useful, please cite:

```bibtex
@misc{kong2025worldwarp,
  title={WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion}, 
  author={Hanyang Kong and Xingyi Yang and Xiaoxu Zheng and Xinchao Wang},
  year={2025},
  eprint={2512.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```