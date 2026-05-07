# Dream Worlds: Physics Simulation on Gaussian Splats

![Course](https://img.shields.io/badge/Course-CGAI-blue)
![Track](https://img.shields.io/badge/Track-Technical%20Exploration-0b7285)
![Topic](https://img.shields.io/badge/Topic-Physics%20Simulation%20on%20Gaussian%20Splats-2b8a3e)
![Status](https://img.shields.io/badge/Status-Working%20Prototype-228be6)

This repository contains our CGAI Dream Worlds final project on adding physical dynamics to Gaussian Splatting. We start from pretrained 3D Gaussian Splatting scene representations and study how simple physics-based updates can be applied to Gaussian primitives so that they exhibit motion while preserving recognizable visual structure.

The current repository is a working prototype centered around one notebook, `3dgs_testing.ipynb`, with two exploratory experiments and exported demo videos.

## Project Overview

Gaussian Splatting is an effective representation for high-quality rendering, but it is typically used for static scenes. This project explores what happens when Gaussian primitives are treated as dynamic elements whose positions can be updated over time using simple physical rules.

Our goal is not to reproduce a full physically accurate pipeline. Instead, we are using a lightweight prototype to study a narrower question: how can pretrained Gaussian splat representations be modified to exhibit motion while still preserving the object's recognizable appearance?

## Research Question

How can pretrained Gaussian splat representations be updated with simple physical motion while maintaining visual coherence in the rendered object?

## Relation to Prior Work

This project is inspired by recent work such as PhysGaussian and GASP, which explore physically grounded motion and simulation-aware manipulation for 3D Gaussians. Those systems are significantly more complete and compute-intensive than the prototype in this repository.

Our implementation is deliberately smaller in scope. It focuses on exploratory motion rules applied to pretrained Gaussians so that we can study behavior, rendering effects, and practical limitations before attempting a more complete simulation pipeline.

## Current Prototype Status

- One working notebook prototype: `3dgs_testing.ipynb`
- Two current experiments: wall smash and mass falling
- Demo videos stored under `assets/demos/`
- Local pretrained inputs and generated frames stored under `output/` and ignored by git

## Repository Layout

```text
.
├── assets/
│   └── demos/
│       ├── mass_falling.mp4
│       └── wall_smash.mp4
├── output/
│   └── ficus_whitebg-trained/
│       ├── cameras.json
│       ├── depths/
│       ├── images/
│       ├── point_cloud/
│       └── results/
├── 3dgs_testing.ipynb
├── README.md
├── environment.yml
└── requirements.txt
```

`output/` is used for local pretrained assets and generated frame sequences, so it is intentionally git-ignored.

## Requirements

- Linux or WSL is recommended
- NVIDIA GPU
- Python 3.10
- `conda`
- `pip`
- `ffmpeg`

If you are on an RTX 50-series GPU, use the CUDA 12.8 PyTorch wheels and prefer Linux or WSL.

## Setup

Clone the repository and create the environment:

```bash
git clone https://github.com/MWalkGATech/CGAI-Final-Project-Dream-Worlds.git
cd CGAI-Final-Project-Dream-Worlds

conda env create --file environment.yml -y
conda activate gaussian_splatting

export TORCH_CUDA_ARCH_LIST="12.0"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-build-isolation --upgrade
```

If `ffmpeg` is missing:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

The repository installs a fork of Gaussian Splatting from `requirements.txt`. That fork includes compatibility fixes needed for the current setup path.

## Data

The notebook expects the pretrained ficus scene under `output/ficus_whitebg-trained/`.

Source:

- [Pretrained Sources that work](https://drive.google.com/drive/folders/1Bl51dHBoTt08T3RBtslM93UIIk9C_gSB?usp=sharing)

For a headless Linux setup, this terminal-only flow works:

```bash
conda activate gaussian_splatting
python -m pip install gdown

mkdir -p output
gdown --folder "https://drive.google.com/drive/folders/1Bl51dHBoTt08T3RBtslM93UIIk9C_gSB?usp=sharing" -O output

unzip output/ficus_whitebg-trained.zip -d output
mkdir -p output/ficus_whitebg-trained/results
```

After extraction, the important paths should exist:

```text
output/ficus_whitebg-trained/cameras.json
output/ficus_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply
output/ficus_whitebg-trained/results/
```

## Running the Notebook

1. Open `3dgs_testing.ipynb` in VS Code or Jupyter.
2. Select the `gaussian_splatting` Python kernel.
3. If you want a clean rerun, delete old rendered PNGs:

   ```bash
   rm -f output/ficus_whitebg-trained/results/*.png
   ```

4. Run the notebook from top to bottom.
5. The render cells write PNG frames into `output/ficus_whitebg-trained/results/`.
6. The export cells write demo videos to:

   - `assets/demos/wall_smash.mp4`
   - `assets/demos/mass_falling.mp4`

## Current Experiments

### Wall Smash

Applies uniform gravity in the negative `y` direction to all Gaussians and clamps positions into the range `[-5, 5]`.

### Mass Falling

Applies gravity in the negative `z` direction and scales motion by randomized inverse mass, then clamps positions into the range `[-10, 10]`.

These experiments are simple baselines rather than full physical models. They are intended to provide a starting point for studying how direct motion updates affect the rendered appearance of pretrained Gaussian splats.

## Known Limitations

- The project is still notebook-first and not yet refactored into reusable Python modules.
- Paths are currently hardcoded to the ficus pretrained scene.
- The current physics rules are intentionally simple and exploratory.
- There is no automated test suite yet.
- There is no CLI or configuration system yet.

## Next Steps

- Refactor loading, rendering, and update logic out of the notebook
- Add more physics update rules and scene experiments
- Compare simple baselines against more structured physical updates
- Introduce lightweight tests once the physics update code stabilizes

## References

- Official final project document: https://cgai-gatech.vercel.app/assignment/Final_doc
- Gaussian Splatting fork used in this repo: https://github.com/yindaheng98/gaussian-splatting.git
- PhysGaussian: https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Neural_Physics-Informed_3D_Gaussians_for_Physics-Based_View_Synthesis_CVPR_2024_paper.html
- GASP: https://www.sciencedirect.com/science/article/pii/S0097849325001774
- CUDA 12.8 install notes:
  - https://github.com/graphdeco-inria/gaussian-splatting/issues/1215
  - https://github.com/graphdeco-inria/gaussian-splatting/issues/1313