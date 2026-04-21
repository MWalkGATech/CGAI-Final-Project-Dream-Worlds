# Dream Worlds: Physics Simulation on Gaussian Splats

![Course](https://img.shields.io/badge/Course-CGAI-blue)
![Track](https://img.shields.io/badge/Track-Technical%20Exploration-0b7285)
![Topic](https://img.shields.io/badge/Topic-Physics%20Simulation%20on%20Gaussian%20Splats-2b8a3e)
![Status](https://img.shields.io/badge/Status-Planning-f08c00)
![Deadline](https://img.shields.io/badge/Deadline-May%207%2C%202026-c92a2a)

## Introduction

This repository contains our CGAI final project for the Dream Worlds technical exploration track. Our goal is to study how physics-based motion can be coupled with Gaussian splat representations to create dynamic, controllable, and visually coherent scenes.

## Description

We are exploring a technical question at the intersection of particle simulation and point-based rendering: how should physically simulated state drive Gaussian splat attributes such as position, orientation, scale, and opacity? The project will focus on building a prototype pipeline, evaluating its stability and rendering behavior, and documenting the results in a short scientific paper.

Our current direction is to investigate whether a physics layer can make Gaussian splats more expressive for dynamic scenes while preserving rendering quality and reasonable performance. This aligns with the course focus on Gaussian splats and physics, and fits the technical exploration track requirement to study one course topic in depth.

## Team

- Ethan Villalovoz
- Michael Walker

Final per-member contribution details will be added before submission.

## Project Scope

This project is being developed for the CGAI Dream Worlds final project under the technical exploration track. The planned submission includes:

- A technical report in short paper style (3-4 pages)
- Source code for the implementation
- Qualitative and quantitative results
- A clear breakdown of team contributions

Course deadline: May 7, 2026 at 11:59 pm ET.

## Visuals

Visual results, comparison figures, and simulation clips will be added here as the implementation matures.

Suggested future additions:

- A short animated GIF comparing static splats against physics-driven splats
- Side-by-side render comparisons for different simulation settings
- Plots for runtime, stability, or reconstruction quality

## Prerequisites / Requirements

The exact stack may change during implementation, but the current baseline assumptions are:

- Git
- Python 3.10 or newer
- `pip` or Conda for dependency management
- A machine with an NVIDIA GPU is recommended if the implementation relies on CUDA acceleration
- A report-writing workflow such as LaTeX, Overleaf, or a similar editor

## Technologies

The project is expected to use a focused subset of the following technologies:

- Python
- PyTorch
- NumPy
- CUDA, if GPU acceleration is needed
- Matplotlib or similar tools for plots and analysis
- Markdown and Mermaid for project documentation and diagrams

## QuickStart

Use the following commands to bootstrap the repository locally:

```bash
git clone <repo-url>
cd CGAI-Final-Project

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
```

As the codebase is added, this section will expand with the shortest path for running the baseline experiment and reproducing report figures.

## Advanced Usage

Planned advanced workflows for this repository include:

- Running ablation studies on simulation parameters such as timestep, solver iterations, and particle count
- Comparing a static Gaussian splat baseline against a physics-driven variant
- Exporting frame sequences, metrics, and plots for the final report
- Testing different ways of mapping physical state to Gaussian splat parameters

## Configuration

The implementation will likely expose configuration through experiment files or command-line flags. The main parameters we expect to tune are:

| Parameter | Purpose |
| --- | --- |
| `scene_name` | Names the experiment or scene configuration |
| `num_splats` | Controls the number of Gaussian splats in the scene |
| `time_step` | Sets the physics integration step size |
| `solver_iterations` | Controls numerical stability and constraint convergence |
| `gravity` | Defines external acceleration applied to the system |
| `splat_update_rule` | Chooses how physics modifies splat parameters |
| `output_dir` | Selects where renders, logs, and plots are saved |

## Automated Tests

Automated tests are not wired into the repository yet. The intended validation plan is to add:

- Unit tests for physics updates and state transitions
- Regression tests for splat parameter updates
- Reproducibility checks for fixed random seeds
- Lightweight experiment smoke tests for core configs

Once the test suite exists, the default local command should be:

```bash
pytest -q
```

## Planned Repository Structure

As the project grows, we expect the repository to follow a structure similar to:

```text
.
├── README.md
├── assets/
├── configs/
├── docs/
├── outputs/
├── src/
└── tests/
```

## Roadmap

- [ ] Select the implementation baseline and experiment scope
- [ ] Build the first physics-driven Gaussian splat prototype
- [ ] Define evaluation criteria for quality, stability, and performance
- [ ] Run ablations and collect visual results
- [ ] Write the technical report
- [ ] Prepare final submission materials and gallery-ready media

## Contribution

This is currently a two-person course project. For now, contributions will be coordinated directly between the team members. Before the final submission, this section will be updated with a clear description of each member's implementation, experimentation, and writing contributions.

If the repository later opens to outside collaboration, a dedicated contribution guide can be added in `CONTRIBUTING.md`.

## References

- Official final project document: https://cgai-gatech.vercel.app/assignment/Final_doc
- Course theme: Dream Worlds
- Chosen direction: Technical Exploration on physics simulation over Gaussian splats