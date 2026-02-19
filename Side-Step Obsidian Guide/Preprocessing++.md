# Preprocessing++

Preprocessing++ is Side-Step's **dataset-aware adapter targeting** system. It uses Fisher Information Matrix (FIM) analysis to determine which parts of the model are most important for *your specific audio data*.

Instead of training every layer with the same capacity (flat rank), Preprocessing++:
1.  **Auto-targets modules**: Finds which attention blocks and MLPs react most to your audio.
2.  **Assigns adaptive ranks**: Allocates more parameters (higher rank) to important layers and fewer to less relevant ones.
3.  **Saves a map**: Writes `fisher_map.json` to your dataset folder, which training automatically detects and uses.

---

## Why Use It?

- **Efficiency**: Puts parameters where they matter most.
- **Quality**: Often captures texture and timbre better than flat-rank training.
- **Stability**: Can reduce "catastrophic forgetting" by leaving irrelevant model parts alone.

> [!WARNING] **Power User Tool**
> Preprocessing++ is extremely effective at adaptation, which means it can **overfit** faster than standard training.
> - **Lower your learning rate** (e.g., if you usually use `1e-4`, try `5e-5`).
> - **Monitor closely** (check samples early).
> - **Turbo Users**: Be careful. Turbo models are fragile. Preprocessing++ can destabilize Turbo training. **Base** models are recommended for this workflow.

---

## How It Works

1.  **Prerequisite**: You must have already run standard **Preprocessing** to generate `.pt` tensor files in a dataset directory.
2.  **Analysis**: You run Preprocessing++ on that dataset directory.
3.  **Output**: It generates a `fisher_map.json` file inside the dataset directory.
4.  **Training**: When you later run training on that dataset, Side-Step detects the map and automatically switches to "Adaptive Rank" mode.

---

## Running via Wizard

1.  Select **Preprocessing++ (auto-targeting + adaptive ranks)** from the main menu.
2.  **Model Selection**: Pick the model you intend to train on (usually `base`).
3.  **Dataset**: Point to your folder of `.pt` files.
4.  **Timestep Focus**: Choose how the analysis should "listen" to your audio:
    - **Texture** (Default): Focuses on timbre and sound design. Best for style transfer.
    - **Structure**: Focuses on rhythm, beat grids, and composition.
    - **Balanced**: A mix of both.
5.  **Rank Budget**:
    - **Base Rank**: The median target (e.g., 64).
    - **Min/Max**: The floor and ceiling for adaptive ranks (e.g., 16 to 128).
6.  **Confirm**:
    - The wizard shows a final confirmation summary (dataset size, focus, rank budget, estimate).
    - If Turbo is selected, the Turbo fragility warning appears here with explicit opt-in.
7.  **Run**: The process takes a few minutes depending on dataset size.

You can optionally load preset-derived rank defaults before stepping through the flow (only overlapping fields are applied).

---

## Running via CLI

Use the `fisher` subcommand:

```bash
uv run train.py fisher \
  --checkpoint-dir ./checkpoints \
  --model-variant base \
  --dataset-dir ./my_dataset_tensors \
  --rank 64 \
  --rank-min 16 \
  --rank-max 128 \
  --timestep-focus texture
```

### Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset-dir` | Path to folder with `.pt` files (Required) | - |
| `--timestep-focus` | `texture`, `structure`, `balanced` | `texture` |
| `--rank` | Target median rank | `64` |
| `--rank-min` | Minimum rank for any layer | `16` |
| `--rank-max` | Maximum rank for any layer | `128` |
| `--convergence-patience` | Stop analysis when stable for N batches | `5` |

---

## Troubleshooting

### "Preprocessing++ map detected"
If you see this during training start-up, it means the system found `fisher_map.json`. It will print details about the adaptive ranks.

### Ignoring the Map
To force standard flat-rank training on a dataset that has a map, use:
`--ignore-fisher-map`

### Turbo Instability
If training collapses (loss spikes, output is noise) on Turbo with Preprocessing++, delete `fisher_map.json` and retry with standard flat ranks, or switch to a **Base** model.
