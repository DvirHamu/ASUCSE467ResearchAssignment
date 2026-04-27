# ASUCSE467ResearchAssignment

Natural Datasets, locally run results, attack logs and weights not included in the repository due to space constraints can be found in [this google drive folder](https://drive.google.com/drive/folders/1mmpZwrlU040ifP4AaxqF0VLRftvv2WI6).


---


<img width="1918" height="834" alt="image6" src="https://github.com/user-attachments/assets/0862ee78-8853-4798-8578-68d4c33f19f4" />

Go to scratch directory and open terminal

```
mkdir CSE467
cd CSE467
git clone https://github.com/minha12/DiffPrivate
```
We’ll be using apptainer instead of docker to run this


Getting resources to ASU SOL

When you first log in you will be on the login node, the compute resources allocated to this is low. DO not run these commands on the login node. Or else you’ll get this.
<img width="1287" height="728" alt="image1" src="https://github.com/user-attachments/assets/ddde05c3-acd1-4b3f-8b7f-ff3fe4d71c7b" />


‘’’
No need for now use gpu
This is for cpu
srun --partition=public --mem=64G --cpus-per-task=8 --time=2:00:00 --pty bash
‘’’

This is for gpu
```
salloc -G a100:1
```

```
Run apptainer on the repo #make sure your on the gpu node!!!

apptainer pull docker://hale0007/diffprivate:latest
```

Before you start the shell there is some configuration you need to do
```
nano configs/config.yaml
```

Change the path to
```
pretrained_diffusion_path: "Manojb/stable-diffusion-2-1-base"
```
Previous one was 404

Then to get a shell you run
```
apptainer shell --nv --bind /scratch/dhamu/CSE469/DiffPrivate:/app/DiffPrivate diffprivate_latest.sif
```

Open a new tab and log into your hugging face
https://huggingface.co/settings/tokens

Make a read token and save the key


Back in the shell run the command 
```
huggingface-cli login
```
Paste your token, then Y 


Token should look like this
hf_ld”...”
```
python run-dpp.py paths.images_root=./data/demo/images paths.save_dir=./data/output
```
Then once done 
Go back to ASU SOL gui

And you can download onto your computer

<img width="1581" height="768" alt="image2" src="https://github.com/user-attachments/assets/09fa3839-9a26-414e-85c3-8763d89794c3" />


<img width="256" height="256" alt="image3" src="https://github.com/user-attachments/assets/dfdab0ef-4e7e-45b7-ab78-d2bf334178c5" />
<img width="512" height="256" alt="image5" src="https://github.com/user-attachments/assets/acdbff13-5fa3-45ec-8580-2f49dc8cc235" />
<img width="256" height="256" alt="image4" src="https://github.com/user-attachments/assets/1a053923-2021-4aec-b35c-8418339464a7" />
<img width="256" height="256" alt="image8" src="https://github.com/user-attachments/assets/653f7881-b9cc-451f-8696-5af87dc4a227" />
<img width="256" height="256" alt="image7" src="https://github.com/user-attachments/assets/e6123353-ae9c-47b6-aca8-8b7ecc31c312" />




 The base model stays as Manojb/stable-diffusion-2-1-base. The LoRA weights are loaded ON TOP of it in the code.

Edit run-dpp.py to load the LoRA weights


bashnano /app/DiffPrivate/run-dpp.py


Find this line (around line 20):


pythonldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to("cuda:0")


Add two lines right after it:


pythonldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to("cuda:0")


pythonldm_stable.load_lora_weights("/app/DiffPrivate/fine_tuned_lora/checkpoint-1000")

ldm_stable.fuse_lora()


```python3 -c "
from safetensors.torch import load_file, save_file

weights = load_file('/app/DiffPrivate/fine_tuned_lora/pytorch_lora_weights.safetensors')

new_weights = {}
for k, v in weights.items():
    # Convert new format to old format
    # unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight
    # -> unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k_lora.down.weight
    new_key = k.replace('unet.', 'unet_')
    new_key = new_key.replace('.lora_A.weight', '_lora.down.weight')
    new_key = new_key.replace('.lora_B.weight', '_lora.up.weight')
    new_key = new_key.replace('.', '_').replace('_lora_down_weight', '_lora.down.weight').replace('_lora_up_weight', '_lora.up.weight')
    new_weights[new_key] = v

save_file(new_weights, '/app/DiffPrivate/fine_tuned_lora/pytorch_lora_weights_converted.safetensors')

# Print first 3 converted keys to verify
for k in list(new_weights.keys())[:3]:
    print(k)
print('Done!')
"```


```
ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to("cuda:0")
ldm_stable.load_lora_weights("/app/DiffPrivate/fine_tuned_lora/pytorch_lora_weights.safetensors")
ldm_stable.fuse_lora()
ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)```


update config/config.yaml file


```model:
  attacker_model: "irse50"
  victim_model: "irse50"
  list_attacker_models: ["irse50"]
  ensemble: false
  ensemble_mode: "mean"
  lora_path: null```


```2. Update run-dpp.py

Right now your script always loads LoRA here:

from safetensors.torch import load_file
lora_state = load_file("/app/DiffPrivate/fine_tuned_lora/pytorch_lora_weights.safetensors")
lora_state = {k.replace("unet.", ""): v for k, v in lora_state.items()}
ldm_stable.unet.load_state_dict(lora_state, strict=False)

Replace that whole section with this:

# Optional LoRA loading
if cfg.model.lora_path is not None:
    from safetensors.torch import load_file

    lora_file = os.path.join(cfg.model.lora_path, "pytorch_lora_weights.safetensors")
    print(f"Loading LoRA weights from {lora_file}")

    lora_state = load_file(lora_file)
    lora_state = {k.replace("unet.", ""): v for k, v in lora_state.items()}
    ldm_stable.unet.load_state_dict(lora_state, strict=False)
```


```chmod +x overnight_compare.sh
nohup ./overnight_compare.sh > overnight_master.log 2>&1 &
tail -f overnight_master.log
```



---

## LoRA Extension: Fine-Tuning for Improved Privacy Protection

### Overview

We extend DiffPrivate by fine-tuning the base Stable Diffusion model using LoRA (Low-Rank Adaptation) on a natural face dataset. The hypothesis is that a model better adapted to the face distribution will produce stronger adversarial perturbations.

Both LoRA versions are loaded **on top of** `Manojb/stable-diffusion-2-1-base` — the base model does not change between runs.

---

### LoRA v1 Training

```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base" \
  --train_data_dir="/app/DiffPrivate/train_faces" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=1e-4 \
  --output_dir="/app/DiffPrivate/fine_tuned_lora" \
  --caption_column="text" \
  --mixed_precision="fp16"
```

---

### LoRA v2 Training (Improved)

LoRA v2 uses a higher rank, longer training, cosine LR schedule, and bf16 precision for better convergence.

```bash
accelerate launch --mixed_precision=bf16 \
  /app/DiffPrivate/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base" \
  --train_data_dir="/app/DiffPrivate/train_faces" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=0 \
  --max_train_steps=5000 \
  --learning_rate=5e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --rank=64 \
  --checkpointing_steps=500 \
  --gradient_checkpointing \
  --allow_tf32 \
  --caption_column="text" \
  --mixed_precision="bf16" \
  --output_dir="/app/DiffPrivate/fine_tuned_lora_v2"
```

**Key differences from v1:**

| Setting | LoRA v1 | LoRA v2 |
|---|---|---|
| Steps | 1,000 | 5,000 |
| Rank | default (4) | 64 |
| Batch size | 1 | 8 |
| LR | 1e-4 | 5e-5 |
| LR schedule | constant | cosine w/ warmup |
| Precision | fp16 | bf16 |

---

### Three-Way Comparison (Baseline vs LoRA v1 vs LoRA v2)

After all three runs complete, use the comparison script:

```bash
python3 compare_attack_logs_v3.py \
  --baseline V2_Kaggle_Run/output_kaggle_baseline_100/attack_log.json \
  --finetuned data/output_finetuned_100/attack_log.json \
  --finetuned2 V2_Kaggle_Run/output_kaggle_finetuned_100_v2/attack_log.json \
  --labels "Baseline" "LoRA v1" "LoRA v2" \
  --output results.csv
```

This produces `results.csv` with per-image identity distances and delta comparisons across all three conditions.

---

### Metrics

We evaluate using identity distance from `attack_log.json`. We **do not** rely on the `protected` field as it is unreliable (often returns all zeros). Instead we use:

- `ID_victim_distance` — primary metric
- Per-model distances: `ir152`, `facenet`, `cur_face`, `mobile_face`
- Δ distance (LoRA − Baseline) per image

---

### Key Results (Kaggle 100-image subset, v1 vs Baseline)

| Metric | Value |
|---|---|
| Avg Δ ID distance | +0.001872 |
| Images improved | 50 / 100 |
| Images worsened | 50 / 100 |
| Best single image gain | +0.144 (681.jpg) |

LoRA v1 showed inconsistent, sample-dependent effects with no clear average gain. LoRA v2 (higher rank, longer training) is expected to improve consistency — results pending.

---

## Reproducing Our Experiments (Grader Instructions)

This section documents everything needed to reproduce our full experimental pipeline from scratch — baseline replication, LoRA v1 fine-tuning, LoRA v2 fine-tuning, and the three-way comparison.

> **Original repo:** https://github.com/minha12/DiffPrivate  
> **Our fork adds:** LoRA integration into `run-dpp.py`, fine-tuned model weights (`fine_tuned_lora/`, `Lora_v2/`), batch experiment scripts, and comparison tooling.

---

### Step 0 — Prerequisites

- Access to ASU SOL HPC (or any machine with an NVIDIA A100 / equivalent 16GB+ GPU)
- Apptainer installed (SOL has this by default)
- HuggingFace account with a read token ([get one here](https://huggingface.co/settings/tokens))

---

### Step 1 — Environment Setup on ASU SOL

```bash
# SSH into SOL, navigate to scratch
cd /scratch/<your_netid>
mkdir CSE469 && cd CSE469

# Clone our repo (contains all scripts + fine-tuned weights)
git clone <your_repo_url> DiffPrivate
cd DiffPrivate

# Allocate a GPU node — do NOT run on login node
salloc -G a100:1

# Pull the container (only needed once)
apptainer pull docker://hale0007/diffprivate:latest
```

---

### Step 2 — Configuration (Required Before Any Run)

Edit `configs/config.yaml` and set the base model path (original repo path is a 404):

```yaml
pretrained_diffusion_path: "Manojb/stable-diffusion-2-1-base"
```

Set LoRA path to `null` for baseline runs:

```yaml
model:
  lora_path: null
```

Authenticate with HuggingFace inside the container shell:

```bash
apptainer shell --nv \
  --bind /scratch/<your_netid>/CSE469/DiffPrivate:/app/DiffPrivate \
  diffprivate_latest.sif

huggingface-cli login   # paste your read token when prompted
```

---

### Step 3 — Run Baseline (No LoRA)

```bash
python run-dpp.py \
  paths.images_root=./data/demo/images \
  paths.save_dir=./data/output_baseline \
  model.lora_path=null \
  attack.targeted_attack=false \
  attack.balance_target=false
```

For the full 100-image Kaggle subset used in our experiments:

```bash
python run-dpp.py \
  paths.images_root=./data/kaggle_subset_100 \
  paths.save_dir=./V2_Kaggle_Run/output_kaggle_baseline_100 \
  model.lora_path=null \
  attack.targeted_attack=false \
  attack.balance_target=false
```

---

### Step 4 — Run with LoRA v1 Weights

Our pre-trained LoRA v1 weights are included in `fine_tuned_lora/`.  
No retraining needed — just point `lora_path` at the directory:

```bash
python run-dpp.py \
  paths.images_root=./data/kaggle_subset_100 \
  paths.save_dir=./data/output_finetuned_100 \
  model.lora_path=./fine_tuned_lora \
  attack.targeted_attack=false \
  attack.balance_target=false
```

---

### Step 5 — Run with LoRA v2 Weights

Our pre-trained LoRA v2 weights are included in `Lora_v2/`.

```bash
python run-dpp.py \
  paths.images_root=./data/kaggle_subset_100 \
  paths.save_dir=./V2_Kaggle_Run/output_kaggle_finetuned_100_v2 \
  model.lora_path=./Lora_v2 \
  attack.targeted_attack=false \
  attack.balance_target=false
```

---

### Step 6 — Run All Three in One Shot (Recommended)

To reproduce the full overnight comparison in a single SLURM job:

```bash
# Submit the batch job
sbatch run_celeba_compare_50.sbatch

# Monitor progress
tail -f slurm-<job_id>.out

# Or run interactively on a GPU node
chmod +x overnight_compare.sh
nohup ./overnight_compare.sh > overnight_master.log 2>&1 &
tail -f overnight_master.log
```

---

### Step 7 — Generate the Three-Way Comparison

After all three runs complete, generate `results.csv`:

```bash
python3 compare_attack_logs_v3.py \
  --baseline V2_Kaggle_Run/output_kaggle_baseline_100/attack_log.json \
  --finetuned data/output_finetuned_100/attack_log.json \
  --finetuned2 V2_Kaggle_Run/output_kaggle_finetuned_100_v2/attack_log.json \
  --labels "Baseline" "LoRA v1" "LoRA v2" \
  --output results.csv
```

`results.csv` contains per-image `ID_victim_distance` and Δ values for all three conditions.

---

### Step 8 — (Optional) Retrain LoRA from Scratch

You do **not** need to retrain — weights are provided. But if you want to reproduce training:

**LoRA v1:**
```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base" \
  --train_data_dir="/app/DiffPrivate/train_faces" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=1e-4 \
  --output_dir="/app/DiffPrivate/fine_tuned_lora" \
  --caption_column="text" \
  --mixed_precision="fp16"
```

**LoRA v2:**
```bash
accelerate launch --mixed_precision=bf16 \
  /app/DiffPrivate/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base" \
  --train_data_dir="/app/DiffPrivate/train_faces" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=0 \
  --max_train_steps=5000 \
  --learning_rate=5e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --rank=64 \
  --checkpointing_steps=500 \
  --gradient_checkpointing \
  --allow_tf32 \
  --caption_column="text" \
  --mixed_precision="bf16" \
  --output_dir="/app/DiffPrivate/fine_tuned_lora_v2"
```

> Training dataset: [Kaggle Natural Face Dataset](https://www.kaggle.com/datasets/cybersimar08/face-recognition-dataset) (~1500 images, cleaned and filtered). Place in `train_faces/` before running.

---

### Repository Structure

```
DiffPrivate/
├── configs/                  # Hydra config (model paths, attack params)
├── data/                     # Input subsets + output attack logs
│   ├── kaggle_subset_100/    # 100-image Kaggle evaluation subset
│   └── celeba_subset_50/     # 50-image CelebA evaluation subset
├── fine_tuned_lora/          # LoRA v1 weights (pre-trained, ready to use)
├── Lora_v2/                  # LoRA v2 weights (pre-trained, ready to use)
├── V2_Kaggle_Run/            # Output logs for v2 Kaggle experiments
├── run-dpp.py                # Main attack script (modified to support LoRA)
├── compare_attack_logs_v3.py # Three-way metric comparison tool
├── overnight_compare.sh      # Runs all three conditions sequentially
├── run_celeba_compare_50.sbatch  # SLURM job script
├── celeba_baseline_50.log    # CelebA baseline output log
├── celeba_finetuned_50.log   # CelebA LoRA v1 output log
└── overnight_master.log      # Full overnight run log
```

---

### Notes for Graders

- **No retraining required** — all LoRA weights are included in the repo.
- **Single command to reproduce results:** `sbatch run_celeba_compare_50.sbatch` (on SOL) or `./overnight_compare.sh` (interactive GPU node).
- The `protected` field in attack logs is **unreliable** (often all zeros); we use `ID_victim_distance` as the primary metric throughout.
- Base model `stabilityai/stable-diffusion-2-base` from the original repo returns a 404 — we use `Manojb/stable-diffusion-2-1-base` as a drop-in replacement.


---

## Running Locally (Windows + Consumer GPU)

This section explains how to run DiffPrivate on a local Windows machine with a consumer NVIDIA GPU (tested on RTX 4060 Laptop 8GB).

> **Note:** The original project requires 16GB+ VRAM. We made adjustments to fit on 8GB by reducing resolution and diffusion steps. Output quality is slightly lower but the privacy protection still works.

---

### Requirements

- Windows 10/11
- NVIDIA GPU (8GB+ VRAM recommended)
- [Anaconda](https://www.anaconda.com/download) installed
- Git installed

---

### Step 1 — Clone the Repo

```bash
git clone https://github.com/DvirHamu/ASUCSE467ResearchAssignment.git
cd ASUCSE467ResearchAssignment/DiffPrivate
```

---

### Step 2 — Create the Conda Environment

Open **Anaconda Prompt** and run:

```bash
cd "path\to\ASUCSE467ResearchAssignment\DiffPrivate"
conda env create -f environment.yml
conda activate diffprivate
pip install accelerate
```

---

### Step 3 — Prepare Demo Images

```bash
cd data
tar -xf demo.zip
cd ..
```

The images will extract to `data/images/`. The config is already set to point there.

---

### Step 4 — Configuration

The `configs/config.yaml` has been pre-configured for local use:

- `images_root` → `./data/images`
- `lora_path` → `./fine_tuned_lora` (LoRA v2 weights)
- `res` → `128` (reduced from 256 to fit in 8GB VRAM)
- `diffusion_steps` → `10` (reduced from 20)

No changes needed — just run.

---

### Step 5 — Run

```bash
python run-dpp.py
```

The first run will automatically download:
- Stable Diffusion model weights (~5GB, cached to `~/.cache/huggingface`)
- Face recognition model weights (~800MB total, saved to `model-weights/`)

Subsequent runs will be much faster since everything is cached.

---

### Step 6 — View Results

Output images are saved to `data/output/`. Each processed image produces:

- `<name>_adv_image.png` — the privacy-protected image
- `<name>_diff_image_ATKSuccess.png` — side-by-side original vs protected
- `<name>_diff_absolute.png` — absolute pixel difference
- `attack_log.json` — identity distances and protection metrics

---

### Troubleshooting

**CUDA out of memory** — reduce resolution further in `configs/config.yaml`:
```yaml
diffusion:
  res: 96
  diffusion_steps: 8
  start_step: 5
```

**Black output images** — caused by fp16 NaN values. Make sure the model is loaded in fp32 (default in `run-dpp.py` — do not add `torch_dtype=torch.float16`).

**`No module named 'src'`** — make sure you cloned this repo (not the original), as the `src/` folder is included here.

**`Primary config directory not found`** — make sure the config folder is named `configs` (not `config`). This repo has it correctly named.


---

## Running the Web App

We built a simple web interface on top of DiffPrivate so anyone can protect their face images without touching the command line. It runs locally on your machine and is accessible from any device on the same WiFi network (phone, tablet, laptop).

---

### How It Works

1. A **Flask server** (`app.py`) loads the Stable Diffusion model + LoRA weights once at startup
2. The server exposes a `/protect` endpoint that accepts an image upload
3. When an image is submitted, it runs the full DiffPrivate pipeline on it and returns the privacy-protected result
4. The **web frontend** (`templates/index.html`) provides a clean UI — upload, preview, process, download — all in the browser

---

### Setup (one time)

Make sure you're in the `diffprivate` conda environment:

```bash
conda activate diffprivate
cd "C:\Users\along\OneDrive\Documents\CSE 467\ASUCSE467ResearchAssignment\DiffPrivate"
pip install flask
```

---

### Running the Server

```bash
python app.py
```

The model loads once at startup (takes ~30 seconds). You'll see `Model ready.` when it's done.

---

### Using the App

1. Open your browser and go to `http://localhost:5000`
2. Drag & drop a face photo onto the upload area (or click to browse)
3. Click **Protect Image**
4. Wait a few minutes while DiffPrivate processes the image
5. The protected image appears on screen — click **Download Protected Image** to save it

That's it. No command line, no config files, no Python knowledge needed.

---

### Access From Other Devices (Same WiFi)

Find your local IP:
```bash
ipconfig
```
Look for `IPv4 Address` (e.g. `192.168.1.42`). Then on any device on the same network, open:
```
http://192.168.1.42:5000
```
