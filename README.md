# ASUCSE467ResearchAssignment

<img width="1918" height="834" alt="image6" src="https://github.com/user-attachments/assets/0862ee78-8853-4798-8578-68d4c33f19f4" />

Go to scratch directory and open terminal

```
mkdir CSE469
cd CSE469
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
import torch
from safetensors.torch import load_file, save_file

# Load the new format
weights = load_file('/app/DiffPrivate/fine_tuned_lora/pytorch_lora_weights.safetensors')

# Print keys to see what format they're in
for k in list(weights.keys())[:5]:
    print(k)
"```
