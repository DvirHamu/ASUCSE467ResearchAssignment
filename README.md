# ASUCSE467ResearchAssignment
todo
Read Me

Go to scratch directory and open terminal
mkdir CSE469
cd CSE469

git clone https://github.com/minha12/DiffPrivate

We’ll be using apptainer instead of docker to run this


Getting resources to ASU SOL

When you first log in you will be on the login node, the compute resources allocated to this is low. DO not run these commands on the login node. Or else you’ll get this.


‘’’
No need for now use gpu
This is for cpu
srun --partition=public --mem=64G --cpus-per-task=8 --time=2:00:00 --pty bash
‘’’

This is for gpu
salloc -G a100:1

Run apptainer on the repo #make sure your on the gpu node!!!

apptainer pull docker://hale0007/diffprivate:latest 

Before you start the shell there is some configuration you need to do
nano configs/config.yaml
Change the path to
pretrained_diffusion_path: "Manojb/stable-diffusion-2-1-base"
Previous one was 404

Then to get a shell you run

apptainer shell --nv --bind /scratch/dhamu/CSE469/DiffPrivate:/app/DiffPrivate diffprivate_latest.sif


Open a new tab and log into your hugging face
https://huggingface.co/settings/tokens

Make a read token and save the key


Back in the shell run the command 
huggingface-cli login

Paste your token, then Y 


Token should look like this
hf_ld”...”

python run-dpp.py paths.images_root=./data/demo/images paths.save_dir=./data/output

Then once done 
Go back to ASU SOL gui

And you can download onto your computer

