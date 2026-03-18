#!/bin/bash



set -e

echo "PWD=$(pwd)"

echo "Starting CelebA baseline..."

python run-dpp.py paths.images_root=./data/celeba_subset_50 paths.save_dir=./data/output_celeba_baseline_50 model.lora_path=null attack.targeted_attack=false attack.balance_target=false | tee celeba_baseline_50.log

echo "Baseline finished."

echo "Starting CelebA fine-tuned..."

python run-dpp.py paths.images_root=./data/celeba_subset_50 paths.save_dir=./data/output_celeba_finetuned_50 model.lora_path=./fine_tuned_lora attack.targeted_attack=false attack.balance_target=false | tee celeba_finetuned_50.log

echo "Fine-tuned finished."
