#!/bin/bash

set -e




echo "Starting baseline run..."

python run-dpp.py paths.images_root=./data/kaggle_subset_100 paths.save_dir=./data/output_baseline_100 model.lora_path=null | tee baseline_run.log



echo "Baseline finished."



echo "Starting fine-tuned LoRA run..."

python run-dpp.py  paths.images_root=./data/kaggle_subset_100 paths.save_dir=./data/output_finetuned_100 model.lora_path=./fine_tuned_lora | tee finetuned_run.log



echo "Fine-tuned run finished."
