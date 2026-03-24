Old one

accelerate launch train_text_to_image_lora.py --pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base" --train_data_dir="/app/DiffPrivate/train_faces" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=1000 --learning_rate=1e-4 --output_dir="/app/DiffPrivate/fine_tuned_lora" --caption_column="text" --mixed_precision="fp16"

new one
accelerate launch --mixed_precision=bf16 /app/DiffPrivate/diffusers/examples/text_to_image/train_text_to_image_lora.py - pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base' --train_data_dir='/app/DiffPrivate/train_faces' --resolution=512 --center_crop --random_flip --train_batch_size=8 --gradient_accumulation_steps=4 --dataloader_num_workers=0 --max_train_steps=5000 --learning_rate=5e-5 --lr_scheduler='cosine' --lr_warmup_steps=500 --rank=64 --checkpointing_steps=500 --gradient_checkpointing --allow_tf32 --caption_column='text' --mixed_precision='bf16' --output_dir='/app/DiffPrivate/fine_tuned_lora_v2


located in google drive actual file