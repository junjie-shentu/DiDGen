accelerate launch finetune_SD_attention.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --output_dir="checkpoint/SD_finetune" \
  --max_train_step=20000 \
  --checkpointing_steps=10000 \
  --report_to="wandb" \