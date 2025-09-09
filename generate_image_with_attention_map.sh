python inference_with_attention.py \
    --ckpt_dir ./checkpoint/SD_finetune \
    --image_save_path ./generated_image \
    --validation_prompt "An image of <lesion> on <skin>" \
    --seed_range 20