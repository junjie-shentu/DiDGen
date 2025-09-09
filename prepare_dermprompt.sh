python llama_annotation.py \
    --input_dir ./data/dermprompt/images \
    --output_file ./data/dermprompt/annotations.json \
    --batch_size 4 \
    --model_name meta-llama/Llama-3.2-11B-Vision-Instruct

python llama_rephase.py \
    --input ./data/dermprompt/annotations.json \
    --output ./data/dermprompt/annotations_processed.json \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --max_tokens 256 \
    --target_tokens 77