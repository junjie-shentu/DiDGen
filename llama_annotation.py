from PIL import Image
import requests
import torch
from pathlib import Path
from transformers import MllamaForConditionalGeneration, AutoProcessor
import json
from tqdm import tqdm
import argparse


class Skindataset(torch.utils.data.Dataset):
    def __init__(self, image_path, max_num = None):


        self.image_path = image_path
        self.max_num = max_num

        self.instance = [x for x in Path(self.image_path).iterdir() if x.suffix in ['.jpg', '.png']]
        self.num_instance_images = len(list(self.instance))
        if max_num is not None and max_num < len(self.instance):
            self.instance = self.instance[:max_num]
            self._length = max_num
        else:
            self._length = len(self.instance)


    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        example = {}
        instance_image_path = self.instance[idx % self._length]
        instance_image = Image.open(instance_image_path).convert("RGB")

        example["name"] = instance_image_path.name
        example["image"] = instance_image


        return example
    
def collate_fn(batch):
    names = [x["name"] for x in batch]
    images = [x["image"] for x in batch]


    return {
        "names": names,
        "images": images,
    }





# Prepare a batch of two prompts
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "This is an dermoscopic image of skin lesion, describe the shape, size, color of the lesion, the position of the lesion in the image, the color of the skin, and other element on the image if there is any. "},
        ],
    },
]



def image_annotation(dataset, batch_size, max_tokens=150, model=None, processor=None):

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    annotations = {}

    # Process the data for one epoch
    for batch in tqdm(dataloader):
        images = batch["images"]
        names = batch["names"]
        prompts = [prompt] * len(images)
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for name, output in zip(names, outputs):
            annotations[name] = output



    return annotations


def main():
    
    parser = argparse.ArgumentParser(description="Generate annotations for skin lesion images using Llama 3.2 Vision")
    
    parser.add_argument("--input_dir", type=str, default="directory/of/images",
                        help="Path to directory containing input images")
    parser.add_argument("--output_file", type=str, default="annotations.json",
                        help="Output file path for annotations")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing images")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        help="Model name to use")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max images: {args.max_images if args.max_images else 'All'}")
    print(f"Max tokens: {args.max_tokens}")

    model = MllamaForConditionalGeneration.from_pretrained(args.model_name, 
                                                            torch_dtype=torch.bfloat16, 
                                                            device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_name)

    print(f"Model device: {model.device}")
    
    # Create dataset
    dataset = Skindataset(args.input_dir, max_num=args.max_images)
    print(f"Found {len(dataset)} images to process")
    
    # Generate annotations
    annotations = image_annotation(dataset=dataset, batch_size=args.batch_size, max_tokens=args.max_tokens, model=model, processor=processor)
    
    # Save annotations
    print(f"Saving annotations to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(annotations, f, indent=4)
    
    print(f"Successfully processed {len(annotations)} images!")


if __name__ == "__main__":
    main()