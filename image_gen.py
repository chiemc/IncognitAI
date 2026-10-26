import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Florence-2 is natively supported in transformers 5; the microsoft/ remote-code version is not compatible
model = Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-large", dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")

task_prompt = "<DETAILED_CAPTION>"

def caption_image(image):
    global task_prompt
    prompt = task_prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    # Return the whole caption: RAKE keywords dropped important words like "Bandwidth's" and added noise
    return parsed_answer["<DETAILED_CAPTION>"].strip()