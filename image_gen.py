import warnings
warnings.filterwarnings("ignore")

import nltk
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from rake_nltk import Metric, Rake

# rake-nltk needs these NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Florence-2 is natively supported in transformers 5; the microsoft/ remote-code version is not compatible
model = Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-large", dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")

task_prompt = "<DETAILED_CAPTION>"
raker = Rake(include_repeated_phrases=False, ranking_metric=Metric.WORD_DEGREE)

def extract_keywords_from_caption(caption: str) -> str:
    raker.extract_keywords_from_text(caption)
    keywords = raker.get_ranked_phrases()[:5]
    fnl = []
    for keyword in keywords:
        if "image" in keyword:
            continue
        else:
            fnl.append(keyword)
    return " ".join(fnl)

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

    caption = parsed_answer["<DETAILED_CAPTION>"]
    search_words = extract_keywords_from_caption(caption)
    return search_words