import warnings
warnings.filterwarnings("ignore")

import re
import gradio as gr
from text_inference import text_inference, MODEL_LABEL
from image_gen import caption_image
from PIL import Image
from websearching import web_search, date_for_debug

def format_sources(sources):
    # The model cites sources by number; list the actual links under the answer
    if not sources:
        return ""
    return "\n\n**Sources**\n\n" + "\n".join(f"{i + 1}. [{url}]({url})" for i, url in enumerate(sources))

# Two or more capitalized words in a row, e.g. "Dylan O'Brien" or "Chase Stokes"
NAME = re.compile(r"\b[A-Z][\w'’-]+(?:[ \t]+[A-Z][\w'’-]+)+")
LEADING_WORDS = {"The", "A", "An", "In", "On", "At", "Yes", "No", "According", "Based", "However", "While", "When"}

def unverified_names(answer, prompt):
    # The model sometimes adds names from its own (often wrong) memory; flag any name the sources never mention
    sources_text = prompt.lower()
    flagged = []
    for match in NAME.findall(answer):
        words = match.split()
        while words and words[0] in LEADING_WORDS:
            words = words[1:]
        name = " ".join(words).rstrip("'’")
        if len(words) >= 2 and name.lower() not in sources_text and name not in flagged:
            flagged.append(name)
    if not flagged:
        return ""
    return "\n\n> ⚠️ **Not found in the sources, may be inaccurate:** " + ", ".join(flagged)

def reply(text_input, image_input=None, max_results=5, enable_rag=False, debug=True):
    if debug:
        print(f"[{date_for_debug()}] Started query processing...")
    if image_input is None:
        prompt, qdrant_success, sources = web_search(text_input, max_results, enable_rag, debug)
        if debug:
            print(qdrant_success)
        results = text_inference(prompt, debug, query=text_input)
        results = results.replace("<|im_end|>", "")
        if debug:
            print(f"[{date_for_debug()}] Finished query processing!")
        return results + unverified_names(results, prompt) + format_sources(sources)
    else:
        img = Image.fromarray(image_input)
        caption = caption_image(img)
        # Put the user's question first so the search engine treats it as the main intent
        full_query = f"{text_input.strip().rstrip('.?!')}. {caption}" if text_input else caption
        if full_query:
            prompt, qdrant_success, sources = web_search(full_query, max_results, enable_rag, debug)
            if debug:
                print(qdrant_success)
            results = text_inference(prompt, debug, query=full_query)
            results = results.replace("<|im_end|>", "")
            if debug:
                print(f"[{date_for_debug()}] Finished query processing!")
            return results + unverified_names(results, prompt) + format_sources(sources)
    return "Please enter a search query or upload an image."

iface = gr.Interface(fn=reply, inputs=[gr.Textbox(value="", label="Search Query"), gr.Image(value=None, label="Image Search Query"), gr.Slider(1,10,value=5,label="Maximum Number of Search Results", step=1), gr.Checkbox(value=True, label="Enable Rag"), gr.Checkbox(value=True, label="Debug")], outputs=[gr.Markdown(value="Your output will be generated here", label = "Search Results")], title="IncognitAI", description=f"Answers written by {MODEL_LABEL}, running on this computer. Change the model with LLM_SIZE in .env.\n\n⚠️ **IncognitAI can make mistakes.** Check important facts against the linked sources.")

iface.launch(server_name="0.0.0.0", server_port=7860)
