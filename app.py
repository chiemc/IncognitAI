import warnings
warnings.filterwarnings("ignore")

import re
import gradio as gr
from text_inference import text_inference, rewrite_query, MODEL_LABEL
from image_gen import caption_image
from PIL import Image
from websearching import web_search, date_for_debug

def format_sources(sources):
    # The model cites sources by number; list the actual links under the answer
    if not sources:
        return ""
    return "\n\n**Sources**\n\n" + "\n".join(f"{i + 1}. [{url}]({url})" for i, url in enumerate(sources))

def format_search(question, search_query):
    # When a follow-up was rewritten, show what was actually searched so a wrong guess is easy to spot
    if search_query == question:
        return ""
    return f"*🔎 Searched for: {search_query}*\n\n"

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

def reply(text_input, image_input=None, max_results=5, debug=True):
    if debug:
        print(f"[{date_for_debug()}] Started query processing...")
    if image_input is None:
        # Turn follow-ups like "What other movies has he made?" into a standalone search.
        # Image searches skip this: there, "this" or "it" usually means the image, not the conversation.
        search_query = rewrite_query(text_input, debug)
        prompt, qdrant_success, sources = web_search(search_query, max_results, debug)
        if debug:
            print(qdrant_success)
        # History keeps the question as the user typed it
        results = text_inference(prompt, debug, query=text_input)
        results = results.replace("<|im_end|>", "")
        if debug:
            print(f"[{date_for_debug()}] Finished query processing!")
        return format_search(text_input, search_query) + results + unverified_names(results, prompt) + format_sources(sources)
    else:
        img = Image.fromarray(image_input)
        caption = caption_image(img)
        # Put the user's question first so the search engine treats it as the main intent
        full_query = f"{text_input.strip().rstrip('.?!')}. {caption}" if text_input else caption
        if full_query:
            prompt, qdrant_success, sources = web_search(full_query, max_results, debug)
            if debug:
                print(qdrant_success)
            results = text_inference(prompt, debug, query=full_query)
            results = results.replace("<|im_end|>", "")
            if debug:
                print(f"[{date_for_debug()}] Finished query processing!")
            return results + unverified_names(results, prompt) + format_sources(sources)
    return "Please enter a search query or upload an image."

iface = gr.Interface(
    fn=reply,
    inputs=[
        gr.Textbox(value="", label="Search Query"),
        gr.Image(value=None, label="Image Search Query"),
        gr.Slider(
            1, 10, value=5, step=1,
            label="Maximum Number of Search Results",
            info="How many web pages to read for each question. Some sites block automated visitors, so usually "
                 "a few fewer pages are read than this number. More pages give a better chance of finding the answer "
                 "but take longer, about 1 second per page. Either way, the answer is written from the 5 passages most "
                 "relevant to your question, picked from all the pages read (RAG, always on). 5 suits most questions. Use 7–8 if an answer seems thin, or 3 for quick facts.",
        ),
        gr.Checkbox(
            value=True,
            label="Debug",
            info="Prints each step (search results, blocked pages, the exact prompt given to the model) in the "
                 "terminal running the app. Useful for troubleshooting; it doesn't change the answer.",
        ),
    ],
    outputs=[gr.Markdown(value="Your output will be generated here", label = "Search Results")],
    title="IncognitAI",
    description=f"Answers written by {MODEL_LABEL}, running on this computer. Change the model with LLM_SIZE in .env.\n\n⚠️ **IncognitAI can make mistakes.** Check important facts against the linked sources.",
    # The Flag button saves the question, settings and answer to .gradio/flagged/ for reviewing bad answers later
    flagging_mode="manual",
)

iface.launch(server_name="0.0.0.0", server_port=7860)
