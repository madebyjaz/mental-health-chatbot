import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st


# -----------------------------------------------------------
# Utility: Detect low-quality or nonsensical generations
# -----------------------------------------------------------
def looks_gibberish(text: str) -> bool:
    """
    Very gentle gibberish detector:
    - Accepts most normal sentences
    - Only flags text that is empty, extremely short, or clearly nonsense
    """
    if not text or len(text.strip()) == 0:
        return True

    words = text.strip().split()
    # If we got at least 6 words, assume it's usable
    if len(words) >= 6:
        return False

    # Only reject if there are obvious nonsense patterns
    bad_patterns = [
        r"[bcdfghjklmnpqrstvwxyz]{5,}",  # long consonant junk
        r"[^\w\s,.!?]",                  # weird symbols
    ]
    return any(re.search(p, text) for p in bad_patterns)


# -----------------------------------------------------------
# Cached model loaders so Streamlit does not reload repeatedly
# -----------------------------------------------------------
@st.cache_resource(show_spinner=True)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", use_fast=False)


@st.cache_resource(show_spinner=True)
def _get_model():
    return AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")


# -----------------------------------------------------------
# Main NLG Class
# -----------------------------------------------------------
class NLGGenerator:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self.tokenizer = _get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = _get_model()

    # -----------------------------------------------------------
    # Core generation function
    # -----------------------------------------------------------
    def generate(self, prompt: str) -> str:
        device = torch.device("cpu")

        # Ensure clear turn-taking cue
        if not prompt.rstrip().endswith("Bot:"):
            prompt = prompt.rstrip() + "\nBot:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        gen_settings = dict(
            do_sample=True,
            top_p=0.92,
            temperature=0.65,
            max_new_tokens=120,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_settings)

        # Extract generated continuation only
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = output[0][prompt_len:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        response = self.clean_response(decoded)

        # ---------------------------------------------------
        # Fallback retry if text is too short or low-quality
        # ---------------------------------------------------
        if looks_gibberish(response) or len(response.split()) < 5:
            with torch.inference_mode():
                retry = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                    max_new_tokens=150,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                )
            gen_ids = retry[0][prompt_len:]
            retry_decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            response = self.clean_response(retry_decoded)

        # Final fallback: guaranteed empathetic message
        if looks_gibberish(response) or len(response.strip()) < 3:
            response = (
                "I’m really hearing the weight of what you’re sharing. "
                "You’re not alone — if you’d like, tell me a bit more about what’s been hardest lately."
            )

        return response

    # -----------------------------------------------------------
    # Cleanup & formatting logic
    # -----------------------------------------------------------
    def clean_response(self, text: str) -> str:
        # Remove any echoed role markers
        text = text.replace("User:", "").replace("Bot:", "")

        # Squash repeated spaces
        text = re.sub(r"\s{2,}", " ", text)
        text = text.strip()

        # If the model gave a very bland acknowledgment, soften it
        lower = text.lower()
        if lower in ["ok", "okay", "yes", "i see", "sure", "hmm"]:
            text = (
                "Thank you for opening up about that. "
                "It sounds really heavy — what part of this feels the most overwhelming right now?"
            )

        # Ensure it ends cleanly
        if text and not text.endswith((".", "!", "?")):
            text += "."

        return text
