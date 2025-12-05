import sys
import os
import streamlit as st
# Ensure the `src` directory is on the Python path when running via Streamlit
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classifiers.emotion_classifier import EmotionModel
from classifiers.intent_classifier import IntentModel
from classifiers.risk_classifier import RiskModel
from dialogue.dialogue_manager import DialogueManager
from generation.nlg_generator import NLGGenerator


st.title("Mental Health Support Chat 👾")
st.write("This is a demo of the mental health support chatbot that's designed to augment empathetic human support, not replace it.✨✨")

emotion_model = EmotionModel()
intent_model = IntentModel()
risk_model = RiskModel()
# risk_model = None  # Placeholder or handle appropriately if RiskModel is unavailable

dialogue_manager = DialogueManager(emotion_model, intent_model, risk_model)

# Persist chat history across reruns
if "history" not in st.session_state:
    st.session_state["history"] = []
dialogue_manager.history = st.session_state["history"]
generator = NLGGenerator()

user_input = st.chat_input("How are you feeling today?")

# Show prior chat history (persisted)
for msg in st.session_state["history"][-20:]:
    if msg.get("user"):
        st.chat_message("user").write(msg["user"]) 
    if msg.get("bot"):
        st.chat_message("assistant").write(msg["bot"]) 

# Handle new user input and generate response
if user_input:
    classification = dialogue_manager.classify(user_input)

    risk_value = classification.get("risk", "")
    # Normalize risk label if it's a dict / structured
    if isinstance(risk_value, dict) and "label" in risk_value:
        risk_label = risk_value["label"]
    else:
        risk_label = str(risk_value)

    if "crisis" in risk_label.lower():
        # Let the model generate, but steer it with a crisis-support strategy
        prompt, strategy = dialogue_manager.build_prompt(
            user_input,
            classification,
            override_strategy="crisis_support",
        )
        response = generator.generate(prompt)
    else:
        prompt, strategy = dialogue_manager.build_prompt(user_input, classification)
        response = generator.generate(prompt)

    st.chat_message("assistant").write(response)

    # Update history
    entry = {"user": user_input, "bot": response}
    st.session_state["history"].append(entry)
    dialogue_manager.history = st.session_state["history"]


