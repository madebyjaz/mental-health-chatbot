#  🌟💝 Empathetic Mental Health Support Chatbot  
> _"Enhancing emotional well-being through empathetic AI conversations."_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗_Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)

A lightweight, classifier-augmented generative chatbot designed to provide **empathetic, emotionally aware, and safe mental-health–oriented conversations**. This project combines **NLP classifiers**, a **dialogue management system**, and a **DialoGPT-based response generator**, deployed through a simple **Streamlit interface**.

This project leverages **Natural Language Processing** and **Affective Computing** to provide emotionally intelligent, context-aware mental-health assistance. It recognizes user emotions, detects risk, understands intent, and responds with empathy — all while maintaining safety via a multi-stage crisis-detection pipeline.

🩵 The chatbot is **not a replacement for therapy**, but a digital companion that enhances accessibility and supports emotional awareness.

The goal is to **augment** human support for indivuals financially incapable of getting support or those who need immediate support.

---

## ⏮️ Background
Empathetic dialogue is still hard for standard chatbots — they’re good at *what to say*, weaker at *how to say it*. This project draws on three key papers:

- Rashkin et al. 2019 — **EMPATHETICDIALOGUES** benchmark for open-domain empathetic responses.  
- Tu et al. 2022 — **MISC**, which mixes strategies + commonsense (COMET) for emotional support.  
- Zhou et al. 2018 — **Emotional Chatting Machine (ECM)**, which adds internal/external emotion memory for more human-like responses.  

These papers show that explicitly modeling **emotion + strategy** improves empathy and user satisfaction.

---

# Project Overview
The system is composed of **three core components**:

## 1. **NLP Classifier Pipeline**
These models analyze the user’s input and provide signals that shape the chatbot’s tone, strategy, and safety.

### **Emotion Classifier (GoEmotions / Electra-Base)**
- Model: `google/electra-base-goemotions`
- Detects **27 fine-grained emotions** (sadness, joy, fear, disappointment, confusion, etc.)
- Output informs **empathy level**, tone, and conversational style

### **Intent Classifier (Zero-Shot MNLI Model)**
- Model: `facebook/bart-large-mnli`
- No fine-tuning required (zero-shot)
- Detects intents such as:
  - Seeking emotional support
  - Venting
  - Asking for advice
  - Requesting information
  - Self-reflection

### **Risk / Safety Classifier (Dreaddit Stress Detection)**
- Model: `microsoft/deberta-v3-large`
- Dataset: `asmaab/dreaddit` (stress + crisis signals)
- Identifies dangerous or crisis-related language and **triggers override safety responses**

These classifiers **do not generate text**. They strictly guide the behavior of the dialogue system.

---

## 2. **Dialogue Management + Generative Response Module**
This is the heart of the chatbot ☺️💜

### **Generative Model: DialoGPT-small**
- Model: `microsoft/DialoGPT-small`
- Used to generate **empathetic, context-aware responses**
- Behaves like a conversational agent rather than a plain classifier pipeline

### How It Works
For every user message:
1. Emotion, intent, and risk classifiers analyze the text.
2. The **Dialogue Manager** decides a conversational strategy:
   - Validation
   - Reflective listening
   - Supportive reassurance
   - Information sharing
   - OR: **Crisis override** (if risk high)
3. A structured prompt is constructed using:
   - Detected emotions
   - Inferred intent
   - Safety risk level
   - Chosen strategy
4. DialoGPT generates a response following that prompt.

This creates a chatbot that is **empathetic**, **situationally aware**, and **safe**.

---

## 3. **Web Interface (Streamlit)**
A simple, fast, lightweight UI using Streamlit.

### 🧩 Features:
- [ ] 💬 Real-time chat window
- [ ] 📈 Model-generated responses
- [ ] 🗣 Emotion classification displayed per message
- [ ] 💻 Local logging for analysis
- [ ] 👩🏽‍💻 Easy to deploy

### 🧩 Future Feature(s)
- [ ] 📈 **Dashboard Visualization** (emotion/sentiment/risk trends)  

Run with:
```
streamlit run src/app/app.py
```

---

# 👩🏽‍💻📁 Project Structure
```
src/
├── classifiers/
│   ├── training.py
│   
│   
│   
│
├── dialogue/
│   ├── dialogue_manager.py
│   
│
├── generation/
│   
│
├── app/
│   ├── app.py
│
├── utils/
│   
│   
│___
```

Models are stored under:
```
models/
  ├── emotion/
  ├── risk/
```
(Intent classifier is zero-shot and needs no training.)

---

# 🧪 Training the Classifiers
Only **emotion** and **risk** classifiers are trainable.

### Train emotion classifier
```
python src/classifiers/training.py --task emotion
```

### Train risk classifier
```
python src/classifiers/training.py --task risk
```

Intent classifier requires **no training**.

---

# 🚨 Safety By Design
The chatbot includes:
- High-risk detection
- Immediate crisis-resource override
- Blocking of unsafe generative outputs

If risk classifier detects danger:
```
"I'm really sorry you're feeling this way. You deserve immediate support…"
```
No generative output is used.

---

# 🎯 Goals of the Project
- Build a functional mental-health conversational agent
- Integrate NLP classifiers to guide emotional and semantic behavior
- Maintain safe and ethical communication
- Provide transparent and reproducible research
- Create a clean and simple UI suitable for demonstrations

---

# 📌 Status
✔ Classifier architecture finalized  
⬜ Dialogue Manager implemented  
⬜ Generation system implemented  
⬜ UI created  
⬜ (Optional) Fine-tune emotion + risk locally  
⬜ Add visualization of emotion trends (if time allows)

## 📚 References 
- Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2019). <em>  Towards empathetic open-domain conversation models: A new benchmark and dataset.</em>  ACL.
- Tu, Y., Meng, Z., Huang, M., & Zhu, X. (2022). <em> MISC: A mixed strategy-aware model integrating COMET for emotional support conversation. </em> ACL Findings.
- Zhou, H., Huang, M., Zhang, T., Zhu, X., & Liu, B. (2018). <em> Emotional chatting machine: Emotional conversation generation with internal and external memory. </em> AAAI.

## ‼️ Disclaimer ‼️
- This is a research prototype intended for educational use only. It should not be used as a diagnostic or therapeutic tool.
<br/>

<b>If you or someone you know is in a crisis, please contact <em><u>988 (United States)</u></em> or your local emergency services. </b>