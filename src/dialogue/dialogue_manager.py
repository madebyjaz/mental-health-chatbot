from typing import Dict, Tuple, List, Any


class DialogueManager:
    def __init__(self, emotion_model, intent_model, risk_model):
        self.emotion_model = emotion_model
        self.intent_model = intent_model
        self.risk_model = risk_model
        self.history: List[Dict[str, str]] = []

    def _label_from(self, value: Any) -> str:
        """
        Try to normalize classifier outputs into a simple label string.
        Handles:
          - plain strings
          - {'label': 'sadness', 'score': 0.8}
          - [{'label': 'sadness', 'score': 0.8}, ...]
        """
        if value is None:
            return "unknown"
        if isinstance(value, str):
            return value

        # huggingface-style dict
        if isinstance(value, dict):
            if "label" in value:
                return str(value["label"])
            # fallback
            return str(value)

        # list / tuple case
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, dict) and "label" in first:
                return str(first["label"])
            return str(first)

        return str(value)

    def classify(self, user_text: str) -> Dict[str, Any]:
        emotion = getattr(self.emotion_model, "predict")(user_text) if self.emotion_model else "neutral"
        intent = getattr(self.intent_model, "predict")(user_text) if self.intent_model else "unknown"
        risk = getattr(self.risk_model, "predict")(user_text) if self.risk_model else "low"
        return {"emotion": emotion, "intent": intent, "risk": risk}

    def _choose_strategy(self, emotion_label: str, intent_label: str, risk_label: str, override: str = None) -> str:
        if override:
            return override

        e = (emotion_label or "").lower()
        i = (intent_label or "").lower()
        r = (risk_label or "").lower()

        if "crisis" in r or "high" in r or "self-harm" in r:
            return "crisis_support"

        if any(k in e for k in ["sad", "depress", "lonely", "grief", "hopeless"]):
            if any(k in i for k in ["vent", "share", "talk"]):
                return "reflection"
            return "validation"

        if any(k in i for k in ["ask", "question", "info", "information"]):
            return "informational_support"

        if any(k in e for k in ["anxious", "worried", "stressed", "afraid"]):
            return "reassurance"

        return "supportive"

    def build_prompt(
        self,
        user_text: str,
        classification: Dict[str, Any],
        override_strategy: str = None,
    ) -> Tuple[str, str]:
        # Normalize classifier outputs to labels
        emotion_label = self._label_from(classification.get("emotion"))
        intent_label = self._label_from(classification.get("intent"))
        risk_label = self._label_from(classification.get("risk"))

        strategy = self._choose_strategy(emotion_label, intent_label, risk_label, override_strategy)

        # Strong system-style instruction for the generator
        sys_note = (
            "You are an empathetic mental health support assistant.\n"
            f"The user currently seems to be feeling: {emotion_label}.\n"
            f"Their intent appears to be: {intent_label}.\n"
            f"Risk level: {risk_label}.\n"
            f"Your conversational strategy for this message is: {strategy}.\n\n"
            "Respond in a warm, validating, and non-judgmental way.\n"
            "Focus on the user's feelings and experiences, not your own.\n"
            "Do NOT say that you need help or that you are struggling.\n"
            "Do NOT provide clinical diagnoses or medical instructions.\n"
            "You can gently encourage reaching out to trusted people or professionals when appropriate.\n"
            "Keep your response concise, clear, and emotionally supportive."
        )

        # Include last few turns as context
        context_lines: List[str] = []
        for turn in self.history[-3:]:
            if turn.get("user"):
                context_lines.append(f"User: {turn['user']}")
            if turn.get("bot"):
                context_lines.append(f"Bot: {turn['bot']}")

        context_block = ("\n".join(context_lines) + "\n") if context_lines else ""

        # Final prompt
        prompt = (
            f"{sys_note}\n\n"
            f"{context_block}"
            f"User: {user_text}\n"
            "Bot:"
        )
        return prompt, strategy

    def update_history(self, user_text: str, bot_text: str) -> None:
        self.history.append({"user": user_text, "bot": bot_text})

