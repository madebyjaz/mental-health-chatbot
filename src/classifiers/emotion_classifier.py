class EmotionModel:
    def predict(self, text: str) -> str:
        lowered = text.lower()

        sad_terms = [
            "sad", "down", "depressed", "unhappy", "cry",
            "lonely", "hopeless", "empty", "numb",
            "worthless", "miserable", "heartbroken",
            "drained", "exhausted", "overwhelmed", "heavy"
        ]
        anxious_terms = [
            "anxious", "nervous", "worried", "panic",
            "scared", "afraid", "on edge", "uneasy"
        ]
        angry_terms = [
            "angry", "mad", "furious", "irritated",
            "upset", "resentful", "frustrated"
        ]

        if any(t in lowered for t in sad_terms):
            return "sadness"
        if any(t in lowered for t in anxious_terms):
            return "anxiety"
        if any(t in lowered for t in angry_terms):
            return "anger"

        return "neutral"
