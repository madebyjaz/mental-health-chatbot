class RiskModel:
    def predict(self, text: str) -> str:
        lowered = text.lower()
        if any(w in lowered for w in ["suicide", "kill myself", "end it", "harm myself", "die"]):
            return "crisis"
        if any(w in lowered for w in ["self-harm", "hurt myself", "cutting"]):
            return "high"
        return "low"

    def predict_proba(self, text: str) -> dict:
        lowered = text.lower()
        classes = ["low", "high", "crisis"]
        scores = {c: 0.1 for c in classes}  # smoothing
        crisis_markers = ["suicide", "kill myself", "end it", "harm myself", "die"]
        high_markers = ["self-harm", "hurt myself", "cutting"]
        scores["crisis"] += sum(1 for w in crisis_markers if w in lowered) * 1.0
        scores["high"] += sum(1 for w in high_markers if w in lowered) * 1.0
        # Low increases with absence of markers and generic language
        neutral_markers = ["okay", "fine", "thanks", "hello", "hi"]
        scores["low"] += 0.5 + sum(0.2 for w in neutral_markers if w in lowered)
        # Normalize
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}
