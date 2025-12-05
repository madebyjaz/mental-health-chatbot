class IntentModel:
    def predict(self, text: str) -> str:
        lowered = text.lower().strip()

        crisis_markers = [
            "suicide", "kill myself", "end it", "hurt myself",
            "self-harm", "harm myself", "can't go on"
        ]
        if any(w in lowered for w in crisis_markers):
            return "crisis"

        help_terms = ["help", "support", "advice", "guidance", "suggest", "what should i do"]
        if any(w in lowered for w in help_terms):
            return "seek_support"

        question_starts = ("how ", "what ", "why ", "can ", "could ", "should ", "is ", "are ")
        if lowered.endswith("?") or lowered.startswith(question_starts):
            return "ask_question"

        emotional_sharing_markers = [
            "i feel", "i'm feeling", "i am", "im ", "feel ", "feeling "
        ]
        if any(w in lowered for w in emotional_sharing_markers):
            return "share_feelings"

        vent_terms = ["vent", "talk", "rant", "get this off my chest"]
        if any(w in lowered for w in vent_terms):
            return "vent"

        return "unknown"
