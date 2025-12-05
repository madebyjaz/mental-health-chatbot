def choose_strategy(emotion, intent, risk):

    if risk == "crisis":
        return "CRISIS_OVERRIDE"

    if intent == "seeking_support":
        return "validate_and_reassure"

    if intent == "venting":
        return "reflective_listening"

    if intent == "information":
        return "provide_information_gently"

    return "general_empathy"