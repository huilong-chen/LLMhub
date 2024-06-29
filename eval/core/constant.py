COT_CODE = "Let's think step by step."

def create_user_message(user: str) -> dict[str, str]:
    return {"role": "user", "content": user}


def create_assistant_message(assistant: str) -> dict[str, str]:
    return {"role": "assistant", "content": assistant}


def create_system_message(system: str) -> dict[str, str]:
    return {"role": "system", "content": system}
