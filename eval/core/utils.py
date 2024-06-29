from typing import Dict
def create_user_message(user: str) -> Dict[str, str]:
    return {"role": "user", "content": user}


def create_assistant_message(assistant: str) -> Dict[str, str]:
    return {"role": "assistant", "content": assistant}


def create_system_message(system: str) -> Dict[str, str]:
    return {"role": "system", "content": system}

