import re
import uuid

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def get_api_key(authorization):
    if authorization:
        match = re.match("^Bearer (.+)$", authorization)
        if match:
            return match.group(1)
    return "unknown"
