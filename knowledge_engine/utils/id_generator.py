from uuid import uuid4


def generate_id() -> str:
    """
    Generate a unique identifier.
    """
    return str(uuid4())