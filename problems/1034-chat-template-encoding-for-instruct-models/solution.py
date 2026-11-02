def encode_chat(user_message: str, system_message: str = None) -> list:
    """
    Build a Llama-3-style chat template token sequence.

    Args:
        user_message: the user's message (string)
        system_message: optional system message (string). If None, use
            "You are a helpful assistant.".

    Returns:
        A list of integer token IDs.
    """
    system_message = system_message if system_message is not None else "You are a helpful assistant."

    def encode(text):
        return [ord(c) for c in text]
    
    def encode_role(role_name):
        return [2] + encode(role_name) + [3] + encode("\n\n")
    
    return [
        1,
        *encode_role("system"),
        *encode(system_message),
        4,
        *encode_role("user"),
        *encode(user_message),
        4,
        *encode_role("assistant")
    ]