def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    return text


def validate_name(name: str, *, field: str = "name") -> str:
    if not isinstance(name, str):
        raise TypeError(f"{field.capitalize()} must be a string.")

    name = name.strip()

    if not name:
        raise ValueError(f"{field.capitalize()} cannot be empty.")

    return name