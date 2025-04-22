import colorsys
import random
import uuid

_client = None


def random_color() -> str:
    hue = random.random()
    sat = 0.1
    val = 0.99
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def random_id() -> str:
    return str(uuid.uuid4())


def get_openai_client():
    global _client

    if _client is not None:
        return _client

    import os

    import openai
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    openai.api_key = api_key
    _client = openai

    return _client
