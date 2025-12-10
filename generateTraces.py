import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

TEACHER_MODEL = "deepseek/deepseek-v3.2-speciale"
MAX_GENERATION_TOKENS = 8192  # Maximum tokens for generation: 8,192 tokens

def generate_trace(prompt: str) -> str:
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_GENERATION_TOKENS,
    )

    return response.choices[0].message.content

