"""Title a chat after its first completed turn."""

from app.ai.prompts import Prompt, register


PROMPT = register(
    Prompt(
        name="chat_title",
        version="chat-title@1",
        system="Write a specific title of at most six words. Return only the structured result.",
        user="User: {question}\nAssistant: {answer}\n\nTitle this conversation.",
    )
)
