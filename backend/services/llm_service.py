"""
LLM service: provider factory and answer generation.

API keys are passed in per-call and never stored or logged.
Supported providers: anthropic, openai, groq.
"""

_ANSWER_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the context \
provided below. If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""


def get_llm_client(provider: str, api_key: str):
    """
    Return an authenticated client for the given LLM provider.

    Raises ValueError for unrecognized provider strings.
    """
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "openai":
        import openai
        return openai.OpenAI(api_key=api_key)
    elif provider == "groq":
        import groq
        return groq.Groq(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Supported: 'anthropic', 'openai', 'groq'."
        )


def generate_answer(
    client,
    provider: str,
    context_chunks: list[str],
    query: str,
) -> str:
    """
    Generate a grounded answer from retrieved context chunks.

    Returns an error string (never raises) so that retrieval results are
    always returned to the caller even when the LLM call fails.
    """
    try:
        numbered = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(context_chunks))
        prompt = _ANSWER_PROMPT.format(context=numbered, query=query)

        if provider == "anthropic":
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        elif provider in ("openai", "groq"):
            model = "gpt-4o-mini" if provider == "openai" else "llama-3.1-8b-instant"
            response = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

        else:
            return f"Error generating answer: unsupported provider '{provider}'."

    except Exception as exc:
        return f"Error generating answer: {exc}"
