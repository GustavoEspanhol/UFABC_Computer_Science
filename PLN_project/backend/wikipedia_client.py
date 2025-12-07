# backend/wikipedia_client.py
import wikipedia

def fetch_wikipedia_summary(query: str, sentences: int = 3):
    """
    Try to fetch a summary for the given query.
    If not found or ambiguous, returns a helpful message.
    """
    if not query:
        return ""
    try:
        wikipedia.set_lang("pt")  # use Portuguese
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True, redirect=True)
        return summary
    except wikipedia.DisambiguationError as e:
        # pick first option
        option = e.options[0] if e.options else query
        try:
            return wikipedia.summary(option, sentences=sentences)
        except Exception:
            return f"Resumo não encontrado diretamente para '{query}'."
    except Exception:
        return f"Resumo não encontrado para '{query}'."
