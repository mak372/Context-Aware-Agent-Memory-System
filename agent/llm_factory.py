import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini").lower()


def get_llm(model_name: str, max_tokens: int = 1024):
    if LLM_BACKEND == "anthropic":
        return ChatAnthropic(
            model=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            max_tokens=max_tokens,
        )
    elif LLM_BACKEND == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY", ""),
            max_output_tokens=max_tokens,
        )
    elif LLM_BACKEND == "ollama":
        return ChatOpenAI(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            max_tokens=max_tokens,
        )
    elif LLM_BACKEND == "groq":
        return ChatOpenAI(
            model=model_name,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("GROQ_API_KEY", ""),
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{LLM_BACKEND}'. Choose: ollama | anthropic | groq | gemini")
