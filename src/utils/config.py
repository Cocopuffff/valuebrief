import os
import json
from enum import Enum
from dotenv import load_dotenv
from utils.logger import get_logger
from pydantic import BaseModel, Field, SecretStr

logger = get_logger(__name__)
load_dotenv()

class Secrets:
    SUPABASE_URI = os.environ["SUPABASE_CONNECTION_STRING"]
    if "sslmode" not in SUPABASE_URI:
        SUPABASE_URI += "?sslmode=require"
    if not SUPABASE_URI:
        raise ValueError("SUPABASE_CONNECTION_STRING is not set in environment")

class Provider(str, Enum):
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"

class AgentConfig(BaseModel):
    provider: Provider = Provider.OPENROUTER
    model: str = "qwen/qwen3.6-plus"
    temperature: float = 0.2
    max_iterations: int = 2
    thinking: bool = False  # Enable DeepSeek thinking mode (deepseek-v4-pro etc.)

class ModelConfig(BaseModel):
    judge: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ["JUDGE_PROVIDER"]),
        model=os.environ["JUDGE_MODEL"],
        temperature=float(os.environ["JUDGE_TEMPERATURE"]),
        thinking=os.environ.get("JUDGE_THINKING", "false").lower() == "true"
    ))
    valuation: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ["VALUATION_PROVIDER"]),
        model=os.environ["VALUATION_MODEL"],
        temperature=float(os.environ["VALUATION_TEMPERATURE"]),
        thinking=os.environ.get("VALUATION_THINKING", "false").lower() == "true"
    ))
    bull: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ["BULL_PROVIDER"]),
        model=os.environ["BULL_MODEL"],
        temperature=float(os.environ["BULL_TEMPERATURE"]),
        max_iterations=int(os.environ["BULL_MAX_ITERATIONS"]),
        thinking=os.environ.get("BULL_THINKING", "false").lower() == "true"
    ))
    bear: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ["BEAR_PROVIDER"]),
        model=os.environ["BEAR_MODEL"],
        temperature=float(os.environ["BEAR_TEMPERATURE"]),
        max_iterations=int(os.environ["BEAR_MAX_ITERATIONS"]),
        thinking=os.environ.get("BEAR_THINKING", "false").lower() == "true"
    ))
    supervisor: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ["SUPERVISOR_PROVIDER"]),
        model=os.environ["SUPERVISOR_MODEL"],
        temperature=float(os.environ["SUPERVISOR_TEMPERATURE"]),
        thinking=os.environ.get("SUPERVISOR_THINKING", "false").lower() == "true"
    ))
    report_generator: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ["REPORT_GENERATOR_PROVIDER"]),
        model=os.environ["REPORT_GENERATOR_MODEL"],
        temperature=float(os.environ["REPORT_GENERATOR_TEMPERATURE"]),
        thinking=os.environ.get("REPORT_GENERATOR_THINKING", "false").lower() == "true"
    ))

def get_llm(config: AgentConfig):
    if config.provider == Provider.OPENROUTER:
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(model=config.model, temperature=config.temperature)
    elif config.provider == Provider.DEEPSEEK:
        if config.thinking:
            from utils.deepseek_thinking import ChatDeepSeekThinking
            return ChatDeepSeekThinking(
                model=config.model,
                temperature=config.temperature,
                model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
            )
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=config.model, temperature=config.temperature)
    elif config.provider == Provider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.model, temperature=config.temperature)
    elif config.provider == Provider.GOOGLE:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=config.model, temperature=config.temperature)
        except ImportError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                api_key=SecretStr(os.environ.get("GEMINI_API_KEY", "dummy")),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

class Models:
    def __init__(self):
        self.config = ModelConfig()
        self.judge_model = get_llm(self.config.judge)
        self.valuation_model = get_llm(self.config.valuation)
        self.bull_model = get_llm(self.config.bull)
        self.bear_model = get_llm(self.config.bear)
        self.supervisor_model = get_llm(self.config.supervisor)
        self.report_generator_model = get_llm(self.config.report_generator)


secrets = Secrets()
models = Models()
bull_model = models.bull_model
bear_model = models.bear_model
supervisor_model = models.supervisor_model
report_generator_model = models.report_generator_model
judge_model = models.judge_model
valuation_model = models.valuation_model

config = models.config

def load_exchange_mappings() -> dict:
    mapping_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "exchange_mappings.json")
    try:
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                data = json.load(f)
                return data.get("yahoo_to_alphavantage", {})
        else:
            logger.warning(f"exchange_mappings.json not found at {mapping_file}")
            return {}
    except Exception as e:
        logger.warning(f"Failed to load exchange_mappings.json: {e}")
        return {}

exchange_mappings = load_exchange_mappings()