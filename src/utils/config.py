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
    timeout_seconds: int = 600
    max_tool_calls: int = 24
    max_search_calls: int = 8
    max_scrape_calls: int = 12

class CuratorConfig(AgentConfig):
    """Curator-specific config, extending AgentConfig with maintenance thresholds."""
    # Fraction of db_limit_mb at which aggressive vector pruning kicks in (0.0–1.0)
    aggressive_threshold: float = 0.8
    # Files older than this many days are eligible for LLM synthesis + archival
    consolidation_cutoff_days: int = 90
    # Hard storage cap in megabytes (Supabase free tier = 500MB)
    db_limit_mb: int = 500

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
    research: AgentConfig = Field(default_factory=lambda: AgentConfig(
        provider=Provider(os.environ.get("RESEARCH_PROVIDER", os.environ["BULL_PROVIDER"])),
        model=os.environ.get("RESEARCH_MODEL", os.environ["BULL_MODEL"]),
        temperature=float(os.environ.get("RESEARCH_TEMPERATURE", os.environ["BULL_TEMPERATURE"])),
        max_iterations=int(os.environ.get("RESEARCH_RECURSION_LIMIT", "80")),
        thinking=os.environ.get("RESEARCH_THINKING", os.environ.get("BULL_THINKING", "false")).lower() == "true",
        timeout_seconds=int(os.environ.get("RESEARCH_TIMEOUT_SECONDS", "600")),
        max_tool_calls=int(os.environ.get("RESEARCH_MAX_TOOL_CALLS", "24")),
        max_search_calls=int(os.environ.get("RESEARCH_MAX_SEARCH_CALLS", "8")),
        max_scrape_calls=int(os.environ.get("RESEARCH_MAX_SCRAPE_CALLS", "12")),
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
    curator: CuratorConfig = Field(default_factory=lambda: CuratorConfig(
        provider=Provider(os.environ.get("CURATOR_PROVIDER", os.environ["REPORT_GENERATOR_PROVIDER"])),
        model=os.environ.get("CURATOR_MODEL", os.environ["REPORT_GENERATOR_MODEL"]),
        temperature=float(os.environ.get("CURATOR_TEMPERATURE", "0.1")),
        thinking=os.environ.get("CURATOR_THINKING", "false").lower() == "true",
        aggressive_threshold=float(os.environ.get("CURATOR_AGGRESSIVE_THRESHOLD", "0.8")),
        consolidation_cutoff_days=int(os.environ.get("CURATOR_CONSOLIDATION_CUTOFF_DAYS", "90")),
        db_limit_mb=int(os.environ.get("CURATOR_DB_LIMIT_MB", "500")),
    ))


def is_deepseek_reasoner_model(model: str) -> bool:
    return model.lower().startswith("deepseek-reasoner")


def _tool_compatible_config(config: AgentConfig, role: str) -> AgentConfig:
    """Return a model config that can be used with tool-calling workflows."""
    if (
        config.provider == Provider.DEEPSEEK
        and is_deepseek_reasoner_model(config.model)
    ):
        fallback_model = os.environ.get(
            f"{role}_TOOL_FALLBACK_MODEL",
            os.environ.get("DEEPSEEK_TOOL_FALLBACK_MODEL", "deepseek-chat"),
        )
        logger.warning(
            "%s_MODEL=%s does not support DeepSeek tool calling; using %s "
            "for this tool-using workflow. Use deepseek-v4-pro with "
            "%s_THINKING=true for thinking-mode tool calls.",
            role,
            config.model,
            fallback_model,
            role,
        )
        return config.model_copy(update={"model": fallback_model, "thinking": False})
    return config


def get_llm(
    config: AgentConfig,
    *,
    require_tool_calling: bool = False,
    role: str = "AGENT",
):
    if require_tool_calling:
        config = _tool_compatible_config(config, role)

    if config.provider == Provider.OPENROUTER:
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(model=config.model, temperature=config.temperature)
    elif config.provider == Provider.DEEPSEEK:
        if config.thinking:
            from utils.deepseek_thinking import ChatDeepSeekThinking
            return ChatDeepSeekThinking(
                model=config.model,
                temperature=config.temperature,
                extra_body={"thinking": {"type": "enabled"}},
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
        self.bull_model = get_llm(
            self.config.bull,
            require_tool_calling=True,
            role="BULL",
        )
        self.bear_model = get_llm(
            self.config.bear,
            require_tool_calling=True,
            role="BEAR",
        )
        self.research_model = get_llm(
            self.config.research,
            require_tool_calling=True,
            role="RESEARCH",
        )
        self.supervisor_model = get_llm(
            self.config.supervisor,
            require_tool_calling=True,
            role="SUPERVISOR",
        )
        self.report_generator_model = get_llm(self.config.report_generator)
        self.curator_model = get_llm(self.config.curator)


secrets = Secrets()
models = Models()
bull_model = models.bull_model
bear_model = models.bear_model
research_model = models.research_model
supervisor_model = models.supervisor_model
report_generator_model = models.report_generator_model
judge_model = models.judge_model
valuation_model = models.valuation_model
curator_model = models.curator_model

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
