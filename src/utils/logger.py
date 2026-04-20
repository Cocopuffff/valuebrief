import logging
import os
from logging.handlers import RotatingFileHandler
import json
import inspect
import functools
import copy

def setup_logging(log_level=logging.INFO, log_file="logs/valuebrief.log"):
    """Sets up logging for the entire project."""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers if any (to avoid duplicate logs)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Rotating: 5MB per file, max 5 files)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=10
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Level: {logging.getLevelName(log_level)}, File: {log_file}")

def get_logger(name):
    """Returns a logger with the specified name."""
    return logging.getLogger(name)

DEBUG_NODES = os.getenv("LOG_NODE_EXECUTION", "false").lower() == "true"
MAX_STR_LEN = 1000

def _safe_json_encoder(obj):
    # Handle known types natively to maintain valid JSON hierarchy
    if hasattr(obj, "model_dump"): return obj.model_dump()
    if hasattr(obj, "dict"): return obj.dict()
    if hasattr(obj, "to_dict"): return obj.to_dict()
    
    # Fallback to string and truncate if enormously large (like full HTML bodies)
    s = str(obj)
    if len(s) > MAX_STR_LEN:
        return s[:MAX_STR_LEN] + f"... [truncated {len(s)-MAX_STR_LEN} chars]"
    return s

def log_node_execution(node_func):
    """Decorator to log the entry, context, output, and exit of a LangGraph node."""
    
    # Create a dynamic logger distinct to the specific function (e.g., 'agents.analysts.bull_analyst')
    node_name = node_func.__name__.upper()
    func_logger = logging.getLogger(f"{node_func.__module__}.{node_func.__name__}")

    async def async_wrapper(state, *args, **kwargs):
        if not DEBUG_NODES:
            return await node_func(state, *args, **kwargs)
            
        state_copy = copy.copy(state)
        func_logger.debug(f"\n{'='*60}\n🚀 [START] NODE: {node_name}\n{'='*60}")
        func_logger.debug(f"📥 CONTEXT SUPPLIED (State):\n```json\n{json.dumps(state_copy, indent=2, default=_safe_json_encoder)}\n```")
        
        result = await node_func(state, *args, **kwargs)
        
        # Format output safely
        output_rep = {"update": getattr(result, "update", None), "goto": getattr(result, "goto", None)} if hasattr(result, "update") else result
        func_logger.debug(f"📤 NODE OUTPUT:\n```json\n{json.dumps(output_rep, indent=2, default=_safe_json_encoder)}\n```")
        func_logger.debug(f"\n{'='*60}\n🏁 [END] NODE: {node_name}\n{'='*60}\n")
        return result

    def sync_wrapper(state, *args, **kwargs):
        if not DEBUG_NODES:
            return node_func(state, *args, **kwargs)
            
        state_copy = copy.copy(state)
        func_logger.debug(f"\n{'='*60}\n🚀 [START] NODE: {node_name}\n{'='*60}")
        func_logger.debug(f"📥 CONTEXT SUPPLIED (State):\n```json\n{json.dumps(state_copy, indent=2, default=_safe_json_encoder)}\n```")
        
        result = node_func(state, *args, **kwargs)
        
        output_rep = {"update": getattr(result, "update", None), "goto": getattr(result, "goto", None)} if hasattr(result, "update") else result
        func_logger.debug(f"📤 NODE OUTPUT:\n```json\n{json.dumps(output_rep, indent=2, default=_safe_json_encoder)}\n```")
        func_logger.debug(f"\n{'='*60}\n🏁 [END] NODE: {node_name}\n{'='*60}\n")
        return result

    if inspect.iscoroutinefunction(node_func):
        return functools.wraps(node_func)(async_wrapper)
    return functools.wraps(node_func)(sync_wrapper)
