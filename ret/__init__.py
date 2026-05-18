# Register custom model types (e.g. gpt_rope) with HuggingFace auto-class
# machinery so AutoModelForCausalLM.from_pretrained works on custom checkpoints.
try:
    from LocalLLMs import gpt_rope  # noqa: F401
except ImportError:
    pass
