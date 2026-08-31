"""Public package surface with GPU-heavy imports kept lazy.

Importing the package is useful for CPU-side control-plane tools and tests. The
LLM implementation pulls in Triton and model layers, so load it only when the
caller actually requests ``LLM``.
"""

from prism_infer.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from prism_infer.llm import LLM

        globals()[name] = LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
