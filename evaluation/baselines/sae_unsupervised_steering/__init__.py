"""SAE-based unsupervised steering baseline.

Parallel pipeline to ``evaluation/unsupervised_steering`` that steers via
direct addition of SAE feature directions to the LLM residual stream
(decoder columns by default). No clustering; per-token; closed-form.
"""
