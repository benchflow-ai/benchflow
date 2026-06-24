"""Vendored OSWorld metrics (xlang-ai/OSWorld, Apache-2.0).

The original __init__ eagerly imported every module (pulling torch/cv2/librosa).
BenchFlow imports metric modules individually + lazily (see
``benchflow.adapters.osworld_vendor``), so this package init is intentionally
minimal — each submodule uses absolute imports and does not need siblings here.
"""
