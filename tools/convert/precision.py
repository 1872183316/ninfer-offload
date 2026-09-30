"""Uniform precision choices applied on top of a recipe (`tools.convert --precision`).

Three classes of text projections can be re-quantized with one grouped-integer width each:

  experts      routed expert gate/up projections
  expert-down  routed expert down projections
  linear       every other text-layer projection the recipe already quantizes (attention, GDN,
               shared experts, dense MLP). Projections a recipe keeps in BF16 (routers, GDN a/b)
               and the embedding, output head and PLE table keep the recipe's choice.

Widths are 4, 5, 6 or 8 bits (the grouped formats NInfer's kernels execute). The estimate
functions report the artifact payload implied by a configured recipe without reading weights.
"""

from __future__ import annotations

from tools.artifact.formats import DirectFormat, Fp8RowFormat, Nvfp4Format, QuantFormat, get_format

from .methods import grouped_absmax
from .recipe import Recipe

FORMATS = {4: "q4_g64_fp16", 5: "q5_g64_fp16", 6: "q6_g64_fp16", 8: "q8_g32_fp16"}
CLASSES = ("experts", "expert-down", "linear")


def parse(spec: str) -> dict[str, int]:
    """Parse `experts=4,expert-down=5,linear=8` (any subset)."""

    result = {}
    for item in spec.split(","):
        key, separator, value = item.strip().partition("=")
        if not separator or key not in CLASSES or key in result:
            raise ValueError(
                f"--precision expects unique CLASS=BITS items with CLASS in {', '.join(CLASSES)}"
            )
        if not value.isdigit() or int(value) not in FORMATS:
            raise ValueError(f"--precision {key}: bits must be one of 4, 5, 6, 8")
        result[key] = int(value)
    return result


def classify(name: str) -> str | None:
    if not name.startswith("text/layers/"):
        return None
    if "/moe/experts/" in name:
        return "expert-down" if name.endswith("/down") else "experts"
    return "linear"


def classes_present(recipe: Recipe) -> set[str]:
    return {
        cls for name in recipe.selections
        if recipe.model.parameters[name].projection and (cls := classify(name)) is not None
    }


def apply(recipe: Recipe, choice: dict[str, int]) -> None:
    """Re-assign each selected class; only projections the recipe quantized are changed."""

    present = classes_present(recipe)
    for key in choice:
        if key not in present:
            raise ValueError(f"--precision {key}: this model has no such projections")
    for name, selections in recipe.selections.items():
        cls = classify(name)
        if cls not in choice or not recipe.model.parameters[name].projection:
            continue
        if not all(isinstance(get_format(s.format), QuantFormat) for s in selections):
            continue
        recipe.assign(name, format=FORMATS[choice[cls]], layout="auto", method=grouped_absmax)


def _payload(format: str, elements: int) -> float:
    kind = get_format(format)
    if isinstance(kind, QuantFormat):
        return elements * kind.bits / 8 + elements / kind.group_size * 2
    if isinstance(kind, DirectFormat):
        return elements * kind.word_bytes
    if isinstance(kind, Nvfp4Format):
        return elements / 2 + elements / kind.group_size
    if isinstance(kind, Fp8RowFormat):
        return elements
    raise ValueError(f"no size model for {format}")


def estimate(recipe: Recipe) -> dict[str, float]:
    """Stored payload bytes by class (plus `other` and `total`), ignoring framing."""

    sizes = {key: 0.0 for key in (*CLASSES, "other")}
    shared = set(recipe.aliases)
    for name, selections in recipe.selections.items():
        if name in shared:
            continue
        cls = classify(name) if recipe.model.parameters[name].projection else None
        for s in selections:
            sizes[cls or "other"] += _payload(s.format, s.end - s.begin)
    sizes["total"] = sum(sizes.values())
    return sizes
