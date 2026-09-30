from __future__ import annotations

import pytest
import torch

from tools.convert import precision
from tools.convert.methods import grouped_absmax
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

NAMES = {
    "text/layers/0/moe/experts/gate_up": ((8, 128), ("input",)),
    "text/layers/0/moe/experts/down": ((8, 128), ("input",)),
    "text/layers/0/attention/query": ((4, 128), ("input",)),
    "text/layers/0/moe/router": ((4, 128), ("input",)),
    "text/output_head": ((4, 128), ("input",)),
    "text/layers/0/norm": ((128,), ()),
}


def _recipe():
    model = Model({"text": {"config": {}}})
    for name, (shape, inputs) in NAMES.items():
        values = torch.ones(shape, dtype=torch.bfloat16)
        model.add(Parameter(name, shape, array_source(values, name), inputs=inputs))
    recipe = Recipe(model)
    for name in ("text/layers/0/moe/experts/gate_up", "text/layers/0/moe/experts/down",
                 "text/layers/0/attention/query", "text/output_head"):
        recipe.assign(name, format="q8_g32_fp16", method=grouped_absmax)
    return recipe


def _formats(recipe):
    return {name: selections[0].format for name, selections in recipe.selections.items()}


def test_classes_change_only_quantized_layer_projections():
    recipe = _recipe()
    precision.apply(recipe, precision.parse("experts=4,expert-down=5,linear=6"))
    assert _formats(recipe) == {
        "text/layers/0/moe/experts/gate_up": "q4_g64_fp16",
        "text/layers/0/moe/experts/down": "q5_g64_fp16",
        "text/layers/0/attention/query": "q6_g64_fp16",
        "text/layers/0/moe/router": "bf16",
        "text/output_head": "q8_g32_fp16",
        "text/layers/0/norm": "bf16",
    }


def test_estimate_counts_codes_and_group_scales():
    recipe = _recipe()
    precision.apply(recipe, {"experts": 4})
    sizes = precision.estimate(recipe)
    assert sizes["experts"] == 8 * 128 * 4 / 8 + 8 * 128 / 64 * 2
    assert sizes["expert-down"] == 8 * 128 + 8 * 128 / 32 * 2
    assert sizes["total"] == sum(v for k, v in sizes.items() if k != "total")


@pytest.mark.parametrize("spec", ["experts=3", "experts=4,experts=5", "embedding=8", "experts"])
def test_invalid_choices_are_rejected(spec):
    with pytest.raises(ValueError):
        precision.parse(spec)


def test_class_absent_from_model_is_rejected():
    model = Model({"text": {"config": {}}})
    model.add(Parameter("text/layers/0/mlp/down", (4, 128),
                        array_source(torch.ones(4, 128, dtype=torch.bfloat16), "d"),
                        inputs=("input",)))
    with pytest.raises(ValueError, match="no such projections"):
        precision.apply(Recipe(model), {"experts": 4})
