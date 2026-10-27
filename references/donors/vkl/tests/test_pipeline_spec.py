from __future__ import annotations

import pytest

from vkl.pipeline import NodeSpec, PipelineSpec


def test_pipeline_spec_rejects_cycle_before_executor_construction():
    spec = PipelineSpec(
        "cycle",
        (
            NodeSpec("a", "canonical_rescale", {"rgb": "b.rgb"}),
            NodeSpec("b", "canonical_rescale", {"rgb": "a.rgb"}),
        ),
    )

    with pytest.raises(ValueError, match="cycle"):
        spec.validate()


def test_pipeline_spec_rejects_unknown_output_reference():
    spec = PipelineSpec(
        "bad-output",
        (NodeSpec("a", "canonical_rescale", {"rgb": "input.rgb"}),),
        ("missing.rgb",),
    )

    with pytest.raises(ValueError, match="Unknown pipeline output"):
        spec.validate()


def test_pipeline_spec_accepts_input_and_node_outputs():
    PipelineSpec(
        "valid",
        (NodeSpec("a", "canonical_rescale", {"rgb": "input.rgb"}),),
        ("input.rgb", "a.rgb"),
    ).validate()
