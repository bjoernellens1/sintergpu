import pytest
import torch

from vkl.bench.input_isolation import isolated_call


def test_isolated_call_rejects_candidate_input_mutation():
    values = (torch.tensor([1.0]), {"control": torch.tensor([2.0])})

    def mutating(value, controls):
        value.add_(1.0)
        controls["control"].add_(1.0)
        return value

    with pytest.raises(RuntimeError, match="mutated benchmark inputs"):
        isolated_call(mutating, values)

    assert torch.equal(values[0], torch.tensor([1.0]))
    assert torch.equal(values[1]["control"], torch.tensor([2.0]))
