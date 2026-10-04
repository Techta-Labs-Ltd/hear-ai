import pytest

from hear.inference.text_generation import VllmTextGenerationEngine

GIB = 1024**3


def test_budget_is_converted_to_a_fraction_of_the_card():
    assert VllmTextGenerationEngine.utilization_for(8.5, total_bytes=24 * GIB) == pytest.approx(
        0.3542, abs=1e-4
    )
    assert VllmTextGenerationEngine.utilization_for(8.5, total_bytes=48 * GIB) == pytest.approx(
        0.1771, abs=1e-4
    )


@pytest.mark.parametrize("budget", [0, -1, 23])
def test_budget_that_leaves_no_room_for_the_pipeline_is_rejected(budget):
    with pytest.raises(ValueError, match="qwen_llm_memory_budget_does_not_fit_gpu"):
        VllmTextGenerationEngine.utilization_for(budget, total_bytes=24 * GIB)
