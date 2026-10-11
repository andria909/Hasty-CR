import pytest
import torch

from sim.policy_soup import task_vector_soup


def test_task_vector_soup_combines_deltas_from_baseline():
    base = {"weight": torch.tensor([2.0, 4.0]),
            "counter": torch.tensor(3)}
    first = {"weight": torch.tensor([4.0, 2.0]),
             "counter": torch.tensor(3)}
    second = {"weight": torch.tensor([1.0, 8.0]),
              "counter": torch.tensor(3)}

    result = task_vector_soup(base, [first, second], [0.25, 0.5])

    assert torch.allclose(result["weight"], torch.tensor([2.0, 5.5]))
    assert result["counter"].item() == 3
    assert result["weight"].data_ptr() != base["weight"].data_ptr()


def test_task_vector_soup_rejects_incompatible_state_dicts():
    base = {"weight": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="incompatible keys"):
        task_vector_soup(base, [{"other": torch.tensor([1.0])}], [0.1])


def test_task_vector_soup_rejects_changed_integer_buffers():
    base = {"counter": torch.tensor(1)}
    with pytest.raises(ValueError, match="non-floating"):
        task_vector_soup(base, [{"counter": torch.tensor(2)}], [0.1])
