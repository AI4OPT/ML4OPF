import pytest
import torch

from ml4opf.layers.slackbus_repair import SlackBusRepair


def test_slackbus_repair():
    slackbus_idx = torch.tensor(0)
    layer = SlackBusRepair(slackbus_idx)
    layer1 = SlackBusRepair(0)
    layer2 = SlackBusRepair(torch.tensor([0]))
    layer3 = SlackBusRepair(torch.tensor([[[0]]]))
    va = torch.tensor([1.0, 2.0, 3.0])
    va2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert (layer(va) == torch.tensor([0.0, 1.0, 2.0])).all()
    assert (layer1(va) == torch.tensor([0.0, 1.0, 2.0])).all()
    assert (layer2(va) == torch.tensor([0.0, 1.0, 2.0])).all()
    assert (layer3(va) == torch.tensor([0.0, 1.0, 2.0])).all()
    assert (layer(va2) == torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])).all()
    assert (layer1(va2) == torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])).all()
    assert (layer2(va2) == torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])).all()
    assert (layer3(va2) == torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])).all()

    with pytest.raises(TypeError):
        SlackBusRepair("0")

    with pytest.raises(TypeError):
        SlackBusRepair(0.0)

    with pytest.raises(TypeError):
        SlackBusRepair(torch.tensor(0.0))
