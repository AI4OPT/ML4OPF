import pytest

import torch
from ml4opf.layers.bound_repair import BoundRepair


def test_bound_repair():
    xmin = torch.tensor([-2.0, -torch.inf, 3.0, 0.0, -torch.inf])
    xmax = torch.tensor([torch.inf, 2.0, 3.0, 10.0, torch.inf])

    x1 = torch.tensor([-3.0, -5.0, -4.0, -6.0, 1.0])
    x2 = torch.tensor([4.0, 6.0, 5.0, 11.0, -2.0])
    x3 = torch.tensor([0.0, 2.0, 3.0, 8.0, 3.0])
    x4 = torch.tensor([3.0, 1.0, 2.0, -10.0, 4.0])
    x5 = torch.stack([x1, x2, x3, x4])
    x6 = 1e4 * torch.randn(100, 5)

    for method in BoundRepair.SUPPORTED_METHODS:
        for mem in [0, 1, 2]:
            xmin_ = xmin.clone()
            xmax_ = xmax.clone()

            if method in ["sigmoid", "tanh"]:
                with pytest.raises(AssertionError):
                    bc = BoundRepair(xmin, xmax, method=method)
                xmin[0] = -torch.inf
                xmax[1] = torch.inf

            bc = BoundRepair(xmin, xmax, method=method, memory_efficient=mem)
            bc2 = BoundRepair(xmin, xmax, method="relu", memory_efficient=2)
            bcf = BoundRepair(None, None, method=method, memory_efficient=mem)
            bcf2 = BoundRepair(None, None, method="tanh", memory_efficient=0)
            bcf2.load_state_dict(bc2.state_dict())
            bcf2.preprocess_bounds(2)
            bcf2.preprocess_bounds(0)
            bc2.load_state_dict(bcf.state_dict())
            bcf2.load_state_dict(bc.state_dict())

            bc.preprocess_bounds(2)
            assert "xmin_lower" in bc._properties
            bc.preprocess_bounds(0)
            assert "xmin_lower" not in bc._properties
            bc.preprocess_bounds(1)
            assert "xmin_lower" in bc._properties
            assert "lower_mask" not in bc._properties

            sd = bc.state_dict()
            sd["extra"] = torch.tensor(1.0)
            with pytest.raises(ValueError):
                bc.load_state_dict(sd, strict=True)

            bc = BoundRepair(xmin, xmax, method=method, memory_efficient=mem)

            assert str(bc) == f"BoundRepair(method={method})"
            assert str(bcf) == f"BoundRepair(method={method})"
            for x in [x1, x2, x3, x4, x5, x6]:
                cx = bc(x)
                cx2 = bc2(x, xmin, xmax)
                cxf = bcf(x, xmin, xmax)
                cxff = bc(x, xmin, xmax)
                cxf2 = bcf2(x)
                if method == "none":
                    assert torch.all(cx == x)
                    assert torch.all(cx2 == x)
                    assert torch.all(cxf == x)
                    assert torch.all(cxff == x)
                    assert torch.all(cxf2 == x)
                else:
                    assert cx.shape == x.shape
                    assert torch.all(cx >= xmin)
                    assert torch.all(cx <= xmax)
                    assert cx2.shape == x.shape
                    assert torch.all(cx2 >= xmin)
                    assert torch.all(cx2 <= xmax)
                    assert cxf.shape == x.shape
                    assert torch.all(cxf >= xmin)
                    assert torch.all(cxf <= xmax)
                    assert cxff.shape == x.shape
                    assert torch.all(cxff >= xmin)
                    assert torch.all(cxff <= xmax)
                    assert cxf2.shape == x.shape
                    assert torch.all(cxf2 >= xmin)
                    assert torch.all(cxf2 <= xmax)

            xmin = xmin_
            xmax = xmax_
