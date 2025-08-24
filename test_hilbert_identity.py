
import torch

def hilbert_projective_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # p,q in simplex (>=0, sum=1)
    ratio = (p / (q + 1e-12))
    mx, _ = ratio.max(dim=1)
    mn, _ = ratio.min(dim=1)
    return torch.log(mx / mn + 1e-12)

def test_softmax_identity():
    torch.manual_seed(0)
    B, K = 32, 7
    z1 = torch.randn(B, K)
    z2 = torch.randn(B, K)
    p = torch.softmax(z1, dim=1)
    q = torch.softmax(z2, dim=1)
    dH = hilbert_projective_distance(p, q)
    dz = z1 - z2
    rng = dz.max(dim=1).values - dz.min(dim=1).values
    assert torch.allclose(dH, rng, atol=1e-5), f"Mismatch: {(dH - rng).abs().max()}"
