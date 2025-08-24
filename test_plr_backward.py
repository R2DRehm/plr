
import torch
from plr_hilbert.losses.plr import plr_loss

def test_plr_backward():
    torch.manual_seed(0)
    B, K, D = 64, 5, 10
    logits = torch.randn(B, K, requires_grad=True)
    X = torch.randn(B, D, requires_grad=False)  # input space (no grad) — standard use
    loss, info = plr_loss(logits, X, k=2, tau=0.3)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Now with feature space requiring grad
    logits2 = torch.randn(B, K, requires_grad=True)
    feat = torch.randn(B, D, requires_grad=True)
    loss2, _ = plr_loss(logits2, feat, k=2, tau=0.3)
    loss2.backward()
    assert logits2.grad is not None and feat.grad is not None
    assert torch.isfinite(logits2.grad).all() and torch.isfinite(feat.grad).all()
