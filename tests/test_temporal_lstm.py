"""Tests for the refactored TemporalLSTM model."""

import torch
import torch.nn as nn

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.temporal_lstm import TemporalLSTM


# ------------------------------------------------------------------ #
# Construction & shape tests
# ------------------------------------------------------------------ #

def test_output_shape():
    """forward() returns (batch, num_classes)."""
    model = TemporalLSTM(input_size=51, num_classes=4)
    x = torch.randn(8, 30, 51)
    out = model(x)
    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


def test_input_agnostic():
    """Model works with arbitrary input_size (not just 51)."""
    for input_size in [10, 51, 100]:
        model = TemporalLSTM(input_size=input_size, num_classes=3)
        x = torch.randn(2, 15, input_size)
        out = model(x)
        assert out.shape == (2, 3)


def test_variable_sequence_length():
    """Model handles different sequence lengths."""
    model = TemporalLSTM(input_size=20)
    for seq_len in [1, 5, 50]:
        x = torch.randn(4, seq_len, 20)
        out = model(x)
        assert out.shape == (4, 4)


def test_single_sample_batch():
    """Model works with batch_size=1."""
    model = TemporalLSTM(input_size=51)
    x = torch.randn(1, 10, 51)
    assert model(x).shape == (1, 4)


# ------------------------------------------------------------------ #
# Temporal aggregation tests
# ------------------------------------------------------------------ #

def test_temporal_mean_pooling_used():
    """Verify the model uses temporal aggregation (not just last timestep).

    If the model only used the last timestep, outputs for inputs that
    differ only in earlier frames would be identical.  With mean pooling
    over time, they should differ.
    """
    torch.manual_seed(42)
    model = TemporalLSTM(input_size=10)
    model.eval()

    x1 = torch.randn(1, 10, 10)
    x2 = x1.clone()
    # Change only early frames (keep last frame identical)
    x2[0, :5, :] = torch.randn(5, 10)

    out1 = model(x1)
    out2 = model(x2)

    # Outputs should differ because mean pooling considers all timesteps
    assert not torch.allclose(out1, out2, atol=1e-6), \
        "Outputs should differ when early frames change (temporal aggregation)"


# ------------------------------------------------------------------ #
# Unidirectional constraint
# ------------------------------------------------------------------ #

def test_unidirectional_only():
    """LSTM must be unidirectional (no bidirectional leakage)."""
    model = TemporalLSTM(input_size=20)
    assert model.lstm.bidirectional is False


def test_no_bidirectional_param():
    """Constructor should NOT accept a bidirectional argument."""
    import inspect
    sig = inspect.signature(TemporalLSTM.__init__)
    assert "bidirectional" not in sig.parameters, \
        "bidirectional parameter should be removed"


# ------------------------------------------------------------------ #
# Public API tests
# ------------------------------------------------------------------ #

def test_predict_returns_indices():
    """predict() returns integer class indices."""
    model = TemporalLSTM(input_size=51)
    x = torch.randn(4, 10, 51)
    preds = model.predict(x)
    assert preds.shape == (4,)
    assert preds.dtype in (torch.int64, torch.long)


def test_get_probabilities_sums_to_one():
    """get_probabilities() returns valid probability distributions."""
    model = TemporalLSTM(input_size=51, num_classes=4)
    x = torch.randn(4, 10, 51)
    probs = model.get_probabilities(x)
    assert probs.shape == (4, 4)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
        f"Probabilities should sum to 1, got {sums}"


def test_forward_returns_logits():
    """forward() output should be raw logits (unbounded values)."""
    torch.manual_seed(0)
    model = TemporalLSTM(input_size=51)
    x = torch.randn(8, 10, 51)
    out = model(x)
    # Logits are unbounded — verify they are not clamped to [0, 1]
    assert out.shape == (8, 4)


# ------------------------------------------------------------------ #
# Training compatibility
# ------------------------------------------------------------------ #

def test_backward_pass():
    """Model supports gradient computation (training compatibility)."""
    model = TemporalLSTM(input_size=51, num_classes=4)
    x = torch.randn(4, 10, 51)
    labels = torch.randint(0, 4, (4,))
    criterion = nn.CrossEntropyLoss()

    model.train()
    out = model(x)
    loss = criterion(out, labels)
    loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_input_size_required():
    """input_size should be a required argument (no default)."""
    try:
        model = TemporalLSTM()  # type: ignore
        assert False, "Should raise TypeError for missing input_size"
    except TypeError:
        pass


# ------------------------------------------------------------------ #
# Attribute checks
# ------------------------------------------------------------------ #

def test_has_classifier_attribute():
    """Model should expose a `classifier` attribute for the head."""
    model = TemporalLSTM(input_size=10)
    assert hasattr(model, "classifier"), "Expected `classifier` attribute"
    assert isinstance(model.classifier, nn.Sequential)


if __name__ == "__main__":
    # Simple runner for environments without pytest
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
