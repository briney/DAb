"""Tests for noise schedules."""

import pytest
import torch

from dab.diffusion.noise_schedule import (
    CosineSchedule,
    LinearSchedule,
    SqrtSchedule,
    StaticSchedule,
    create_schedule,
)


class TestLinearSchedule:
    def test_endpoints(self):
        schedule = LinearSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(0) == 0.0
        assert schedule.get_mask_rate(100) == 1.0

    def test_midpoint(self):
        schedule = LinearSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(50) == 0.5

    def test_tensor_input(self):
        schedule = LinearSchedule(num_timesteps=100)
        timesteps = torch.tensor([0, 50, 100])
        rates = schedule.get_mask_rate(timesteps)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(rates, expected)

    def test_sample_timesteps(self):
        schedule = LinearSchedule(num_timesteps=100)
        timesteps = schedule.sample_timesteps(batch_size=10, device=torch.device("cpu"))
        assert timesteps.shape == (10,)
        assert (timesteps >= 1).all()
        assert (timesteps <= 100).all()


class TestCosineSchedule:
    def test_endpoints(self):
        schedule = CosineSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(0) == pytest.approx(0.0, abs=1e-6)
        assert schedule.get_mask_rate(100) == pytest.approx(1.0, abs=1e-6)

    def test_monotonic(self):
        schedule = CosineSchedule(num_timesteps=100)
        prev_rate = 0.0
        for t in range(1, 101):
            rate = schedule.get_mask_rate(t)
            assert rate > prev_rate
            prev_rate = rate

    def test_tensor_input(self):
        schedule = CosineSchedule(num_timesteps=100)
        timesteps = torch.tensor([0, 100])
        rates = schedule.get_mask_rate(timesteps)
        assert rates[0] == pytest.approx(0.0, abs=1e-6)
        assert rates[1] == pytest.approx(1.0, abs=1e-6)


class TestSqrtSchedule:
    def test_endpoints(self):
        schedule = SqrtSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(0) == pytest.approx(0.0, abs=1e-6)
        assert schedule.get_mask_rate(100) == pytest.approx(1.0, abs=1e-6)

    def test_monotonic(self):
        schedule = SqrtSchedule(num_timesteps=100)
        prev_rate = 0.0
        for t in range(1, 101):
            rate = schedule.get_mask_rate(t)
            assert rate > prev_rate
            prev_rate = rate

    def test_quarter_point(self):
        schedule = SqrtSchedule(num_timesteps=100)
        # sqrt(25/100) = 0.5
        assert schedule.get_mask_rate(25) == pytest.approx(0.5, abs=1e-6)


class TestCreateSchedule:
    def test_create_linear(self):
        schedule = create_schedule("linear", 100)
        assert isinstance(schedule, LinearSchedule)

    def test_create_cosine(self):
        schedule = create_schedule("cosine", 100)
        assert isinstance(schedule, CosineSchedule)

    def test_create_sqrt(self):
        schedule = create_schedule("sqrt", 100)
        assert isinstance(schedule, SqrtSchedule)

    def test_case_insensitive(self):
        schedule = create_schedule("LINEAR", 100)
        assert isinstance(schedule, LinearSchedule)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_schedule("invalid", 100)

    def test_create_static(self):
        schedule = create_schedule("static", 100)
        assert isinstance(schedule, StaticSchedule)
        assert schedule.mask_rate == 0.15

    def test_create_static_custom_rate(self):
        schedule = create_schedule("static", 100, mask_rate=0.25)
        assert isinstance(schedule, StaticSchedule)
        assert schedule.mask_rate == 0.25


class TestStaticSchedule:
    """Tests for static masking rate schedule (MLM-style)."""

    def test_default_mask_rate(self):
        """Test default 15% mask rate."""
        schedule = StaticSchedule(num_timesteps=100)
        assert schedule.mask_rate == 0.15
        assert schedule.get_mask_rate(0) == 0.15
        assert schedule.get_mask_rate(50) == 0.15
        assert schedule.get_mask_rate(100) == 0.15

    def test_custom_mask_rate(self):
        """Test custom mask rate."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.30)
        assert schedule.mask_rate == 0.30
        assert schedule.get_mask_rate(42) == 0.30

    def test_tensor_input(self):
        """Test that tensor input returns tensor of correct shape."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.20)
        timesteps = torch.tensor([1, 25, 50, 75, 100])
        rates = schedule.get_mask_rate(timesteps)
        expected = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20])
        assert torch.allclose(rates, expected)
        assert rates.shape == timesteps.shape

    def test_timestep_ignored(self):
        """Verify that different timesteps produce same rate."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.15)
        rates = [schedule.get_mask_rate(t) for t in range(101)]
        assert all(r == 0.15 for r in rates)

    def test_sample_timesteps(self):
        """Verify sample_timesteps still works (for API compatibility)."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.15)
        timesteps = schedule.sample_timesteps(batch_size=10, device=torch.device("cpu"))
        assert timesteps.shape == (10,)
        assert (timesteps >= 1).all()
        assert (timesteps <= 100).all()

    def test_invalid_mask_rate_zero(self):
        """Test that mask_rate=0 raises ValueError."""
        with pytest.raises(ValueError):
            StaticSchedule(num_timesteps=100, mask_rate=0.0)

    def test_invalid_mask_rate_one(self):
        """Test that mask_rate=1 raises ValueError."""
        with pytest.raises(ValueError):
            StaticSchedule(num_timesteps=100, mask_rate=1.0)

    def test_invalid_mask_rate_negative(self):
        """Test that negative mask_rate raises ValueError."""
        with pytest.raises(ValueError):
            StaticSchedule(num_timesteps=100, mask_rate=-0.1)
