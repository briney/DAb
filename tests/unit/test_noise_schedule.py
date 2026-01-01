"""Tests for noise schedules."""

import pytest
import torch

from dab.diffusion.noise_schedule import (
    CosineSchedule,
    LinearSchedule,
    SqrtSchedule,
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
