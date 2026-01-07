"""Tests for noise schedules."""

import pytest
import torch

from dab.diffusion.noise_schedule import (
    CosineSchedule,
    LinearSchedule,
    PowerSchedule,
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


class TestPowerSchedule:
    """Tests for power schedule (variable average masking rate)."""

    def test_endpoints(self):
        """Test that endpoints are correct."""
        schedule = PowerSchedule(num_timesteps=100, power=2.0)
        assert schedule.get_mask_rate(0) == pytest.approx(0.0, abs=1e-6)
        assert schedule.get_mask_rate(100) == pytest.approx(1.0, abs=1e-6)

    def test_power_1_equals_linear(self):
        """Test that power=1 is equivalent to linear schedule."""
        power_schedule = PowerSchedule(num_timesteps=100, power=1.0)
        linear_schedule = LinearSchedule(num_timesteps=100)
        for t in [0, 25, 50, 75, 100]:
            assert power_schedule.get_mask_rate(t) == pytest.approx(
                linear_schedule.get_mask_rate(t), abs=1e-6
            )

    def test_monotonic(self):
        """Test that mask rate increases monotonically."""
        schedule = PowerSchedule(num_timesteps=100, power=4.0)
        prev_rate = 0.0
        for t in range(1, 101):
            rate = schedule.get_mask_rate(t)
            assert rate > prev_rate
            prev_rate = rate

    def test_higher_power_lower_rates(self):
        """Test that higher power results in lower mask rates for same timestep."""
        schedule_p2 = PowerSchedule(num_timesteps=100, power=2.0)
        schedule_p4 = PowerSchedule(num_timesteps=100, power=4.0)
        # At any non-zero timestep < T, higher power should give lower rate
        for t in [25, 50, 75]:
            assert schedule_p4.get_mask_rate(t) < schedule_p2.get_mask_rate(t)

    def test_tensor_input(self):
        """Test that tensor input returns tensor of correct shape."""
        schedule = PowerSchedule(num_timesteps=100, power=2.0)
        timesteps = torch.tensor([0, 25, 50, 100])
        rates = schedule.get_mask_rate(timesteps)
        # power=2: (t/100)^2
        expected = torch.tensor([0.0, 0.0625, 0.25, 1.0])
        assert torch.allclose(rates, expected, atol=1e-6)

    def test_default_power(self):
        """Test that default power is 4.0."""
        schedule = PowerSchedule(num_timesteps=100)
        assert schedule.power == 4.0

    def test_invalid_power_zero(self):
        """Test that power=0 raises ValueError."""
        with pytest.raises(ValueError):
            PowerSchedule(num_timesteps=100, power=0.0)

    def test_invalid_power_negative(self):
        """Test that negative power raises ValueError."""
        with pytest.raises(ValueError):
            PowerSchedule(num_timesteps=100, power=-1.0)

    def test_average_mask_rate_power_4(self):
        """Test approximate average mask rate for power=4."""
        schedule = PowerSchedule(num_timesteps=1000, power=4.0)
        # Sample many timesteps uniformly and compute average mask rate
        timesteps = torch.arange(1, 1001)
        rates = schedule.get_mask_rate(timesteps)
        avg_rate = rates.mean().item()
        # Theoretical average for (t/T)^4 with uniform sampling is 1/5 = 0.2
        assert avg_rate == pytest.approx(0.2, abs=0.01)

    def test_average_mask_rate_power_5(self):
        """Test approximate average mask rate for power=5."""
        schedule = PowerSchedule(num_timesteps=1000, power=5.0)
        timesteps = torch.arange(1, 1001)
        rates = schedule.get_mask_rate(timesteps)
        avg_rate = rates.mean().item()
        # Theoretical average for (t/T)^5 with uniform sampling is 1/6 ≈ 0.167
        assert avg_rate == pytest.approx(1 / 6, abs=0.01)


class TestCreateSchedulePower:
    """Tests for power schedule creation via factory."""

    def test_create_power(self):
        schedule = create_schedule("power", 100)
        assert isinstance(schedule, PowerSchedule)
        assert schedule.power == 4.0  # default

    def test_create_power_custom(self):
        schedule = create_schedule("power", 100, power=3.0)
        assert isinstance(schedule, PowerSchedule)
        assert schedule.power == 3.0


class TestCurriculumSampling:
    """Tests for curriculum learning timestep sampling."""

    def test_no_curriculum_default(self):
        """Test that default behavior (no curriculum) samples full range."""
        schedule = LinearSchedule(num_timesteps=100)
        # Without training_progress, should sample from [1, 100]
        timesteps = schedule.sample_timesteps(batch_size=1000, device=torch.device("cpu"))
        assert timesteps.min() >= 1
        assert timesteps.max() <= 100
        # With 1000 samples, we should see values near both ends
        assert timesteps.max() >= 90  # Very likely to hit high values

    def test_curriculum_at_start(self):
        """Test curriculum sampling at beginning of training."""
        schedule = LinearSchedule(num_timesteps=100)
        # At progress=0 with curriculum_start=0.1, max_timestep = 100 * 0.1 = 10
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=0.0,
            curriculum_start=0.1,
        )
        assert timesteps.min() >= 1
        assert timesteps.max() <= 10

    def test_curriculum_at_midpoint(self):
        """Test curriculum sampling at middle of training."""
        schedule = LinearSchedule(num_timesteps=100)
        # At progress=0.5 with curriculum_start=0.1:
        # effective_progress = 0.1 + (1-0.1)*0.5 = 0.1 + 0.45 = 0.55
        # max_timestep = 100 * 0.55 = 55
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=0.5,
            curriculum_start=0.1,
        )
        assert timesteps.min() >= 1
        assert timesteps.max() <= 55

    def test_curriculum_at_end(self):
        """Test curriculum sampling at end of training (full range)."""
        schedule = LinearSchedule(num_timesteps=100)
        # At progress=1.0, should sample from full range
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=1.0,
            curriculum_start=0.1,
        )
        assert timesteps.min() >= 1
        assert timesteps.max() <= 100
        # With 1000 samples at full range, should hit high values
        assert timesteps.max() >= 90

    def test_curriculum_none_progress_full_range(self):
        """Test that None training_progress gives full range (backwards compat)."""
        schedule = LinearSchedule(num_timesteps=100)
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=None,
            curriculum_start=0.1,
        )
        # Should behave as if no curriculum
        assert timesteps.max() <= 100
        assert timesteps.max() >= 90  # Very likely

    def test_curriculum_different_start_values(self):
        """Test curriculum with different curriculum_start values."""
        schedule = LinearSchedule(num_timesteps=100)

        # curriculum_start=0.2 at progress=0: max = 100 * 0.2 = 20
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=0.0,
            curriculum_start=0.2,
        )
        assert timesteps.max() <= 20

        # curriculum_start=0.5 at progress=0: max = 100 * 0.5 = 50
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=0.0,
            curriculum_start=0.5,
        )
        assert timesteps.max() <= 50

    def test_curriculum_works_with_power_schedule(self):
        """Test that curriculum sampling works with PowerSchedule."""
        schedule = PowerSchedule(num_timesteps=100, power=4.0)
        timesteps = schedule.sample_timesteps(
            batch_size=1000,
            device=torch.device("cpu"),
            training_progress=0.0,
            curriculum_start=0.1,
        )
        assert timesteps.min() >= 1
        assert timesteps.max() <= 10


class TestNELBOWeights:
    """Tests for NELBO weight computation across all schedule types."""

    def test_linear_nelbo_weight_formula(self):
        """Test linear schedule NELBO weight: T/t."""
        schedule = LinearSchedule(num_timesteps=100)
        # At t=1: weight = 100/1 = 100
        assert schedule.get_nelbo_weight(1) == pytest.approx(100.0, rel=1e-6)
        # At t=50: weight = 100/50 = 2
        assert schedule.get_nelbo_weight(50) == pytest.approx(2.0, rel=1e-6)
        # At t=100: weight = 100/100 = 1
        assert schedule.get_nelbo_weight(100) == pytest.approx(1.0, rel=1e-6)

    def test_linear_nelbo_weight_tensor(self):
        """Test linear schedule NELBO weight with tensor input."""
        schedule = LinearSchedule(num_timesteps=100)
        timesteps = torch.tensor([1, 10, 50, 100])
        weights = schedule.get_nelbo_weight(timesteps)
        expected = torch.tensor([100.0, 10.0, 2.0, 1.0])
        assert torch.allclose(weights, expected, rtol=1e-6)

    def test_cosine_nelbo_weight_positive(self):
        """Test cosine schedule NELBO weights are positive."""
        schedule = CosineSchedule(num_timesteps=100)
        for t in [1, 25, 50, 75, 100]:
            weight = schedule.get_nelbo_weight(t)
            assert weight > 0

    def test_cosine_nelbo_weight_decreasing(self):
        """Test cosine schedule NELBO weights decrease with timestep."""
        schedule = CosineSchedule(num_timesteps=100)
        prev_weight = float("inf")
        for t in [1, 10, 25, 50, 75, 100]:
            weight = schedule.get_nelbo_weight(t)
            assert weight < prev_weight
            prev_weight = weight

    def test_sqrt_nelbo_weight_formula(self):
        """Test sqrt schedule NELBO weight: T/(2t)."""
        schedule = SqrtSchedule(num_timesteps=100)
        # At t=1: weight = 100/(2*1) = 50
        assert schedule.get_nelbo_weight(1) == pytest.approx(50.0, rel=1e-6)
        # At t=50: weight = 100/(2*50) = 1
        assert schedule.get_nelbo_weight(50) == pytest.approx(1.0, rel=1e-6)
        # At t=100: weight = 100/(2*100) = 0.5
        assert schedule.get_nelbo_weight(100) == pytest.approx(0.5, rel=1e-6)

    def test_power_nelbo_weight_formula(self):
        """Test power schedule NELBO weight: p*T/t."""
        schedule = PowerSchedule(num_timesteps=100, power=4.0)
        # At t=1: weight = 4*100/1 = 400
        assert schedule.get_nelbo_weight(1) == pytest.approx(400.0, rel=1e-6)
        # At t=50: weight = 4*100/50 = 8
        assert schedule.get_nelbo_weight(50) == pytest.approx(8.0, rel=1e-6)
        # At t=100: weight = 4*100/100 = 4
        assert schedule.get_nelbo_weight(100) == pytest.approx(4.0, rel=1e-6)

    def test_power_nelbo_weight_different_powers(self):
        """Test that higher power gives proportionally higher weights."""
        schedule_p2 = PowerSchedule(num_timesteps=100, power=2.0)
        schedule_p4 = PowerSchedule(num_timesteps=100, power=4.0)
        # At same timestep, weight ratio should equal power ratio
        for t in [10, 50, 100]:
            ratio = schedule_p4.get_nelbo_weight(t) / schedule_p2.get_nelbo_weight(t)
            assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_static_nelbo_weight_constant(self):
        """Test static schedule NELBO weight is always 1.0."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.15)
        for t in [1, 25, 50, 75, 100]:
            assert schedule.get_nelbo_weight(t) == pytest.approx(1.0, rel=1e-6)

    def test_static_nelbo_weight_tensor(self):
        """Test static schedule returns tensor of ones for tensor input."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.15)
        timesteps = torch.tensor([1, 25, 50, 75, 100])
        weights = schedule.get_nelbo_weight(timesteps)
        expected = torch.ones(5)
        assert torch.allclose(weights, expected)


class TestNELBOWeightNormalization:
    """Tests for NELBO weight normalization options."""

    def test_no_normalization(self):
        """Test that normalize=None returns raw weights."""
        schedule = LinearSchedule(num_timesteps=100)
        raw_weight = schedule.get_nelbo_weight(1)
        normalized_weight = schedule.get_normalized_nelbo_weight(1, normalize=None)
        assert raw_weight == normalized_weight

    def test_clip_normalization(self):
        """Test clip normalization caps weights at max value."""
        schedule = LinearSchedule(num_timesteps=100)
        # At t=1, raw weight = 100, should be clipped to 10
        clipped = schedule.get_normalized_nelbo_weight(1, normalize="clip", clip_max=10.0)
        assert clipped == pytest.approx(10.0, rel=1e-6)
        # At t=50, raw weight = 2, should not be clipped
        not_clipped = schedule.get_normalized_nelbo_weight(50, normalize="clip", clip_max=10.0)
        assert not_clipped == pytest.approx(2.0, rel=1e-6)

    def test_clip_normalization_tensor(self):
        """Test clip normalization with tensor input."""
        schedule = LinearSchedule(num_timesteps=100)
        timesteps = torch.tensor([1, 10, 50, 100])
        clipped = schedule.get_normalized_nelbo_weight(timesteps, normalize="clip", clip_max=5.0)
        # Raw: [100, 10, 2, 1] -> Clipped: [5, 5, 2, 1]
        expected = torch.tensor([5.0, 5.0, 2.0, 1.0])
        assert torch.allclose(clipped, expected, rtol=1e-6)

    def test_minmax_normalization(self):
        """Test min-max normalization scales to [0, 1]."""
        schedule = LinearSchedule(num_timesteps=100)
        # For linear: weights range from 100 (t=1) to 1 (t=100)
        # At t=1 (max weight): should be 1.0
        max_normalized = schedule.get_normalized_nelbo_weight(1, normalize="minmax")
        assert max_normalized == pytest.approx(1.0, rel=1e-6)
        # At t=100 (min weight): should be 0.0
        min_normalized = schedule.get_normalized_nelbo_weight(100, normalize="minmax")
        assert min_normalized == pytest.approx(0.0, rel=1e-6)

    def test_minmax_normalization_tensor(self):
        """Test min-max normalization with tensor input."""
        schedule = LinearSchedule(num_timesteps=100)
        timesteps = torch.tensor([1, 100])
        normalized = schedule.get_normalized_nelbo_weight(timesteps, normalize="minmax")
        # t=1 maps to 1.0, t=100 maps to 0.0
        expected = torch.tensor([1.0, 0.0])
        assert torch.allclose(normalized, expected, rtol=1e-6)

    def test_invalid_normalization(self):
        """Test that invalid normalization method raises error."""
        schedule = LinearSchedule(num_timesteps=100)
        with pytest.raises(ValueError, match="Unknown normalization"):
            schedule.get_normalized_nelbo_weight(50, normalize="invalid")

    def test_static_schedule_normalization(self):
        """Test normalization with static schedule (constant weights)."""
        schedule = StaticSchedule(num_timesteps=100, mask_rate=0.15)
        # With constant weights, minmax normalization returns same value
        # (because w_max - w_min = 0, so we return raw weights)
        normalized = schedule.get_normalized_nelbo_weight(50, normalize="minmax")
        assert normalized == pytest.approx(1.0, rel=1e-6)
