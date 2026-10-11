from datetime import datetime, timedelta, timezone

import pytest

from a_real_time_gen_ai_customer_service_bot.core.clock import (
    Clock,
    FakeClock,
    SystemClock,
)

START = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)


# --- Normal cases ---------------------------------------------------------

def test_system_clock_returns_timezone_aware_datetime():
    now = SystemClock().now()
    assert now.tzinfo is not None


def test_system_clock_and_fake_clock_are_clocks():
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock(START), Clock)


def test_fake_clock_starts_at_given_time(fake_clock):
    assert fake_clock.now() == START


def test_advance_moves_time_forward(fake_clock):
    fake_clock.advance(minutes=16)
    assert fake_clock.now() == START + timedelta(minutes=16)


def test_time_does_not_move_on_its_own(fake_clock):
    assert fake_clock.now() == fake_clock.now()


# --- Simulated time changes ----------------------------------------------

def test_set_can_jump_forward(fake_clock):
    target = datetime(2026, 12, 25, 2, 0, tzinfo=timezone.utc)
    fake_clock.set(target)
    assert fake_clock.now() == target


def test_set_can_jump_backward(fake_clock):
    target = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    fake_clock.set(target)
    assert fake_clock.now() == target


def test_multiple_advances_accumulate(fake_clock):
    fake_clock.advance(minutes=10)
    fake_clock.advance(minutes=5)
    fake_clock.advance(hours=1)
    assert fake_clock.now() == START + timedelta(hours=1, minutes=15)


# --- Boundary cases -------------------------------------------------------

def test_advance_zero_leaves_time_unchanged(fake_clock):
    fake_clock.advance(minutes=0)
    assert fake_clock.now() == START


def test_advance_across_midnight(fake_clock):
    fake_clock.advance(hours=14)  # 10:00 + 14h = 00:00 next day
    assert fake_clock.now() == datetime(2026, 10, 13, 0, 0, tzinfo=timezone.utc)


# --- Invalid input --------------------------------------------------------

def test_advance_negative_raises(fake_clock):
    with pytest.raises(ValueError):
        fake_clock.advance(minutes=-5)


def test_naive_start_time_raises():
    with pytest.raises(ValueError):
        FakeClock(datetime(2026, 10, 12, 10, 0))  # no tzinfo


def test_set_naive_time_raises(fake_clock):
    with pytest.raises(ValueError):
        fake_clock.set(datetime(2026, 10, 12, 10, 0))


def test_failed_advance_does_not_change_time(fake_clock):
    with pytest.raises(ValueError):
        fake_clock.advance(minutes=-5)
    assert fake_clock.now() == START


# --- Isolation ------------------------------------------------------------

def test_two_fake_clocks_are_independent():
    a = FakeClock(START)
    b = FakeClock(START)
    a.advance(hours=3)
    assert b.now() == START
