from datetime import date, datetime, timezone

import pytest

from a_real_time_gen_ai_customer_service_bot.core.business_calendar import BusinessCalendar
from conftest import ist

# 2026: Mon 12, Fri 16, Sat 17, Sun 18, Mon 19, Tue 20 = test holiday, Wed 21.


@pytest.fixture
def calendar(config):
    config.set("holidays", [{"date": date(2026, 10, 20), "name": "Test holiday"}])
    return BusinessCalendar(config)


def test_open_boundaries(calendar):
    assert calendar.is_open(ist(12, 9, 0)) and calendar.is_open(ist(12, 17, 59))
    assert not calendar.is_open(ist(12, 8, 59)) and not calendar.is_open(ist(12, 18, 0))


def test_closed_weekend_and_holiday(calendar):
    assert not calendar.is_open(ist(17, 10)) and not calendar.is_open(ist(18, 10))
    assert not calendar.is_open(ist(20, 10)) and not calendar.is_working_day(ist(20, 10))


def test_utc_input_converted_to_business_timezone(calendar):
    assert calendar.is_open(datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc))
    assert not calendar.is_open(datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc))


def test_naive_datetime_rejected(calendar):
    with pytest.raises(ValueError):
        calendar.is_open(datetime(2026, 10, 12, 10, 0))


def test_next_open(calendar):
    assert calendar.next_open(ist(12, 10)) == ist(12, 10)
    assert calendar.next_open(ist(12, 8)) == ist(12, 9)
    assert calendar.next_open(ist(16, 19)) == ist(19, 9)
    assert calendar.next_open(ist(19, 18, 0)) == ist(21, 9)


def test_next_working_day_start(calendar):
    assert calendar.next_working_day_start(ist(12, 10)) == ist(13, 9)
    assert calendar.next_working_day_start(ist(16, 10)) == ist(19, 9)
    assert calendar.next_working_day_start(ist(17, 10)) == ist(19, 9)
    assert calendar.next_working_day_start(ist(19, 10)) == ist(21, 9)


def test_calendar_mode_deadlines(calendar):
    assert calendar.add_minutes(ist(12, 10), 150) == ist(12, 12, 30)
    assert calendar.add_minutes(ist(16, 17), 480) == ist(19, 1)      # skips weekend
    assert calendar.add_minutes(ist(19, 23), 120) == ist(21, 1)      # skips holiday
    assert calendar.add_minutes(ist(17, 10), 60) == ist(19, 1)       # starts on weekend
    assert calendar.add_minutes(ist(17, 10), 0) == ist(17, 10)


def test_negative_minutes_rejected(calendar):
    with pytest.raises(ValueError):
        calendar.add_minutes(ist(12, 10), -1)


def test_calendar_mode_elapsed(calendar):
    assert calendar.elapsed_minutes(ist(12, 10), ist(12, 12, 30)) == 150
    assert calendar.elapsed_minutes(ist(16, 17), ist(19, 10)) == 17 * 60
    assert calendar.elapsed_minutes(ist(19, 22), ist(21, 1)) == 180
    assert calendar.elapsed_minutes(ist(12, 10), ist(12, 10)) == 0
    with pytest.raises(ValueError):
        calendar.elapsed_minutes(ist(12, 10), ist(12, 9))


def test_elapsed_and_add_are_consistent(calendar):
    start = ist(16, 17)
    assert calendar.elapsed_minutes(start, calendar.add_minutes(start, 600)) == pytest.approx(600)


def test_business_hours_mode(calendar, config):
    config.set("sla.count_mode", "business_hours")
    assert calendar.add_minutes(ist(16, 17), 480) == ist(19, 16)
    assert calendar.add_minutes(ist(17, 10), 60) == ist(19, 10)
    assert calendar.elapsed_minutes(ist(16, 17), ist(19, 10)) == 120


def test_runtime_config_changes_move_deadlines(calendar, config):
    before = calendar.add_minutes(ist(16, 17), 480)
    config.set("holidays", [{"date": date(2026, 10, 19), "name": "New holiday"}])
    assert (before, calendar.add_minutes(ist(16, 17), 480)) == (ist(19, 1), ist(20, 1))
    config.set("business.working_days", ["mon", "tue", "wed", "thu", "fri", "sat"])
    assert calendar.is_open(ist(17, 10))
    config.set("business.hours.end", "20:00")
    assert calendar.is_open(ist(12, 19))
