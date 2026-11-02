from datetime import date

import pytest
import yaml

from a_real_time_gen_ai_customer_service_bot.core.config import Config, ConfigError, load_config


def test_default_file_loads_and_is_valid(config):
    assert config.get("business.timezone") == "Asia/Kolkata"
    assert config.get("sla.warning_ratio") == 0.75
    assert config.get("kb_pipeline.retry_delays_minutes") == [15, 30, 60]


def test_holidays_are_loaded_as_dates(config):
    assert all(isinstance(h["date"], date) for h in config.get("holidays"))


def test_unknown_key_raises_or_defaults(config):
    with pytest.raises(ConfigError):
        config.get("does.not.exist")
    assert config.get("does.not.exist", default=42) == 42


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_missing_required_section_raises(config):
    data = config.to_dict()
    del data["sla"]
    with pytest.raises(ConfigError):
        Config(data)


@pytest.mark.parametrize("path,value", [
    ("business.timezone", "Mars/Olympus"),
    ("business.hours.start", "19:00"),
    ("session.context_messages", 5),
    ("sla.count_mode", "whenever"),
    ("sla.warning_ratio", 1.5),
    ("confidence.language_min", 2),
    ("languages.default", "xx"),
])
def test_invalid_values_rejected_and_old_value_kept(config, path, value):
    before = config.get(path)
    with pytest.raises(ConfigError):
        config.set(path, value)
    assert config.get(path) == before


def test_set_changes_value_and_unknown_key_rejected(config):
    config.set("sla.warning_ratio", 0.5)
    assert config.get("sla.warning_ratio") == 0.5
    with pytest.raises(ConfigError):
        config.set("sla.nonsense", 1)


def test_mutating_returned_value_does_not_change_config(config):
    delays = config.get("kb_pipeline.retry_delays_minutes")
    delays.append(999)
    assert config.get("kb_pipeline.retry_delays_minutes") == [15, 30, 60]


def test_two_configs_are_independent():
    a, b = load_config(), load_config()
    a.set("sla.warning_ratio", 0.5)
    assert b.get("sla.warning_ratio") == 0.75


def _write_config(tmp_path, ratio):
    data = load_config().to_dict()
    data.pop("holidays")
    data["sla"]["warning_ratio"] = ratio
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_reload_picks_up_file_change(tmp_path):
    path = _write_config(tmp_path, 0.75)
    cfg = load_config(path, holidays_path=None)
    _write_config(tmp_path, 0.9)
    cfg.reload()
    assert cfg.get("sla.warning_ratio") == 0.9


def test_broken_or_invalid_reload_keeps_previous_config(tmp_path):
    path = _write_config(tmp_path, 0.75)
    cfg = load_config(path, holidays_path=None)
    path.write_text("sla: [this is : not valid")
    with pytest.raises(ConfigError):
        cfg.reload()
    _write_config(tmp_path, 7)
    with pytest.raises(ConfigError):
        cfg.reload()
    assert cfg.get("sla.warning_ratio") == 0.75
