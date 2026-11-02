from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_HOLIDAYS_PATH = PROJECT_ROOT / "config" / "holidays.yaml"

DAY_NAMES = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
_MISSING = object()


class ConfigError(Exception):
    """Raised for missing, unknown or invalid configuration."""


# --- helpers ---------------------------------------------------------------

def _lookup(data: dict, path: str) -> Any:
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _is_hhmm(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    datetime.strptime(value, "%H:%M")  # raises ValueError if malformed
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive(value: Any) -> bool:
    return _is_number(value) and value > 0


def _unit_interval(value: Any) -> bool:
    return _is_number(value) and 0 <= value <= 1


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value)


# --- validation ------------------------------------------------------------

def validate_config(data: dict) -> None:
    """Raise ConfigError listing every problem found."""
    errors: list[str] = []

    def check(path: str, test, message: str) -> None:
        try:
            value = _lookup(data, path)
        except KeyError:
            errors.append(f"missing key: {path}")
            return
        try:
            ok = test(value)
        except Exception:
            ok = False
        if not ok:
            errors.append(f"{path}: {message}")

    # business calendar
    check("business.timezone", lambda v: bool(ZoneInfo(v)), "unknown timezone")
    check("business.working_days",
          lambda v: isinstance(v, list) and v and set(v) <= DAY_NAMES,
          f"must be a non-empty list of {sorted(DAY_NAMES)}")
    check("business.hours.start", _is_hhmm, "must be a quoted 'HH:MM' string")
    check("business.hours.end", _is_hhmm, "must be a quoted 'HH:MM' string")
    try:
        if _lookup(data, "business.hours.start") >= _lookup(data, "business.hours.end"):
            errors.append("business.hours: start must be before end")
    except (KeyError, TypeError):
        pass  # already reported above

    # SLA
    check("sla.count_mode", lambda v: v in ("calendar", "business_hours"),
          "must be 'calendar' or 'business_hours'")
    check("sla.warning_ratio", lambda v: _is_number(v) and 0 < v < 1,
          "must be between 0 and 1 (exclusive)")
    check("sla.response_minutes",
          lambda v: isinstance(v, dict) and v and all(_positive(x) for x in v.values()),
          "must be a non-empty mapping of priority -> positive minutes")

    # escalation
    check("escalation.repeated_negative_count",
          lambda v: isinstance(v, int) and v >= 2, "must be an integer >= 2")
    check("escalation.unresolved_negative_minutes", _positive, "must be positive")
    check("escalation.high_risk_min_confidence", _unit_interval, "must be between 0 and 1")
    check("escalation.high_risk_intents", _str_list, "must be a non-empty list of intent names")

    # confidence thresholds
    for key in ("sentiment_min", "language_min", "intent_min"):
        check(f"confidence.{key}", _unit_interval, "must be between 0 and 1")

    # languages
    check("languages.supported", _str_list, "must be a non-empty list of language codes")
    try:
        if _lookup(data, "languages.default") not in _lookup(data, "languages.supported"):
            errors.append("languages.default must be one of languages.supported")
    except (KeyError, TypeError):
        errors.append("languages.default / languages.supported missing or invalid")

    # session
    check("session.timeout_minutes", _positive, "must be positive")
    check("session.restore_hours", _positive, "must be positive")
    check("session.context_messages",
          lambda v: isinstance(v, int) and v >= 10,
          "must be an integer >= 10 (assignment requires at least 10)")

    # tickets / routing
    check("tickets.mandatory_fields", _str_list, "must be a non-empty list of field names")
    check("tickets.duplicate_similarity", _unit_interval, "must be between 0 and 1")
    check("tickets.known_products", lambda v: isinstance(v, list), "must be a list")
    check("tickets.priority_thresholds",
          lambda v: isinstance(v, dict) and {"critical", "high", "medium"} <= set(v)
          and v["critical"] > v["high"] > v["medium"] > 0,
          "needs critical > high > medium > 0")
    check("routing.max_workload", lambda v: isinstance(v, int) and v >= 1, "must be an integer >= 1")
    check("routing.skill_map",
          lambda v: isinstance(v, dict) and "general" in v, "must be a mapping with a 'general' key")

    # auth
    check("auth.role_levels", lambda v: isinstance(v, dict) and all(isinstance(x, int) for x in v.values()),
          "must map role -> integer level")
    check("auth.access_levels",
          lambda v: isinstance(v, dict) and v.get("public") == 0 and all(isinstance(x, int) for x in v.values()),
          "must map access level name -> integer, with public = 0")
    check("auth.permissions",
          lambda v: isinstance(v, dict) and all(isinstance(x, list) for x in v.values()),
          "must map permission -> list of roles")

    # RAG
    check("rag.top_k", lambda v: isinstance(v, int) and v >= 1, "must be an integer >= 1")
    check("rag.min_score", _unit_interval, "must be between 0 and 1")
    check("rag.clarification_margin", _unit_interval, "must be between 0 and 1")
    check("rag.min_support_ratio", _unit_interval, "must be between 0 and 1")

    # multimodal
    check("multimodal.processing_timeout_seconds", _positive, "must be positive")
    check("multimodal.file_retention_hours", _positive, "must be positive")
    check("multimodal.min_ocr_confidence", _unit_interval, "must be between 0 and 1")
    check("multimodal.min_text_chars", lambda v: isinstance(v, int) and v >= 0, "must be an integer >= 0")
    check("multimodal.max_file_mb", _positive, "must be positive")
    check("multimodal.allowed_extensions", _str_list, "must be a non-empty list")

    # KB pipeline
    check("kb_pipeline.update_time", _is_hhmm, "must be a quoted 'HH:MM' string")
    check("kb_pipeline.retry_delays_minutes",
          lambda v: isinstance(v, list) and v and all(_positive(x) for x in v),
          "must be a non-empty list of positive minutes")
    check("kb_pipeline.maintenance_window.start", _is_hhmm, "must be 'HH:MM'")
    check("kb_pipeline.maintenance_window.end", _is_hhmm, "must be 'HH:MM'")
    check("kb_pipeline.health_check_minutes", _positive, "must be positive")
    for key in ("min_accuracy", "min_grounding"):
        check(f"kb_pipeline.{key}", _unit_interval, "must be between 0 and 1")
    for key in ("max_accuracy_drop", "max_grounding_drop"):
        check(f"kb_pipeline.{key}", lambda v: _is_number(v) and v >= 0, "must be >= 0")

    # monitoring
    check("monitoring.window_minutes", _positive, "must be positive")
    check("monitoring.max_avg_latency_ms", _positive, "must be positive")
    check("monitoring.min_avg_confidence", _unit_interval, "must be between 0 and 1")
    check("monitoring.max_failures", lambda v: isinstance(v, int) and v >= 0, "must be an integer >= 0")
    check("monitoring.max_escalations", lambda v: isinstance(v, int) and v >= 0, "must be an integer >= 0")

    check("holidays",
          lambda v: isinstance(v, list)
          and all(isinstance(h, dict) and isinstance(h.get("date"), date) for h in v),
          "must be a list of {date: YYYY-MM-DD, name: ...}")

    if errors:
        raise ConfigError("Invalid configuration: " + "; ".join(errors))


# --- the Config object -----------------------------------------------------

class Config:
    """Validated, runtime-changeable configuration.

    Rules:
      * every change is validated before it is applied
      * a rejected change leaves the previous values untouched
      * callers get copies, so they cannot mutate internal state by accident
    """

    def __init__(self, data: dict, config_path: Path | None = None,
                 holidays_path: Path | None = None):
        validate_config(data)
        self._data = deepcopy(data)
        self._config_path = config_path
        self._holidays_path = holidays_path

    # -- reading
    def get(self, path: str, default: Any = _MISSING) -> Any:
        try:
            return deepcopy(_lookup(self._data, path))
        except KeyError:
            if default is _MISSING:
                raise ConfigError(f"Unknown config key: {path}") from None
            return default

    def to_dict(self) -> dict:
        return deepcopy(self._data)

    # -- runtime change
    def set(self, path: str, value: Any) -> None:
        """Change one existing setting, e.g. set('sla.warning_ratio', 0.5)."""
        parts = path.split(".")
        new_data = deepcopy(self._data)
        node = new_data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Unknown config key: {path}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError(f"Unknown config key: {path}")
        node[parts[-1]] = value
        validate_config(new_data)      # raises -> self._data is unchanged
        self._data = new_data

    # -- reload from disk
    def reload(self) -> None:
        """Re-read the files. If they are broken, keep the current values."""
        if self._config_path is None:
            raise ConfigError("This Config was not loaded from files")
        new_data = _read_files(self._config_path, self._holidays_path)
        validate_config(new_data)
        self._data = new_data


# --- loading ---------------------------------------------------------------

def _read_files(config_path: Path, holidays_path: Path | None) -> dict:
    try:
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {config_path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {exc}") from None

    holidays: list = []
    if holidays_path is not None and Path(holidays_path).exists():
        try:
            loaded = yaml.safe_load(Path(holidays_path).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Holidays file is not valid YAML: {exc}") from None
        holidays = loaded.get("holidays", [])
    data["holidays"] = holidays
    return data


def load_config(config_path: Path = DEFAULT_CONFIG_PATH,
                holidays_path: Path | None = DEFAULT_HOLIDAYS_PATH) -> Config:
    data = _read_files(config_path, holidays_path)
    return Config(data, config_path=config_path, holidays_path=holidays_path)
