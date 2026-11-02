from .config import Config


class PermissionDenied(Exception):
    """Raised when a role is not allowed to perform an action."""


class AuthorizationService:
    """Role -> access level and role -> permission checks. Reads Config on every call."""

    def __init__(self, config: Config):
        self._config = config

    def role_level(self, role: str | None) -> int:
        """Unknown / missing roles get level 0 (public documents only)."""
        return self._config.get("auth.role_levels").get(role or "", 0)

    def access_value(self, access_level: str) -> int:
        """Unknown access labels are treated as MORE restricted than anything configured."""
        levels = self._config.get("auth.access_levels")
        return levels.get(access_level, max(levels.values()) + 1)

    def can_read(self, role: str | None, access_level: str) -> bool:
        return self.role_level(role) >= self.access_value(access_level)

    def has_permission(self, role: str | None, permission: str) -> bool:
        return (role or "") in self._config.get("auth.permissions").get(permission, [])

    def require(self, role: str | None, permission: str) -> None:
        if not self.has_permission(role, permission):
            raise PermissionDenied(f"role '{role}' lacks permission '{permission}'")
