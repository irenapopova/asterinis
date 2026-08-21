from dataclasses import dataclass, field


@dataclass(slots=True)
class PermissionPolicy:
    allowed_tools: set[str] = field(default_factory=set)
    allowed_providers: set[str] = field(default_factory=set)

    def can_use_tool(self, name: str) -> bool:
        return name in self.allowed_tools

    def can_use_provider(self, name: str) -> bool:
        return name in self.allowed_providers

    def allow_tool(self, name: str) -> None:
        name = name.strip()

        if not name:
            raise ValueError("Tool name cannot be empty.")

        self.allowed_tools.add(name)

    def allow_provider(self, name: str) -> None:
        name = name.strip()

        if not name:
            raise ValueError("Provider name cannot be empty.")

        self.allowed_providers.add(name)

    def revoke_tool(self, name: str) -> None:
        self.allowed_tools.discard(name)

    def revoke_provider(self, name: str) -> None:
        self.allowed_providers.discard(name)