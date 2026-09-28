"""Payment-provider-independent course entitlements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CourseEntitlement:
    learner_id: str
    course_id: str
    status: str
    provider_customer_id: str
    expires_at: datetime | None = None


class EntitlementManager:
    def __init__(self) -> None:
        self._entitlements: dict[tuple[str, str], CourseEntitlement] = {}

    def apply_webhook(self, entitlement: CourseEntitlement) -> None:
        if entitlement.status not in {"active", "canceled", "expired"}:
            raise ValueError("unsupported entitlement status.")
        self._entitlements[(entitlement.learner_id, entitlement.course_id)] = entitlement

    def can_access(self, learner_id: str, course_id: str) -> bool:
        entitlement = self._entitlements.get((learner_id, course_id))
        return entitlement is not None and entitlement.status == "active"
