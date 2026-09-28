"""Course-completion certificates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Certificate:
    certificate_id: str
    learner_id: str
    course_id: str
    issued_at: datetime
    verification_code: str


class CertificateIssuer:
    def issue(self, learner_id: str, course_id: str, *, completion: float) -> Certificate:
        if not 0.0 <= completion <= 1.0:
            raise ValueError("completion must be between 0 and 1.")
        if completion < 1.0:
            raise ValueError("a certificate requires complete course progress.")
        return Certificate(uuid4().hex, learner_id, course_id, datetime.now(timezone.utc), uuid4().hex)
