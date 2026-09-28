"""Project-based learning assignments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProjectMilestone:
    milestone_id: str
    title: str
    description: str
    concepts: tuple[str, ...] = ()


@dataclass(slots=True)
class ProjectAssignment:
    project_id: str
    title: str
    brief: str
    milestones: tuple[ProjectMilestone, ...]
    completed_milestones: set[str] = field(default_factory=set)

    def complete(self, milestone_id: str) -> None:
        if milestone_id not in {item.milestone_id for item in self.milestones}:
            raise ValueError("unknown project milestone.")
        self.completed_milestones.add(milestone_id)

    @property
    def completion(self) -> float:
        return len(self.completed_milestones) / len(self.milestones) if self.milestones else 0.0
