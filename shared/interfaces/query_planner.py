"""Stable QueryPlanner interface shared by all online planner providers."""

from abc import ABC, abstractmethod

from shared.schemas.online import TaskType, UnifiedQueryPlan


class QueryPlanner(ABC):
    @abstractmethod
    def plan(self, text: str, task_type: TaskType) -> UnifiedQueryPlan:
        """Return a valid plan or raise so the planner chain can fall back."""
