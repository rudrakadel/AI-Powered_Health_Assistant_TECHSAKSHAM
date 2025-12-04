"""
Celery task definitions.
"""

from .celery_tasks import query_task

__all__ = ["query_task"]
