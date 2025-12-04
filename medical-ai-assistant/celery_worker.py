"""
Celery worker entry point.
Run this to start Celery workers for async task processing.

Usage:
    celery -A celery_worker.celery_app worker --loglevel=info
"""

from app.tasks.celery_tasks import celery_app

if __name__ == "__main__":
    celery_app.start()
