"""Celery application. Stage tasks are thin wrappers over pure pipeline functions."""

from celery import Celery

from voxint.config import get_settings

settings = get_settings()

app = Celery(
    "voxint",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[],  # task modules registered as pipeline stages land (P3)
)
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
