"""SQLAlchemy models — fresh Base, no upstream lineage.

The v1 schema (~9 tables: media_items, pipeline_runs, stage_runs, audio_artifacts,
audio_chunks, transcript_segments, speakers, speaker_embeddings vector(192),
speaker_assignments, adjudication_decisions) lands in P1 alongside alembic revision 0001.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
