"""创建 research_task_events 表（研究过程事件日志）。

Revision ID: 0002_create_research_task_events
Revises: 0001_create_research_tasks
Create Date: 2026-10-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_create_research_task_events"
down_revision: Union[str, None] = "0001_create_research_tasks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建研究过程事件表。

    该表是不可变的事件日志：SSE 以它作为“可回放的历史”，
    因此重新进入任务详情页 / 进程重启后仍能恢复研究时间线。
    """
    op.create_table(
        "research_task_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("event", sa.String(length=64), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_task_events_task_id", "research_task_events", ["task_id"], unique=False
    )
    op.create_index(
        "ix_research_task_events_created_at",
        "research_task_events",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    """删除研究过程事件表。"""
    op.drop_index("ix_research_task_events_created_at", table_name="research_task_events")
    op.drop_index("ix_research_task_events_task_id", table_name="research_task_events")
    op.drop_table("research_task_events")
