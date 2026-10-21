"""创建 research_tasks 表。

Revision ID: 0001_create_research_tasks
Revises:
Create Date: 2026-10-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_create_research_tasks"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# 状态枚举（使用 VARCHAR + CHECK 约束，不依赖 PostgreSQL 原生枚举类型）
status_enum = sa.Enum(
    "PENDING",
    "RUNNING",
    "WAITING_USER",
    "COMPLETED",
    "FAILED",
    name="researchtaskstatus",
    native_enum=False,
    length=32,
    validate_strings=True,
)


def upgrade() -> None:
    """创建研究任务表。"""
    op.create_table(
        "research_tasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("status", status_enum, nullable=False),
        sa.Column("final_report", sa.Text(), nullable=True),
        sa.Column("clarification_question", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_research_tasks_task_id", "research_tasks", ["task_id"], unique=True)
    op.create_index("ix_research_tasks_thread_id", "research_tasks", ["thread_id"], unique=True)
    op.create_index("ix_research_tasks_status", "research_tasks", ["status"], unique=False)
    op.create_index(
        "ix_research_tasks_created_at", "research_tasks", ["created_at"], unique=False
    )


def downgrade() -> None:
    """删除研究任务表。"""
    op.drop_index("ix_research_tasks_created_at", table_name="research_tasks")
    op.drop_index("ix_research_tasks_status", table_name="research_tasks")
    op.drop_index("ix_research_tasks_thread_id", table_name="research_tasks")
    op.drop_index("ix_research_tasks_task_id", table_name="research_tasks")
    op.drop_table("research_tasks")
