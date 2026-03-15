"""Initial schema
Revision ID: 001_initial
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'training_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('trigger', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='PENDING'),
        sa.Column('mlflow_run_id', sa.String(255), nullable=True),
        sa.Column('mlflow_experiment_id', sa.String(255), nullable=True),
        sa.Column('triggered_by', sa.String(100), server_default='system'),
        sa.Column('feature_store_version', sa.String(50), server_default='v2.0.0'),
        sa.Column('n_training_pairs', sa.Integer(), server_default='0'),
        sa.Column('n_positive', sa.Integer(), server_default='0'),
        sa.Column('n_negative', sa.Integer(), server_default='0'),
        sa.Column('transformer_f1', sa.Float(), nullable=True),
        sa.Column('gnn_f1', sa.Float(), nullable=True),
        sa.Column('xgb_f1', sa.Float(), nullable=True),
        sa.Column('ensemble_f1', sa.Float(), nullable=True),
        sa.Column('ensemble_precision', sa.Float(), nullable=True),
        sa.Column('ensemble_recall', sa.Float(), nullable=True),
        sa.Column('ensemble_auc', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('model_artifact_uri', sa.Text(), nullable=True),
        sa.Column('promoted_to_production', sa.Boolean(), server_default='false'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('data_collected_at', sa.DateTime(), nullable=True),
        sa.Column('features_extracted_at', sa.DateTime(), nullable=True),
        sa.Column('training_completed_at', sa.DateTime(), nullable=True),
        sa.Column('evaluation_completed_at', sa.DateTime(), nullable=True),
        sa.Column('registered_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_tr_status', 'training_runs', ['status'])
    op.create_index('ix_tr_trigger', 'training_runs', ['trigger'])
    op.create_index('ix_tr_created', 'training_runs', ['created_at'])

    op.create_table(
        'trained_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('mlflow_model_name', sa.String(255)),
        sa.Column('mlflow_version', sa.String(50), nullable=True),
        sa.Column('model_type', sa.String(50), server_default='ENSEMBLE'),
        sa.Column('status', sa.String(50), server_default='STAGING'),
        sa.Column('artifact_uri', sa.Text(), nullable=True),
        sa.Column('f1_score', sa.Float(), server_default='0.0'),
        sa.Column('precision', sa.Float(), server_default='0.0'),
        sa.Column('recall', sa.Float(), server_default='0.0'),
        sa.Column('auc_roc', sa.Float(), server_default='0.0'),
        sa.Column('inference_p95_ms', sa.Float(), server_default='0.0'),
        sa.Column('traffic_pct', sa.Integer(), server_default='0'),
        sa.Column('feature_store_version', sa.String(50), server_default='v2.0.0'),
        sa.Column('rollback_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime()),
        sa.Column('promoted_at', sa.DateTime(), nullable=True),
        sa.Column('rolled_back_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_tm_status', 'trained_models', ['status'])
    op.create_index('ix_tm_run_id', 'trained_models', ['run_id'])


def downgrade():
    op.drop_table('trained_models')
    op.drop_table('training_runs')
