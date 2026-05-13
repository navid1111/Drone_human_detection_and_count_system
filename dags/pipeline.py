"""Compatibility entrypoint for Airflow DAG discovery.

Canonical DAG implementation lives in `src.pipeline.dags.pipeline`.
"""

from src.pipeline.dags.pipeline import branch_dag
