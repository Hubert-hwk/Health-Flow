"""Initialise the canonical HealthFlow Neo4j ontology from environment config."""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CONSTRAINTS = (
    "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (n:Symptom) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT department_id IF NOT EXISTS FOR (n:Department) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT examination_id IF NOT EXISTS FOR (n:Examination) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT report_id IF NOT EXISTS FOR (n:Report) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT guideline_id IF NOT EXISTS FOR (n:Guideline) REQUIRE n.id IS UNIQUE",
)

INDEXES = (
    "CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name)",
    "CREATE INDEX symptom_name IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
    "CREATE INDEX drug_name IF NOT EXISTS FOR (n:Drug) ON (n.name)",
    "CREATE INDEX department_name IF NOT EXISTS FOR (n:Department) ON (n.name)",
    "CREATE INDEX guideline_name IF NOT EXISTS FOR (n:Guideline) ON (n.name)",
)


def _run_statements(driver: Any, statements: tuple[str, ...]) -> None:
    with driver.session() as session:
        for statement in statements:
            session.run(statement)


def create_constraints(driver: Any) -> None:
    _run_statements(driver, CONSTRAINTS)


def create_indexes(driver: Any) -> None:
    _run_statements(driver, INDEXES)


def load_medical_ontology(driver: Any) -> None:
    """Load a small canonical ontology used for local GraphRAG smoke tests."""

    departments = {
        "dept_endo": "内分泌科",
        "dept_cardio": "心内科",
        "dept_gi": "消化科",
        "dept_respiratory": "呼吸科",
    }
    symptoms = {
        "sym_chest_pain": ("胸痛", "high"),
        "sym_cough": ("咳嗽", "medium"),
        "sym_abdominal_pain": ("腹痛", "medium"),
        "sym_fatigue": ("乏力", "low"),
    }
    diseases = {
        "disease_hypertension": ("高血压", "dept_cardio"),
        "disease_diabetes_type2": ("2型糖尿病", "dept_endo"),
        "disease_gerd": ("胃食管反流病", "dept_gi"),
        "disease_asthma": ("哮喘", "dept_respiratory"),
    }
    disease_symptoms = (
        ("disease_hypertension", "sym_chest_pain"),
        ("disease_diabetes_type2", "sym_fatigue"),
        ("disease_gerd", "sym_abdominal_pain"),
        ("disease_asthma", "sym_cough"),
    )
    symptom_departments = (
        ("sym_chest_pain", "dept_cardio"),
        ("sym_cough", "dept_respiratory"),
        ("sym_abdominal_pain", "dept_gi"),
        ("sym_fatigue", "dept_endo"),
    )

    with driver.session() as session:
        for node_id, name in departments.items():
            session.run(
                "MERGE (n:Department {id: $id}) SET n.name = $name",
                id=node_id,
                name=name,
            )
        for node_id, (name, severity) in symptoms.items():
            session.run(
                "MERGE (n:Symptom {id: $id}) SET n.name = $name, n.severity = $severity",
                id=node_id,
                name=name,
                severity=severity,
            )
        for node_id, (name, department_id) in diseases.items():
            session.run(
                """
                MERGE (d:Disease {id: $id}) SET d.name = $name
                WITH d MATCH (de:Department {id: $department_id})
                MERGE (d)-[:BELONGS_TO]->(de)
                """,
                id=node_id,
                name=name,
                department_id=department_id,
            )
        for disease_id, symptom_id in disease_symptoms:
            session.run(
                """
                MATCH (d:Disease {id: $disease_id}), (s:Symptom {id: $symptom_id})
                MERGE (d)-[:HAS_SYMPTOM]->(s)
                """,
                disease_id=disease_id,
                symptom_id=symptom_id,
            )
        for symptom_id, department_id in symptom_departments:
            session.run(
                """
                MATCH (s:Symptom {id: $symptom_id}), (de:Department {id: $department_id})
                MERGE (s)-[:BELONGS_TO]->(de)
                """,
                symptom_id=symptom_id,
                department_id=department_id,
            )


def init_neo4j() -> None:
    """Connect, initialise schema, then load the local ontology."""

    from neo4j import GraphDatabase

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            session.run("RETURN 1 AS ok").single()
        create_constraints(driver)
        create_indexes(driver)
        load_medical_ontology(driver)
        logger.info("Neo4j ontology initialised")
    finally:
        driver.close()


if __name__ == "__main__":
    init_neo4j()
