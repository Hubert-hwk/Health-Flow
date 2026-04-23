"""
Neo4j knowledge graph initialization script.
Creates constraints, indexes, and loads initial medical ontology for HealthFlow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "healthflow123"


def create_constraints(driver: "Driver") -> None:
    """Create uniqueness constraints for primary node types."""
    constraints = [
        "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.id IS UNIQUE",
        "CREATE CONSTRAINT treatment_id IF NOT EXISTS FOR (t:Treatment) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT body_part_id IF NOT EXISTS FOR (b:BodyPart) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT department_id IF NOT EXISTS FOR (de:Department) REQUIRE de.id IS UNIQUE",
        "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
    ]

    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Constraint created: {constraint}")
            except Exception as e:
                logger.debug(f"Constraint may already exist: {e}")


def create_indexes(driver: "Driver") -> None:
    """Create indexes for efficient querying."""
    indexes = [
        "CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)",
        "CREATE INDEX symptom_name IF NOT EXISTS FOR (s:Symptom) ON (s.name)",
        "CREATE INDEX drug_name IF NOT EXISTS FOR (dr:Drug) ON (dr.name)",
        "CREATE INDEX body_part_name IF NOT EXISTS FOR (b:BodyPart) ON (b.name)",
        "CREATE INDEX department_name IF NOT EXISTS FOR (de:Department) ON (de.name)",
    ]

    with driver.session() as session:
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"Index created: {index}")
            except Exception as e:
                logger.debug(f"Index may already exist: {e}")


def load_medical_ontology(driver: "Driver") -> None:
    """Load initial medical ontology: diseases, symptoms, drugs, body parts, departments."""
    cypher_scripts = [
        # Departments
        "MERGE (de:Department {id: 'dept_internal'}) SET de.name = '内科', de.description = 'Internal medicine department'",
        "MERGE (de:Department {id: 'dept_surgery'}) SET de.name = '外科', de.description = 'Surgery department'",
        "MERGE (de:Department {id: 'dept_cardio'}) SET de.name = '心血管内科', de.description = 'Cardiovascular department'",
        "MERGE (de:Department {id: 'dept_neuro'}) SET de.name = '神经内科', de.description = 'Neurology department'",
        "MERGE (de:Department {id: 'dept_ortho'}) SET de.name = '骨科', de.description = 'Orthopedics department'",
        "MERGE (de:Department {id: 'dept_derma'}) SET de.name = '皮肤科', de.description = 'Dermatology department'",
        "MERGE (de:Department {id: 'dept_eye'}) SET de.name = '眼科', de.description = 'Ophthalmology department'",
        "MERGE (de:Department {id: 'dept_ent'}) SET de.name = '耳鼻喉科', de.description = 'ENT department'",
        "MERGE (de:Department {id: 'dept_respiratory'}) SET de.name = '呼吸内科', de.description = 'Respiratory medicine department'",
        "MERGE (de:Department {id: 'dept_gi'}) SET de.name = '消化内科', de.description = 'Gastroenterology department'",
        "MERGE (de:Department {id: 'dept_endo'}) SET de.name = '内分泌科', de.description = 'Endocrinology department'",
        "MERGE (de:Department {id: 'dept_urology'}) SET de.name = '泌尿外科', de.description = 'Urology department'",
        # Common symptoms
        "MERGE (s:Symptom {id: 'sym_fever'}) SET s.name = '发热', s.severity = 'medium'",
        "MERGE (s:Symptom {id: 'sym_cough'}) SET s.name = '咳嗽', s.severity = 'low'",
        "MERGE (s:Symptom {id: 'sym_headache'}) SET s.name = '头痛', s.severity = 'medium'",
        "MERGE (s:Symptom {id: 'sym_chest_pain'}) SET s.name = '胸痛', s.severity = 'high'",
        "MERGE (s:Symptom {id: 'sym_fatigue'}) SET s.name = '乏力', s.severity = 'low'",
        "MERGE (s:Symptom {id: 'sym_nausea'}) SET s.name = '恶心', s.severity = 'low'",
        "MERGE (s:Symptom {id: 'sym_dizziness'}) SET s.name = '头晕', s.severity = 'medium'",
        "MERGE (s:Symptom {id: 'sym_abdominal_pain'}) SET s.name = '腹痛', s.severity = 'medium'",
        "MERGE (s:Symptom {id: 'sym_joint_pain'}) SET s.name = '关节痛', s.severity = 'medium'",
        "MERGE (s:Symptom {id: 'sym_rash'}) SET s.name = '皮疹', s.severity = 'medium'",
        # Common diseases
        "MERGE (d:Disease {id: 'disease_hypertension'}) SET d.name = '高血压', d.category = 'cardiovascular'",
        "MERGE (d:Disease {id: 'disease_diabetes_type2'}) SET d.name = '2型糖尿病', d.category = 'endocrine'",
        "MERGE (d:Disease {id: 'disease_covid19'}) SET d.name = '新冠病毒感染', d.category = 'infectious'",
        "MERGE (d:Disease {id: 'disease_asthma'}) SET d.name = '哮喘', d.category = 'respiratory'",
        "MERGE (d:Disease {id: 'disease_gerd'}) SET d.name = '胃食管反流病', d.category = 'gi'",
        # Common drugs
        "MERGE (dr:Drug {id: 'drug_amlodipine'}) SET dr.name = '氨氯地平', dr.category = 'antihypertensive'",
        "MERGE (dr:Drug {id: 'drug_metformin'}) SET dr.name = '二甲双胍', dr.category = 'antidiabetic'",
        "MERGE (dr:Drug {id: 'drug_ibuprofen'}) SET dr.name = '布洛芬', dr.category = 'nsaid'",
        "MERGE (dr:Drug {id: 'drug_omeprazole'}) SET dr.name = '奥美拉唑', dr.category = 'ppi'",
        # Body parts
        "MERGE (b:BodyPart {id: 'bp_head'}) SET b.name = '头部'",
        "MERGE (b:BodyPart {id: 'bp_chest'}) SET b.name = '胸部'",
        "MERGE (b:BodyPart {id: 'bp_abdomen'}) SET b.name = '腹部'",
        "MERGE (b:BodyPart {id: 'bp_limb'}) SET b.name = '四肢'",
        # Relationships
        "MATCH (d:Disease {id: 'disease_hypertension'}), (de:Department {id: 'dept_cardio'}) MERGE (d)-[:BELONGS_TO]->(de)",
        "MATCH (d:Disease {id: 'disease_diabetes_type2'}), (de:Department {id: 'dept_endo'}) MERGE (d)-[:BELONGS_TO]->(de)",
        "MATCH (d:Disease {id: 'disease_covid19'}), (de:Department {id: 'dept_respiratory'}) MERGE (d)-[:BELONGS_TO]->(de)",
        "MATCH (s:Symptom {id: 'sym_chest_pain'}), (d:Disease {id: 'disease_hypertension'}) MERGE (s)-[:ASSOCIATED_WITH]->(d)",
        "MATCH (s:Symptom {id: 'sym_cough'}), (d:Disease {id: 'disease_covid19'}) MERGE (s)-[:ASSOCIATED_WITH]->(d)",
        "MATCH (s:Symptom {id: 'sym_fever'}), (d:Disease {id: 'disease_covid19'}) MERGE (s)-[:ASSOCIATED_WITH]->(d)",
        "MATCH (s:Symptom {id: 'sym_fatigue'}), (d:Disease {id: 'disease_diabetes_type2'}) MERGE (s)-[:ASSOCIATED_WITH]->(d)",
        "MATCH (dr:Drug {id: 'drug_amlodipine'}), (d:Disease {id: 'disease_hypertension'}) MERGE (dr)-[:TREATS]->(d)",
        "MATCH (dr:Drug {id: 'drug_metformin'}), (d:Disease {id: 'disease_diabetes_type2'}) MERGE (dr)-[:TREATS]->(d)",
    ]

    with driver.session() as session:
        for cypher in cypher_scripts:
            try:
                session.run(cypher)
            except Exception as e:
                logger.warning(f"Script failed: {cypher[:60]}... -> {e}")

    logger.info("Medical ontology loaded")


def init_neo4j(
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
) -> "Driver":
    """Connect to Neo4j, create constraints, indexes, and load ontology."""
    from neo4j import GraphDatabase

    logger.info(f"Connecting to Neo4j at {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Verify connectivity
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Neo4j connection verified")

    create_constraints(driver)
    create_indexes(driver)
    load_medical_ontology(driver)

    logger.info("Neo4j initialisation complete")
    return driver


if __name__ == "__main__":
    init_neo4j()
