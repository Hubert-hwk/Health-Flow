"""Neo4j client for source-aware medical graph retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import get_settings


class Neo4jClient:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ) -> None:
        settings = get_settings()
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database
        self._driver = None
        self.last_error: Optional[str] = None

    @property
    def driver(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except Exception as exc:
                self.last_error = str(exc)
                self._driver = None
        return self._driver

    @property
    def available(self) -> bool:
        return self.driver is not None

    def connect(self) -> bool:
        if not self.driver:
            return False
        try:
            with self.driver.session(database=self.database) as session:
                return session.run("RETURN 1 AS ok").single()["ok"] == 1
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def get_related_symptoms(self, disease: str) -> List[Dict[str, Any]]:
        return self._query_named(
            """
            MATCH (d:Disease {name: $name})-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN s.name AS name, coalesce(s.description, '') AS description
            LIMIT $limit
            """,
            {"name": disease, "limit": 20},
        )

    def get_related_drugs(self, disease: str) -> List[Dict[str, Any]]:
        return self._query_named(
            """
            MATCH (d:Disease {name: $name})-[:TREATED_BY]->(dr:Drug)
            RETURN dr.name AS name, coalesce(dr.description, '') AS description
            LIMIT $limit
            """,
            {"name": disease, "limit": 20},
        )

    def get_related_examinations(self, disease: str) -> List[Dict[str, Any]]:
        return self._query_named(
            """
            MATCH (d:Disease {name: $name})-[:DIAGNOSED_BY]->(e:Examination)
            RETURN e.name AS name, coalesce(e.description, '') AS description
            LIMIT $limit
            """,
            {"name": disease, "limit": 20},
        )

    def get_department(self, symptom: str) -> Optional[str]:
        rows = self._query_named(
            """
            MATCH (s:Symptom {name: $name})-[:BELONGS_TO]->(d:Department)
            RETURN d.name AS name LIMIT 1
            """,
            {"name": symptom},
        )
        return rows[0].get("name") if rows else None

    def query_by_entity(self, entity: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        query = """
        MATCH (n {name: $entity})
        OPTIONAL MATCH path = (n)-[r]-(related)
        RETURN n.name AS entity, labels(n) AS entity_type,
               type(r) AS relation, related.name AS related_entity,
               coalesce(related.description, '') AS description,
               [node IN nodes(path) | coalesce(node.name, '')] AS path
        LIMIT $limit
        """
        return self._query_named(query, {"entity": entity, "limit": max(1, min(limit, 100))})

    def find_diagnosis_path(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        if not self.driver or not symptoms:
            return []
        query = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH d, collect(DISTINCT s.name) AS matched_symptoms
        RETURN d.name AS disease, coalesce(d.description, '') AS description,
               size(matched_symptoms) AS symptom_count, matched_symptoms
        ORDER BY symptom_count DESC LIMIT $limit
        """
        return self._query_named(query, {"symptoms": symptoms[:20], "limit": 5})

    def _query_named(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.driver:
            return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **parameters)
                return [dict(record) for record in result]
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None


_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
