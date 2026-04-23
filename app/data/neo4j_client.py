"""Neo4j knowledge graph client."""

from typing import List, Optional, Dict, Any

from app.config import get_settings


class Neo4jClient:
    """Neo4j knowledge graph client."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j URI
            user: Neo4j user
            password: Neo4j password
            database: Database name
        """
        settings = get_settings()
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database
        self._driver = None

    @property
    def driver(self):
        """Get Neo4j driver (lazy initialization)."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                )
            except ImportError:
                # For testing without actual Neo4j
                self._driver = None
        return self._driver

    def connect(self) -> bool:
        """Test connection."""
        if not self.driver:
            return False
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1")
                return result.single() is not None
        except Exception:
            return False

    def get_related_symptoms(self, disease: str) -> List[Dict[str, Any]]:
        """
        Get symptoms related to a disease.

        Args:
            disease: Disease name

        Returns:
            List of symptom nodes
        """
        if not self.driver:
            return []

        query = """
        MATCH (d:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.name AS name, s.description AS description
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, disease=disease)
                return [{"name": r["name"], "description": r["description"]} for r in result]
        except Exception:
            return []

    def get_related_drugs(self, disease: str) -> List[Dict[str, Any]]:
        """
        Get drugs for treating a disease.

        Args:
            disease: Disease name

        Returns:
            List of drug nodes
        """
        if not self.driver:
            return []

        query = """
        MATCH (d:Disease {name: $disease})-[:TREATED_BY]->(dr:Drug)
        RETURN dr.name AS name, dr.description AS description
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, disease=disease)
                return [{"name": r["name"], "description": r["description"]} for r in result]
        except Exception:
            return []

    def get_related_examinations(self, disease: str) -> List[Dict[str, Any]]:
        """
        Get examinations for diagnosing a disease.

        Args:
            disease: Disease name

        Returns:
            List of examination nodes
        """
        if not self.driver:
            return []

        query = """
        MATCH (d:Disease {name: $disease})-[:DIAGNOSED_BY]->(e:Examination)
        RETURN e.name AS name, e.description AS description
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, disease=disease)
                return [{"name": r["name"], "description": r["description"]} for r in result]
        except Exception:
            return []

    def get_department(self, symptom: str) -> Optional[str]:
        """
        Get department for a symptom.

        Args:
            symptom: Symptom name

        Returns:
            Department name
        """
        if not self.driver:
            return None

        query = """
        MATCH (s:Symptom {name: $symptom})-[:BELONGS_TO]->(d:Department)
        RETURN d.name AS name
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, symptom=symptom)
                record = result.single()
                return record["name"] if record else None
        except Exception:
            return None

    def query_by_entity(self, entity: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query knowledge graph by entity name.

        Args:
            entity: Entity name
            limit: Result limit

        Returns:
            List of related knowledge
        """
        if not self.driver:
            return []

        query = """
        MATCH (n {name: $entity})
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN n.name AS entity,
               labels(n) AS entity_type,
               type(r) AS relation,
               related.name AS related_entity
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity=entity, limit=limit)
                return [dict(r) for r in result]
        except Exception:
            return []

    def find_diagnosis_path(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Find potential diagnosis paths from symptoms.

        Args:
            symptoms: List of symptoms

        Returns:
            List of possible diseases with paths
        """
        if not self.driver:
            return []

        query = """
        MATCH path = (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH d, collect(DISTINCT s.name) AS matched_symptoms
        WHERE size(matched_symptoms) >= 2
        RETURN d.name AS disease,
               d.description AS description,
               size(matched_symptoms) AS symptom_count,
               matched_symptoms
        ORDER BY symptom_count DESC
        LIMIT 5
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, symptoms=symptoms)
                return [
                    {
                        "disease": r["disease"],
                        "description": r["description"],
                        "symptom_count": r["symptom_count"],
                        "matched_symptoms": r["matched_symptoms"],
                    }
                    for r in result
                ]
        except Exception:
            return []

    def close(self):
        """Close driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


# Global instance
_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client singleton."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
