"""Persistence module for storing validation results."""
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import json
import os
import sqlite3
from datetime import datetime
import logging
from pathlib import Path
import pymongo
from dataclasses import asdict
import yaml

from .validators import ValidationResult

logger = logging.getLogger(__name__)

class BasePersistence(ABC):
    """Base class for persistence implementations."""
    
    @abstractmethod
    def save_result(
        self,
        objective_id: str,
        validation_type: str,
        result: ValidationResult
    ) -> bool:
        """Save a validation result.
        
        Args:
            objective_id: ID of the objective
            validation_type: Type of validation
            result: Validation result to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def get_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a validation result.
        
        Args:
            objective_id: ID of the objective
            validation_type: Type of validation
            timestamp: Optional specific timestamp
            
        Returns:
            Validation result if found, None otherwise
        """
        pass
        
    @abstractmethod
    def get_results(
        self,
        objective_id: Optional[str] = None,
        validation_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get multiple validation results.
        
        Args:
            objective_id: Optional objective ID filter
            validation_type: Optional validation type filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of validation results
        """
        pass
        
    @abstractmethod
    def delete_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> bool:
        """Delete a validation result.
        
        Args:
            objective_id: ID of the objective
            validation_type: Type of validation
            timestamp: Optional specific timestamp
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass

class FilePersistence(BasePersistence):
    """File-based persistence implementation."""
    
    def __init__(self, base_dir: str, format: str = "json"):
        """Initialize file persistence.
        
        Args:
            base_dir: Base directory for storing files
            format: File format (json or yaml)
        """
        self.base_dir = Path(base_dir)
        self.format = format.lower()
        if self.format not in ["json", "yaml"]:
            raise ValueError("Format must be either 'json' or 'yaml'")
            
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_file_path(self, objective_id: str, validation_type: str) -> Path:
        """Get the file path for a validation result."""
        return self.base_dir / f"{objective_id}_{validation_type}.{self.format}"
        
    def save_result(
        self,
        objective_id: str,
        validation_type: str,
        result: ValidationResult
    ) -> bool:
        try:
            file_path = self._get_file_path(objective_id, validation_type)
            
            # Convert result to dict
            result_dict = {
                "objective_id": objective_id,
                "validation_type": validation_type,
                "timestamp": result.timestamp,
                "result": asdict(result)
            }
            
            # Load existing results
            existing_results = []
            if file_path.exists():
                with open(file_path, "r") as f:
                    if self.format == "json":
                        existing_results = json.load(f)
                    else:
                        existing_results = yaml.safe_load(f) or []
                        
            # Add new result
            existing_results.append(result_dict)
            
            # Save updated results
            with open(file_path, "w") as f:
                if self.format == "json":
                    json.dump(existing_results, f, indent=2)
                else:
                    yaml.safe_dump(existing_results, f)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False
            
    def get_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            file_path = self._get_file_path(objective_id, validation_type)
            if not file_path.exists():
                return None
                
            with open(file_path, "r") as f:
                if self.format == "json":
                    results = json.load(f)
                else:
                    results = yaml.safe_load(f) or []
                    
            if timestamp:
                # Find specific result
                for result in results:
                    if result["timestamp"] == timestamp:
                        return result
                return None
            else:
                # Return most recent result
                return results[-1] if results else None
                
        except Exception as e:
            logger.error(f"Error getting result: {e}")
            return None
            
    def get_results(
        self,
        objective_id: Optional[str] = None,
        validation_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            all_results = []
            
            # Get all relevant files
            for file_path in self.base_dir.glob(f"*.{self.format}"):
                try:
                    with open(file_path, "r") as f:
                        if self.format == "json":
                            results = json.load(f)
                        else:
                            results = yaml.safe_load(f) or []
                            
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    continue
                    
            # Apply filters
            filtered_results = all_results
            
            if objective_id:
                filtered_results = [
                    r for r in filtered_results
                    if r["objective_id"] == objective_id
                ]
                
            if validation_type:
                filtered_results = [
                    r for r in filtered_results
                    if r["validation_type"] == validation_type
                ]
                
            if start_time:
                filtered_results = [
                    r for r in filtered_results
                    if r["timestamp"] >= start_time
                ]
                
            if end_time:
                filtered_results = [
                    r for r in filtered_results
                    if r["timestamp"] <= end_time
                ]
                
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            return []
            
    def delete_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> bool:
        try:
            file_path = self._get_file_path(objective_id, validation_type)
            if not file_path.exists():
                return False
                
            with open(file_path, "r") as f:
                if self.format == "json":
                    results = json.load(f)
                else:
                    results = yaml.safe_load(f) or []
                    
            if timestamp:
                # Remove specific result
                results = [r for r in results if r["timestamp"] != timestamp]
            else:
                # Remove all results
                results = []
                
            if results:
                with open(file_path, "w") as f:
                    if self.format == "json":
                        json.dump(results, f, indent=2)
                    else:
                        yaml.safe_dump(results, f)
            else:
                # Delete file if no results left
                file_path.unlink()
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting result: {e}")
            return False

class SQLitePersistence(BasePersistence):
    """SQLite-based persistence implementation."""
    
    def __init__(self, db_path: str):
        """Initialize SQLite persistence.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    objective_id TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_valid BOOLEAN NOT NULL,
                    score REAL,
                    details TEXT,
                    message TEXT
                )
            """)
            conn.commit()
            
    def save_result(
        self,
        objective_id: str,
        validation_type: str,
        result: ValidationResult
    ) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO validation_results
                    (objective_id, validation_type, timestamp, is_valid, score,
                     details, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        objective_id,
                        validation_type,
                        result.timestamp,
                        result.is_valid,
                        result.score,
                        json.dumps(result.details) if result.details else None,
                        result.message
                    )
                )
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving result to SQLite: {e}")
            return False
            
    def get_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if timestamp:
                    cursor.execute(
                        """
                        SELECT * FROM validation_results
                        WHERE objective_id = ? AND validation_type = ?
                        AND timestamp = ?
                        """,
                        (objective_id, validation_type, timestamp)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM validation_results
                        WHERE objective_id = ? AND validation_type = ?
                        ORDER BY timestamp DESC LIMIT 1
                        """,
                        (objective_id, validation_type)
                    )
                    
                row = cursor.fetchone()
                if row:
                    return {
                        "objective_id": row[1],
                        "validation_type": row[2],
                        "timestamp": row[3],
                        "result": {
                            "is_valid": bool(row[4]),
                            "score": row[5],
                            "details": json.loads(row[6]) if row[6] else None,
                            "message": row[7]
                        }
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting result from SQLite: {e}")
            return None
            
    def get_results(
        self,
        objective_id: Optional[str] = None,
        validation_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM validation_results WHERE 1=1"
                params = []
                
                if objective_id:
                    query += " AND objective_id = ?"
                    params.append(objective_id)
                    
                if validation_type:
                    query += " AND validation_type = ?"
                    params.append(validation_type)
                    
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        "objective_id": row[1],
                        "validation_type": row[2],
                        "timestamp": row[3],
                        "result": {
                            "is_valid": bool(row[4]),
                            "score": row[5],
                            "details": json.loads(row[6]) if row[6] else None,
                            "message": row[7]
                        }
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting results from SQLite: {e}")
            return []
            
    def delete_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if timestamp:
                    cursor.execute(
                        """
                        DELETE FROM validation_results
                        WHERE objective_id = ? AND validation_type = ?
                        AND timestamp = ?
                        """,
                        (objective_id, validation_type, timestamp)
                    )
                else:
                    cursor.execute(
                        """
                        DELETE FROM validation_results
                        WHERE objective_id = ? AND validation_type = ?
                        """,
                        (objective_id, validation_type)
                    )
                    
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error deleting result from SQLite: {e}")
            return False

class MongoPersistence(BasePersistence):
    """MongoDB-based persistence implementation."""
    
    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str = "validation_results"
    ):
        """Initialize MongoDB persistence.
        
        Args:
            connection_string: MongoDB connection string
            database: Database name
            collection: Collection name
        """
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]
        
        # Create indexes
        self.collection.create_index([
            ("objective_id", pymongo.ASCENDING),
            ("validation_type", pymongo.ASCENDING),
            ("timestamp", pymongo.DESCENDING)
        ])
        
    def save_result(
        self,
        objective_id: str,
        validation_type: str,
        result: ValidationResult
    ) -> bool:
        try:
            document = {
                "objective_id": objective_id,
                "validation_type": validation_type,
                "timestamp": result.timestamp,
                "result": asdict(result)
            }
            
            self.collection.insert_one(document)
            return True
            
        except Exception as e:
            logger.error(f"Error saving result to MongoDB: {e}")
            return False
            
    def get_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            query = {
                "objective_id": objective_id,
                "validation_type": validation_type
            }
            
            if timestamp:
                query["timestamp"] = timestamp
                document = self.collection.find_one(query)
            else:
                document = self.collection.find_one(
                    query,
                    sort=[("timestamp", pymongo.DESCENDING)]
                )
                
            return document if document else None
            
        except Exception as e:
            logger.error(f"Error getting result from MongoDB: {e}")
            return None
            
    def get_results(
        self,
        objective_id: Optional[str] = None,
        validation_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            query = {}
            
            if objective_id:
                query["objective_id"] = objective_id
                
            if validation_type:
                query["validation_type"] = validation_type
                
            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time
                if end_time:
                    query["timestamp"]["$lte"] = end_time
                    
            return list(
                self.collection.find(
                    query,
                    sort=[("timestamp", pymongo.DESCENDING)]
                )
            )
            
        except Exception as e:
            logger.error(f"Error getting results from MongoDB: {e}")
            return []
            
    def delete_result(
        self,
        objective_id: str,
        validation_type: str,
        timestamp: Optional[str] = None
    ) -> bool:
        try:
            query = {
                "objective_id": objective_id,
                "validation_type": validation_type
            }
            
            if timestamp:
                query["timestamp"] = timestamp
                
            result = self.collection.delete_many(query)
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting result from MongoDB: {e}")
            return False

class PersistenceFactory:
    """Factory for creating persistence instances."""
    
    @staticmethod
    def create_persistence(
        persistence_type: str,
        **kwargs
    ) -> BasePersistence:
        """Create a persistence instance.
        
        Args:
            persistence_type: Type of persistence (file, sqlite, mongo)
            **kwargs: Additional arguments for specific persistence type
            
        Returns:
            Persistence instance
            
        Raises:
            ValueError: If persistence type is not supported
        """
        if persistence_type == "file":
            return FilePersistence(
                base_dir=kwargs.get("base_dir", "validation_results"),
                format=kwargs.get("format", "json")
            )
        elif persistence_type == "sqlite":
            return SQLitePersistence(
                db_path=kwargs.get("db_path", "validation_results.db")
            )
        elif persistence_type == "mongo":
            return MongoPersistence(
                connection_string=kwargs["connection_string"],
                database=kwargs["database"],
                collection=kwargs.get("collection", "validation_results")
            )
        else:
            raise ValueError(f"Unsupported persistence type: {persistence_type}")
