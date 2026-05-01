"""
Database Schema for CrimeSketch AI

Defines SQLite tables for storing suspect information, embeddings, and search history.
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import numpy as np


class DatabaseManager:
    """
    Manages SQLite database for CrimeSketch AI system.
    
    Tables:
    - suspects: Core suspect information with metadata
    - embeddings: High-dimensional face embeddings for similarity search
    - search_logs: Query history and results for analytics
    """
    
    def __init__(self, db_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crimesketch.db")):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self.init_database()
    
    def init_database(self):
        """Initialize database connection and create tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables."""
        # Suspects table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS suspects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                city TEXT,
                crime_type TEXT,
                risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high')),
                image_path TEXT NOT NULL UNIQUE,
                image_url TEXT,
                embedding_id INTEGER,
                dataset_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Embeddings table (stores normalized 512D or 128D vectors)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suspect_id INTEGER UNIQUE NOT NULL,
                embedding_vector BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (suspect_id) REFERENCES suspects(id)
            )
        ''')
        
        # Search logs table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_sketch_path TEXT,
                query_sketch_url TEXT,
                top_k_results TEXT,
                best_match_id INTEGER,
                best_match_score REAL,
                search_time_ms REAL,
                user_id TEXT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (best_match_id) REFERENCES suspects(id)
            )
        ''')
        
        # Dataset statistics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_suspects INTEGER,
                total_sketches INTEGER,
                total_embeddings INTEGER,
                avg_match_confidence REAL,
                last_indexed_at TIMESTAMP,
                last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def add_suspect(self, name: str, image_path: str, age: Optional[int] = None,
                   gender: Optional[str] = None, city: Optional[str] = None,
                   crime_type: Optional[str] = None, risk_level: str = 'medium',
                   dataset_source: str = 'unknown', image_url: Optional[str] = None) -> int:
        """
        Add a suspect to the database.
        
        Args:
            name: Suspect name
            image_path: Path to suspect image
            age: Age of suspect
            gender: Gender
            city: City
            crime_type: Type of crime
            risk_level: Risk level (low, medium, high)
            dataset_source: Source dataset (dataset1, dataset2, dataset3, etc.)
            image_url: URL to suspect image (for frontend display)
            
        Returns:
            Suspect ID
        """
        try:
            self.cursor.execute('''
                INSERT INTO suspects (name, image_path, age, gender, city, crime_type, 
                                     risk_level, dataset_source, image_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, image_path, age, gender, city, crime_type, risk_level, dataset_source, image_url))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError as e:
            print(f"Integrity error adding suspect: {e}")
            return -1
    
    def add_embedding(self, suspect_id: int, embedding: np.ndarray, model_version: str = "resnet50-v1"):
        """
        Store embedding vector for a suspect.
        
        Args:
            suspect_id: ID of suspect
            embedding: Embedding vector (numpy array)
            model_version: Version of model used to generate embedding
        """
        embedding_blob = embedding.astype(np.float32).tobytes()
        embedding_dim = embedding.shape[0]
        
        self.cursor.execute('''
            INSERT OR REPLACE INTO embeddings (suspect_id, embedding_vector, embedding_dim, model_version)
            VALUES (?, ?, ?, ?)
        ''', (suspect_id, embedding_blob, embedding_dim, model_version))
        
        # Update suspect's embedding_id
        self.cursor.execute('UPDATE suspects SET embedding_id = ? WHERE id = ?',
                          (self.cursor.lastrowid, suspect_id))
        self.conn.commit()
    
    def get_embedding(self, suspect_id: int) -> Optional[np.ndarray]:
        """
        Retrieve embedding vector for a suspect.
        
        Args:
            suspect_id: ID of suspect
            
        Returns:
            Embedding vector as numpy array or None
        """
        self.cursor.execute('SELECT embedding_vector, embedding_dim FROM embeddings WHERE suspect_id = ?',
                          (suspect_id,))
        result = self.cursor.fetchone()
        
        if result:
            embedding_blob, embedding_dim = result
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(embedding_dim)
            return embedding
        return None
    
    def get_all_embeddings(self) -> tuple:
        """
        Retrieve all embeddings for FAISS indexing.
        
        Returns:
            Tuple of (embeddings_array, suspect_ids)
        """
        self.cursor.execute('''
            SELECT e.embedding_vector, e.embedding_dim, s.id
            FROM embeddings e
            JOIN suspects s ON e.suspect_id = s.id
            ORDER BY s.id
        ''')
        
        results = self.cursor.fetchall()
        if not results:
            return np.array([]), []
        
        embeddings = []
        suspect_ids = []
        
        for embedding_blob, embedding_dim, suspect_id in results:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(embedding_dim)
            embeddings.append(embedding)
            suspect_ids.append(suspect_id)
        
        embeddings_array = np.stack(embeddings, axis=0)
        return embeddings_array, suspect_ids
    
    def get_suspect(self, suspect_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve suspect information.
        
        Args:
            suspect_id: ID of suspect
            
        Returns:
            Dictionary with suspect information or None
        """
        self.cursor.execute('''
            SELECT id, name, age, gender, city, crime_type, risk_level, image_path, image_url, dataset_source
            FROM suspects WHERE id = ?
        ''', (suspect_id,))
        
        result = self.cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'age': result[2],
                'gender': result[3],
                'city': result[4],
                'crime_type': result[5],
                'risk_level': result[6],
                'image_path': result[7],
                'image_url': result[8],
                'dataset_source': result[9]
            }
        return None
    
    def log_search(self, query_sketch_path: str, top_k_results: List[Dict], 
                  best_match_id: int, best_match_score: float, search_time_ms: float,
                  user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Log a search query and results.
        
        Args:
            query_sketch_path: Path to query sketch
            top_k_results: List of top-K match dictionaries
            best_match_id: ID of best match
            best_match_score: Confidence score of best match
            search_time_ms: Search time in milliseconds
            user_id: User ID (optional)
            session_id: Session ID (optional)
        """
        results_json = json.dumps(top_k_results)
        
        self.cursor.execute('''
            INSERT INTO search_logs (query_sketch_path, top_k_results, best_match_id, 
                                    best_match_score, search_time_ms, user_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (query_sketch_path, results_json, best_match_id, best_match_score, search_time_ms, user_id, session_id))
        
        self.conn.commit()
    
    def get_search_history(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve search history.
        
        Args:
            limit: Maximum number of records to retrieve
            session_id: Filter by session ID (optional)
            
        Returns:
            List of search log dictionaries
        """
        if session_id:
            self.cursor.execute('''
                SELECT id, query_sketch_path, best_match_id, best_match_score, search_time_ms, created_at
                FROM search_logs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
            ''', (session_id, limit))
        else:
            self.cursor.execute('''
                SELECT id, query_sketch_path, best_match_id, best_match_score, search_time_ms, created_at
                FROM search_logs ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
        
        results = self.cursor.fetchall()
        return [
            {
                'id': r[0],
                'query_sketch_path': r[1],
                'best_match_id': r[2],
                'best_match_score': r[3],
                'search_time_ms': r[4],
                'created_at': r[5]
            }
            for r in results
        ]
    
    def get_suspects_filtered(self, city: Optional[str] = None, crime_type: Optional[str] = None,
                             risk_level: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve suspects with optional filtering.
        
        Args:
            city: Filter by city
            crime_type: Filter by crime type
            risk_level: Filter by risk level (low, medium, high)
            limit: Maximum number of results
            
        Returns:
            List of suspect dictionaries
        """
        query = 'SELECT id, name, age, gender, city, crime_type, risk_level, image_path, image_url FROM suspects WHERE 1=1'
        params = []
        
        if city:
            query += ' AND city = ?'
            params.append(city)
        if crime_type:
            query += ' AND crime_type = ?'
            params.append(crime_type)
        if risk_level:
            # Normalize risk_level to lowercase for database query
            query += ' AND risk_level = ?'
            params.append(risk_level.lower())
        
        query += ' LIMIT ?'
        params.append(limit)
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        suspects = []
        for r in results:
            suspects.append({
                'id': r[0],
                'name': r[1],
                'age': r[2],
                'gender': r[3],
                'city': r[4],
                'crime_type': r[5],
                'risk_level': r[6],
                'image_path': r[7],
                'image_url': r[8]
            })
        
        return suspects
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        self.cursor.execute('SELECT COUNT(*) FROM suspects')
        total_suspects = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM embeddings')
        total_embeddings = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT AVG(best_match_score) FROM search_logs')
        avg_confidence = self.cursor.fetchone()[0] or 0.0
        
        return {
            'total_suspects': total_suspects,
            'total_embeddings': total_embeddings,
            'avg_match_confidence': float(avg_confidence),
            'last_updated': 'now'
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Test database initialization
    db = DatabaseManager()
    print("Database initialized successfully!")
    print(f"Stats: {db.get_stats()}")
    db.close()
