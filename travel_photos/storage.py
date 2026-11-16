"""Fotoğrafları diske, bilgilerini SQLite veritabanına kaydetme."""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    country TEXT NOT NULL,
    country_code TEXT,
    city TEXT,
    landmark TEXT,
    description TEXT,
    latitude REAL,
    longitude REAL,
    taken_at TEXT,
    camera TEXT,
    source TEXT NOT NULL,          -- 'gps', 'ai', 'manuel'
    confidence REAL,
    reasoning TEXT,
    added_at TEXT NOT NULL
);
"""


@dataclass
class Photo:
    id: int
    sha256: str
    filename: str
    stored_path: str
    country: str
    country_code: str | None
    city: str | None
    landmark: str | None
    description: str | None
    latitude: float | None
    longitude: float | None
    taken_at: str | None
    camera: str | None
    source: str
    confidence: float | None
    reasoning: str | None
    added_at: str


class PhotoStore:
    def __init__(self, base_dir: Path | str = DEFAULT_DIR):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "photos.db"
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def exists(self, sha256: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM photos WHERE sha256=?", (sha256,)).fetchone() is not None

    def add(self, *, original: bytes, jpeg: bytes, filename: str, country: str, source: str,
            country_code: str | None = None, city: str | None = None, landmark: str | None = None,
            description: str | None = None, latitude: float | None = None,
            longitude: float | None = None, taken_at: str | None = None, camera: str | None = None,
            confidence: float | None = None, reasoning: str | None = None) -> int:
        sha = self.hash_bytes(original)
        path = self.images_dir / f"{sha[:16]}.jpg"
        path.write_bytes(jpeg)
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO photos (sha256, filename, stored_path, country, country_code, city,
                   landmark, description, latitude, longitude, taken_at, camera, source,
                   confidence, reasoning, added_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sha, filename, path.name, country.strip(), country_code, (city or "").strip() or None,
                 landmark, description, latitude, longitude, taken_at, camera, source,
                 confidence, reasoning, datetime.now().isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def all(self) -> list[Photo]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM photos ORDER BY country, city, COALESCE(taken_at, added_at)"
            ).fetchall()
        return [Photo(**dict(r)) for r in rows]

    def image_path(self, photo: Photo) -> Path:
        return self.images_dir / photo.stored_path

    def update_location(self, photo_id: int, country: str, city: str | None,
                        country_code: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE photos SET country=?, city=?, country_code=COALESCE(?, country_code), "
                "source='manuel' WHERE id=?",
                (country.strip(), (city or "").strip() or None, country_code, photo_id),
            )

    def delete(self, photo_id: int) -> None:
        with self._conn() as c:
            row = c.execute("SELECT stored_path FROM photos WHERE id=?", (photo_id,)).fetchone()
            c.execute("DELETE FROM photos WHERE id=?", (photo_id,))
        if row:
            (self.images_dir / row["stored_path"]).unlink(missing_ok=True)

    def grouped(self) -> dict[str, dict[str, list[Photo]]]:
        """{ülke: {şehir: [fotoğraflar]}} — şehri olmayanlar 'Diğer' altında."""
        out: dict[str, dict[str, list[Photo]]] = {}
        for p in self.all():
            out.setdefault(p.country, {}).setdefault(p.city or "Diğer", []).append(p)
        return out
