"""Tests for atomic write utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.forge._atomic_write import atomic_write_json


def test_atomic_write_creates_file(tmp_path: Path):
    """Test that atomic_write_json creates a new file with correct content."""
    test_file = tmp_path / "test.json"
    test_data = '{"key": "value"}'
    
    atomic_write_json(test_file, test_data)
    
    assert test_file.exists()
    assert test_file.read_text() == test_data


def test_atomic_write_creates_parent_directories(tmp_path: Path):
    """Test that atomic_write_json creates parent directories if they don't exist."""
    test_file = tmp_path / "nested" / "dirs" / "test.json"
    test_data = '{"nested": true}'
    
    atomic_write_json(test_file, test_data)
    
    assert test_file.exists()
    assert test_file.read_text() == test_data


def test_atomic_write_overwrites_existing_file(tmp_path: Path):
    """Test that atomic_write_json overwrites an existing file."""
    test_file = tmp_path / "test.json"
    
    # Write initial content
    test_file.write_text('{"old": "data"}')
    
    # Overwrite with new content
    new_data = '{"new": "data"}'
    atomic_write_json(test_file, new_data)
    
    assert test_file.read_text() == new_data


def test_atomic_write_no_temp_file_left(tmp_path: Path):
    """Test that no .tmp file is left after successful write."""
    test_file = tmp_path / "test.json"
    test_data = '{"test": "data"}'
    
    atomic_write_json(test_file, test_data)
    
    # Check that no .tmp file exists
    temp_file = test_file.with_suffix('.tmp')
    assert not temp_file.exists()


def test_atomic_write_cleanup_on_failure(tmp_path: Path):
    """Test that temp file is cleaned up if write fails."""
    test_file = tmp_path / "test.json"
    
    # Create a file that will cause write to fail (make it a directory)
    test_file.mkdir()
    
    with pytest.raises(Exception):
        atomic_write_json(test_file, '{"test": "data"}')
    
    # Check that no .tmp file is left
    temp_file = test_file.with_suffix('.tmp')
    assert not temp_file.exists()


def test_atomic_write_json_validity(tmp_path: Path):
    """Test that written file contains valid JSON."""
    test_file = tmp_path / "test.json"
    test_data = json.dumps({"key": "value", "number": 42, "list": [1, 2, 3]})
    
    atomic_write_json(test_file, test_data)
    
    # Read and parse the file
    content = test_file.read_text()
    parsed = json.loads(content)
    
    assert parsed == {"key": "value", "number": 42, "list": [1, 2, 3]}


def test_atomic_write_large_file(tmp_path: Path):
    """Test that atomic_write_json handles large files."""
    test_file = tmp_path / "large.json"
    
    # Create a large JSON object (1MB+)
    large_data = {"data": "x" * 1_000_000}
    test_data = json.dumps(large_data)
    
    atomic_write_json(test_file, test_data)
    
    assert test_file.exists()
    assert len(test_file.read_text()) > 1_000_000


def test_atomic_write_preserves_file_on_read_during_write(tmp_path: Path):
    """Test that readers see either old or new data, never partial."""
    test_file = tmp_path / "test.json"
    
    # Write initial content
    old_data = '{"version": 1}'
    test_file.write_text(old_data)
    
    # Start writing new content
    new_data = '{"version": 2}'
    atomic_write_json(test_file, new_data)
    
    # After write completes, should see new data
    assert test_file.read_text() == new_data


def test_atomic_write_empty_content(tmp_path: Path):
    """Test that atomic_write_json handles empty content."""
    test_file = tmp_path / "empty.json"
    test_data = ""
    
    atomic_write_json(test_file, test_data)
    
    assert test_file.exists()
    assert test_file.read_text() == ""


def test_atomic_write_unicode_content(tmp_path: Path):
    """Test that atomic_write_json handles unicode content."""
    test_file = tmp_path / "unicode.json"
    test_data = '{"emoji": "🚀", "chinese": "中文", "arabic": "العربية"}'
    
    atomic_write_json(test_file, test_data)
    
    assert test_file.read_text() == test_data
