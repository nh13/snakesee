"""Tests for utility functions."""

import json
from pathlib import Path

from snakesee.utils import safe_file_size
from snakesee.utils import safe_mtime
from snakesee.utils import safe_read_json
from snakesee.utils import safe_read_text


class TestSafeMtime:
    """Tests for safe_mtime function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Test mtime for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        mtime = safe_mtime(test_file)
        assert mtime > 0

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test mtime for nonexistent file returns 0.0."""
        nonexistent = tmp_path / "nonexistent.txt"
        assert safe_mtime(nonexistent) == 0.0

    def test_directory(self, tmp_path: Path) -> None:
        """Test mtime for directory works."""
        mtime = safe_mtime(tmp_path)
        assert mtime > 0


class TestSafeReadText:
    """Tests for safe_read_text function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Test reading existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        content = safe_read_text(test_file)
        assert content == "hello world"

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading nonexistent file returns default."""
        nonexistent = tmp_path / "nonexistent.txt"
        assert safe_read_text(nonexistent) == ""

    def test_nonexistent_file_custom_default(self, tmp_path: Path) -> None:
        """Test reading nonexistent file with custom default."""
        nonexistent = tmp_path / "nonexistent.txt"
        assert safe_read_text(nonexistent, default="N/A") == "N/A"

    def test_file_with_encoding_errors(self, tmp_path: Path) -> None:
        """Test reading file with encoding errors is handled."""
        test_file = tmp_path / "binary.txt"
        test_file.write_bytes(b"hello \xff\xfe world")
        # Should not raise, should handle encoding errors
        content = safe_read_text(test_file)
        assert "hello" in content
        assert "world" in content


class TestSafeReadJson:
    """Tests for safe_read_json function."""

    def test_valid_json_file(self, tmp_path: Path) -> None:
        """Test reading valid JSON file."""
        test_file = tmp_path / "test.json"
        data = {"key": "value", "count": 42}
        test_file.write_text(json.dumps(data))
        result = safe_read_json(test_file)
        assert result == data

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading nonexistent file returns None."""
        nonexistent = tmp_path / "nonexistent.json"
        assert safe_read_json(nonexistent) is None

    def test_nonexistent_file_custom_default(self, tmp_path: Path) -> None:
        """Test reading nonexistent file with custom default."""
        nonexistent = tmp_path / "nonexistent.json"
        default = {"default": True}
        assert safe_read_json(nonexistent, default=default) == default

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Test reading invalid JSON returns None."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json {{{")
        assert safe_read_json(test_file) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test reading empty file returns None (invalid JSON)."""
        test_file = tmp_path / "empty.json"
        test_file.write_text("")
        assert safe_read_json(test_file) is None


class TestSafeFileSize:
    """Tests for safe_file_size function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Test size for existing file."""
        test_file = tmp_path / "test.txt"
        content = "hello world"
        test_file.write_text(content)
        size = safe_file_size(test_file)
        assert size == len(content)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test size for nonexistent file returns 0."""
        nonexistent = tmp_path / "nonexistent.txt"
        assert safe_file_size(nonexistent) == 0

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test size for empty file returns 0."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        assert safe_file_size(test_file) == 0

    def test_binary_file(self, tmp_path: Path) -> None:
        """Test size for binary file."""
        test_file = tmp_path / "binary.bin"
        data = b"\x00\x01\x02\x03" * 100
        test_file.write_bytes(data)
        assert safe_file_size(test_file) == len(data)
