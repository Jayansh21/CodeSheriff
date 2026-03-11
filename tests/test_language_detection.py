"""
Tests for utils/language_detection.py
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.language_detection import detect_language, detect_languages_from_chunks, OPTIMIZED_LANGUAGE


# ---------------------------------------------------------------------------
# detect_language — extension mapping
# ---------------------------------------------------------------------------
class TestDetectLanguage:

    @pytest.mark.parametrize("filename,expected", [
        ("user_service.py", "Python"),
        ("server.js", "JavaScript"),
        ("App.tsx", "TypeScript"),
        ("Main.java", "Java"),
        ("service.go", "Go"),
        ("script.rb", "Ruby"),
        ("index.php", "PHP"),
        ("Program.cs", "C#"),
        ("main.cpp", "C++"),
        ("lib.c", "C"),
        ("util.rs", "Rust"),
        ("app.kt", "Kotlin"),
        ("view.swift", "Swift"),
        ("build.scala", "Scala"),
        ("deploy.sh", "Shell"),
        ("query.sql", "SQL"),
        ("page.html", "HTML"),
        ("style.css", "CSS"),
    ], ids=lambda v: v if isinstance(v, str) and "." in v else "")
    def test_known_extensions(self, filename, expected):
        assert detect_language(filename) == expected

    def test_unknown_extension(self):
        assert detect_language("unknown.xyz") == "Unknown"

    def test_no_extension(self):
        assert detect_language("Makefile") == "Unknown"

    def test_empty_string(self):
        assert detect_language("") == "Unknown"

    def test_case_insensitive_extension(self):
        assert detect_language("Module.PY") == "Python"
        assert detect_language("app.JS") == "JavaScript"

    def test_nested_path(self):
        assert detect_language("src/utils/helper.py") == "Python"

    def test_multiple_dots(self):
        assert detect_language("my.config.js") == "JavaScript"


# ---------------------------------------------------------------------------
# detect_languages_from_chunks — aggregation
# ---------------------------------------------------------------------------
class TestDetectLanguagesFromChunks:

    def test_single_language(self):
        chunks = [
            {"file": "a.py", "code": "x"},
            {"file": "b.py", "code": "y"},
        ]
        result = detect_languages_from_chunks(chunks)
        assert result["primary"] == "Python"
        assert result["languages"] == {"Python": 2}

    def test_mixed_languages(self):
        chunks = [
            {"file": "a.py", "code": "x"},
            {"file": "b.js", "code": "y"},
            {"file": "c.py", "code": "z"},
        ]
        result = detect_languages_from_chunks(chunks)
        assert result["primary"] == "Python"
        assert result["languages"]["Python"] == 2
        assert result["languages"]["JavaScript"] == 1

    def test_empty_chunks(self):
        result = detect_languages_from_chunks([])
        assert result["primary"] == "Unknown"
        assert result["languages"] == {}


# ---------------------------------------------------------------------------
# OPTIMIZED_LANGUAGE constant
# ---------------------------------------------------------------------------
class TestOptimizedLanguage:

    def test_optimized_is_python(self):
        assert OPTIMIZED_LANGUAGE == "Python"
