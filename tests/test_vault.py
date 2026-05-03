"""
test_vault.py
~~~~~~~~~~~~~
Unit tests for the local vault file handler.
"""

import pytest

from utils.vault import VaultWriter, VaultReader, _content_hash, _tag_blocks


@pytest.fixture
def vault_dir(tmp_path):
    """Provide a temporary vault directory."""
    return tmp_path / "vault"


class TestContentHash:
    def test_deterministic(self):
        assert _content_hash("hello world") == _content_hash("hello world")

    def test_whitespace_normalised(self):
        assert _content_hash("hello  world") == _content_hash("hello world")
        assert _content_hash("hello\nworld") == _content_hash("hello world")

    def test_case_insensitive(self):
        assert _content_hash("Hello World") == _content_hash("hello world")


class TestTagBlocks:
    def test_tags_each_paragraph(self):
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        tagged, block_map = _tag_blocks(content)

        assert len(block_map) == 3
        assert "Paragraph one." in list(block_map.values())
        assert "Paragraph two." in list(block_map.values())
        assert "Paragraph three." in list(block_map.values())

        # Each paragraph in the tagged output should have a ^block-XXXX
        for part in tagged.split("\n\n"):
            assert "^block-" in part

    def test_empty_content(self):
        tagged, block_map = _tag_blocks("")
        assert tagged == ""
        assert len(block_map) == 0


class TestVaultWriter:
    def test_write_creates_file(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        path = writer.write_document(
            ticker="AAPL",
            content="This is a test paragraph.\n\nThis is another paragraph.",
            metadata={
                "source_type": "news",
                "url": "https://example.com/article",
                "sentiment": "bullish",
            },
        )

        assert path.exists()
        assert path.suffix == ".md"
        assert path.parent.name == "AAPL"

    def test_deduplication_by_hash(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        content = "Identical content for dedup test."
        path1 = writer.write_document(ticker="AAPL", content=content)
        path2 = writer.write_document(ticker="AAPL", content=content)

        # Same content → same filename → same path (no duplicate created)
        assert path1 == path2

    def test_different_content_different_files(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        path1 = writer.write_document(ticker="AAPL", content="Content A")
        path2 = writer.write_document(ticker="AAPL", content="Content B")

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


class TestVaultReader:
    def test_roundtrip(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        path = writer.write_document(
            ticker="MSFT",
            content="Revenue grew 15% YoY.\n\nMargins expanded to 45%.",
            metadata={
                "source_type": "10-K",
                "url": "https://sec.gov/msft",
                "sentiment": "neutral",
            },
        )

        reader = VaultReader(root=vault_dir)
        doc = reader.read_document(path)

        assert doc.ticker == "MSFT"
        assert doc.source_type == "10-K"
        assert doc.url == "https://sec.gov/msft"
        assert len(doc.block_map) == 2
        assert any("Revenue grew" in v for v in doc.block_map.values())
        assert any("Margins expanded" in v for v in doc.block_map.values())

    def test_list_documents(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        writer.write_document(ticker="GOOG", content="Doc 1")
        writer.write_document(ticker="GOOG", content="Doc 2")
        writer.write_document(ticker="AMZN", content="Doc 3")

        reader = VaultReader(root=vault_dir)
        goog_docs = reader.list_documents("GOOG")
        amzn_docs = reader.list_documents("AMZN")

        assert len(goog_docs) == 2
        assert len(amzn_docs) == 1

    def test_resolve_block(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        path = writer.write_document(
            ticker="META",
            content="First paragraph.\n\nSecond paragraph.",
        )

        reader = VaultReader(root=vault_dir)
        doc = reader.read_document(path)

        # Resolve each block ID
        for block_id, expected_text in doc.block_map.items():
            resolved = reader.resolve_block(path, block_id)
            assert resolved == expected_text

    def test_resolve_missing_block(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        path = writer.write_document(ticker="TSLA", content="Some content.")

        reader = VaultReader(root=vault_dir)
        resolved = reader.resolve_block(path, "block-nonexistent")
        assert resolved is None

    def test_mark_archived(self, vault_dir):
        writer = VaultWriter(root=vault_dir)
        path = writer.write_document(ticker="NVDA", content="GPU revenue data.")

        reader = VaultReader(root=vault_dir)
        reader.mark_archived(path)

        doc = reader.read_document(path)
        assert doc.archived is True
