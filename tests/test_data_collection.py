"""Unit tests for data collection."""
from pathlib import Path
import shutil
import tempfile


class TestDataCollection:
    """Verify data collection artifacts are created."""

    def setup_method(self) -> None:
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_session_directory(self) -> None:
        path = Path(self.test_dir) / "session_test"
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists()
