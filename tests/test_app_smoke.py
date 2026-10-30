import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_landing_page_renders_without_errors(self):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
        app.run(timeout=90)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.title[0].value, "📡 GeoTwin AI Indoor")


if __name__ == "__main__":
    unittest.main()
