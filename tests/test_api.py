import unittest
from fastapi.testclient import TestClient
from main import app


class TestRAGPlatformAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        """Verify health check endpoint returns 200 and operational details."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["engine"], "Groq")
        self.assertIn("model", data)
        self.assertIn("vector_store_initialized", data)

    def test_query_blank_question(self):
        """Verify blank questions return 400 Bad Request."""
        response = self.client.post("/api/v1/query", json={"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be blank", response.json()["detail"])

    def test_upload_invalid_file_type(self):
        """Verify uploading non-PDF files returns 400 Bad Request."""
        response = self.client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"Invalid text content", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("PDF", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

