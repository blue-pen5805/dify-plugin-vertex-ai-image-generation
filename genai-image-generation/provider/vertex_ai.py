from typing import Any
import base64
import json

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from google import genai
from google.oauth2 import service_account

class GenaiImageGenerationProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            service_account_info = json.loads(base64.b64decode(credentials["vertex_service_account_key"]))
            project = credentials["vertex_project_id"]
            location = credentials["vertex_location"]

            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=service_account.Credentials.from_service_account_info(
                    service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                ),
            )

            client.models.list(config={'page_size': 1})
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
