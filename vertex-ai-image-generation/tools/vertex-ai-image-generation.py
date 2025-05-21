import base64
import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from google import genai
from google.oauth2 import service_account

class GenaiImageGenerationTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        model: str = tool_parameters.get("model")
        prompt: str = tool_parameters.get("prompt")
        negative_prompt: str = tool_parameters.get("negative_prompt")
        number_of_images: int = tool_parameters.get("number_of_images")
        enhance_prompt: bool = tool_parameters.get("enhance_prompt")
        aspect_ratio: str = tool_parameters.get("aspect_ratio") or "1:1"
        guidance_scale: float | None = tool_parameters.get("guidance_scale")

        # Credentials
        service_account_info = json.loads(base64.b64decode(self.runtime.credentials["vertex_service_account_key"]))
        project = self.runtime.credentials["vertex_project_id"]
        location = self.runtime.credentials["vertex_location"]

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=service_account.Credentials.from_service_account_info(
                service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            ),
        )

        person_generation = "ALLOW_ADULT" if model == "imagen-3.0-generate-002" else None
        safety_filter_level = "BLOCK_ONLY_HIGH"
        mime_type = "image/jpeg"


        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(
                add_watermark=False,
                include_rai_reason=True,
                negative_prompt=negative_prompt,
                number_of_images=number_of_images,
                aspect_ratio=aspect_ratio,
                enhance_prompt=enhance_prompt,
                guidance_scale=guidance_scale,
                output_mime_type=mime_type,
                output_compression_quality=95,
                person_generation=person_generation,
                safety_filter_level=safety_filter_level,
            ),
        )

        for i, generated_image in enumerate(response.generated_images):
            if rai_filtered_reason := generated_image.rai_filtered_reason:
                raise ValueError(
                    f"Image generation failed with reason: {rai_filtered_reason}"
                )
            if generated_image.image.image_bytes:
                yield self.create_blob_message(
                    blob=generated_image.image.image_bytes, meta={"mime_type": mime_type}
                )
