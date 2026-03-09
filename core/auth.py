import os
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai.types import HttpOptions

print("--- Initializing Core Authentication ---")

# Load .env from the plugin directory (ComfyUI_Nano_Banana/)
_plugin_dir = Path(__file__).resolve().parent.parent
_env_path = _plugin_dir / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    print(f"✅ Loaded .env from {_env_path}")
else:
    # Fallback: try current working directory
    load_dotenv()
    print(f"⚠️ No .env found at {_env_path}, trying default locations")

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOM_BASE_URL = os.getenv("CUSTOM_BASE_URL")  # e.g. https://api.apiyi.com

def detect_approach():
    """
    Detect which approach to use based on available credentials.

    Returns:
        str: "VERTEXAI" if PROJECT_ID and LOCATION are available,
             "API" if GOOGLE_API_KEY is available,
             raises Exception if no valid credentials found
    """
    if PROJECT_ID and LOCATION:
        return "VERTEXAI"
    elif GOOGLE_API_KEY:
        return "API"
    else:
        raise Exception("No valid credentials found. Need either PROJECT_ID + LOCATION for VertexAI or GOOGLE_API_KEY for API approach")


def create_client(approach, model_name=None):
    """
    Create a genai.Client with proper configuration.
    Supports custom base URLs (e.g. apiyi.com) via CUSTOM_BASE_URL env var.
    """
    if approach == "VERTEXAI":
        if not PROJECT_ID or not LOCATION:
            raise ValueError("PROJECT_ID or LOCATION not configured in .env for Vertex AI approach")

        # Use global location for nanobanana models
        location = LOCATION
        if model_name and "gemini-3" in model_name:
            location = "global"

        return genai.Client(vertexai=True, project=PROJECT_ID, location=location)

    else:  # API approach
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not configured in .env for API approach")

        if CUSTOM_BASE_URL:
            return genai.Client(
                api_key=GOOGLE_API_KEY,
                http_options=HttpOptions(
                    api_version='v1',
                    base_url=CUSTOM_BASE_URL
                )
            )
        else:
            return genai.Client(api_key=GOOGLE_API_KEY)


try:
    if PROJECT_ID and LOCATION:
        import google.auth
        import vertexai
        from google.cloud import aiplatform

        CREDENTIALS, discovered_project_id = google.auth.default()
        if not PROJECT_ID and discovered_project_id:
            PROJECT_ID = discovered_project_id

        if PROJECT_ID and LOCATION:
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=CREDENTIALS)
            print("✅ NanoBanana with Vertex AI (Legacy SDK) Initialized.")

            aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=CREDENTIALS)
            print("✅ NanoBanana with AI Platform SDK Initialized.")
        else:
            print("\033[93mNanoBanana Config Warning: PROJECT_ID or LOCATION not found. Nodes will fail.\033[0m")
    elif GOOGLE_API_KEY:
        if CUSTOM_BASE_URL:
            print(f"✅ NanoBanana with API approach initialized (custom endpoint: {CUSTOM_BASE_URL})")
        else:
            print("✅ NanoBanana with API approach initialized.")
    else:
        print("\033[93mNanoBanana Config Warning: No credentials found.\033[0m")

except Exception as e:
    print(f"\033[91mAn unexpected error occurred during NanoBanana initialization: {e}\033[0m")
