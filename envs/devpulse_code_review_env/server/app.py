from openenv.core.env_server import create_fastapi_app
from .environment import DevPulseCodeReviewEnvironment

app = create_fastapi_app(DevPulseCodeReviewEnvironment)