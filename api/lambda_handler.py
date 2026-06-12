"""AWS Lambda entrypoint — adapts the FastAPI ASGI app to Lambda via Mangum.

Used by the query/chat Lambda container image. Streaming responses (the `/chat`
SSE endpoint) work when the function is fronted by a Lambda Function URL with
response streaming enabled.
"""

from mangum import Mangum

from api.server import app

handler = Mangum(app, lifespan="auto")
