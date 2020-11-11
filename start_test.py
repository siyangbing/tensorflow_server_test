import os
import uvicorn

uvicorn.run ("main:app",host="192.168.3.124 ")
# uvicorn.run("main:app", host="0.0.0.0", port=8000)