from fastapi import FastAPI
import uvicorn
from routes import index_router, query_document_router
from middleware.middleware import add_middleware

app = FastAPI()

app.middleware("http")(add_middleware)
app.include_router(index_router)
app.include_router(query_document_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)