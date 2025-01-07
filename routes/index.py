from fastapi import APIRouter


index_router = APIRouter()


@index_router.get("/")
def index():
    return {"message": "API is up and running"}