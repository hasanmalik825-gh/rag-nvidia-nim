from constants import IP_WHITELIST
from fastapi import Request, HTTPException, status

async def add_middleware(request: Request, call_next):
    if request.client.host not in IP_WHITELIST:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    print(f"Request from {request.client.host} (middleware)")
    return await call_next(request)