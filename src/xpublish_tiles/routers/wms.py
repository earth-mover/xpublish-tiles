from fastapi import APIRouter

wms_router = APIRouter()


@wms_router.get("/")
def handle_wms_request():
    return {"message": "Hello from WMS!"}
