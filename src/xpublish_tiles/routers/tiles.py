from fastapi import APIRouter

tiles_router = APIRouter()


@tiles_router.get("/")
async def xyz_tile():
    # TODO: Implement XYZ Tiles
    return {"message": "Hello World from tiles router"}
