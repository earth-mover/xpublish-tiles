from fastapi import APIRouter

xyz_tiles_router = APIRouter()


@xyz_tiles_router.get("/{x}/{y}/{z}")
async def xyz_tile(x: int, y: int, z: int):
    # TODO: Implement XYZ Tiles
    return {"message": "Hello World"}
