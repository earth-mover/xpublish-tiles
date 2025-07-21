import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xpublish_tiles.routers.xyz import xyz_tiles_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(xyz_tiles_router)
    return TestClient(app)


def test_xyz_tile_endpoint(client):
    response = client.get("/1/2/3")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
