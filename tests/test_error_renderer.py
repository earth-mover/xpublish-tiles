import pytest

from xpublish_tiles.render import render_error_image
from xpublish_tiles.types import ImageFormat


@pytest.mark.parametrize("format", [ImageFormat.PNG, ImageFormat.JPEG])
def test_render_error_image(format):
    width, height = 256, 256
    message = "Test Error"
    format = ImageFormat.PNG

    buffer = render_error_image(message, width=width, height=height, format=format)

    # Check that the buffer is not empty
    assert buffer.getbuffer().nbytes > 0

    # Check that the buffer starts with PNG signature
    if format == ImageFormat.PNG:
        assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    elif format == ImageFormat.JPEG:
        assert buffer.getvalue().startswith(b"\xff\xd8")
