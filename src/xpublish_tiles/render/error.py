import io

from PIL import Image, ImageDraw

from xpublish_tiles.types import ImageFormat


def render_error_image(
    message: str, *, width: int, height: int, format: ImageFormat
) -> io.BytesIO:
    buffer = io.BytesIO()
    img = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), message, fill=(255, 255, 255, 255))
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer
