from __future__ import annotations

import hashlib
import os
import struct
import zlib
from pathlib import Path
from threading import Event

import cv2
import numpy as np


FORMATS = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF",
    ".tiff": "TIFF", ".bmp": "BMP", ".webp": "WEBP",
}
PNG = b"\x89PNG\r\n\x1a\n"


class Cancelled(Exception):
    """A cooperative operation was cancelled."""


def check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("操作已取消。")


def absolute_path(path: str | Path) -> Path:
    """Use the extended Win32 namespace for a long local or UNC path."""
    name = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.name != "nt" or len(name) < 220 or name.startswith("\\\\?\\"):
        return Path(name)
    return Path("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name)


def display_path(path: Path) -> Path:
    name = str(path)
    if os.name == "nt":
        if name.startswith("\\\\?\\UNC\\"):
            return Path("\\\\" + name[8:])
        if name.startswith("\\\\?\\"):
            return Path(name[4:])
    return path


def canonical_path(path: Path) -> Path:
    return display_path(absolute_path(path).resolve())


def sha256(path: Path, cancel: Event | None = None) -> str:
    digest = hashlib.sha256()
    with absolute_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest()


def image_format(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(PNG):
        return "PNG"
    if data.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "TIFF"
    if data.startswith(b"BM"):
        return "BMP"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def check_png(data: bytes) -> None:
    if not data.startswith(PNG):
        raise ValueError("输出缺少 PNG 签名。")
    offset = len(PNG)
    while offset + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 8 + length
        if end + 4 > len(data) or zlib.crc32(memoryview(data)[offset + 4:end]) != struct.unpack_from(">I", data, end)[0]:
            raise ValueError("PNG 数据不完整或 CRC 校验失败。")
        offset = end + 4
        if kind == b"IEND" and length == 0 and offset == len(data):
            return
    raise ValueError("PNG 缺少完整的 IEND。")


def decode(data: bytes, path: Path, *, engine_input: bool = False, preserve_depth: bool = False) -> np.ndarray:
    if not data:
        raise ValueError(f"图片为空 ({path})")
    flags = cv2.IMREAD_ANYCOLOR if engine_input else cv2.IMREAD_UNCHANGED
    if engine_input and preserve_depth:
        flags |= cv2.IMREAD_ANYDEPTH
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    except cv2.error as error:
        raise ValueError(f"无法解码图片 ({path})") from error
    if image is None or not image.size:
        raise ValueError(f"无法解码图片 ({path})")
    return image


def read_image(path: Path, *, engine_input: bool = False, preserve_depth: bool = False) -> np.ndarray:
    path = absolute_path(path)
    return decode(path.read_bytes(), path, engine_input=engine_input, preserve_depth=preserve_depth)


def png_info(path: Path) -> dict:
    data = absolute_path(path).read_bytes()
    check_png(data)
    image = decode(data, path)
    return {"width": int(image.shape[1]), "height": int(image.shape[0]),
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def normalized_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return image.astype(np.float32) * (255.0 / 65535)
    raise ValueError(f"图像评价支持 8 位或 16 位无符号图像，实际类型为 {image.dtype}。")
