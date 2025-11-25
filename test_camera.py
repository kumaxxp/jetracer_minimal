"""Simple camera test without UI."""

from __future__ import annotations

import time

from camera import JetCamera


def test_camera() -> None:
    print("Testing JetCamera wrapper...")
    camera = JetCamera(width=640, height=480, fps=30)
    if not camera.start():
        print("\u2717 Camera failed to start")
        return
    print("\u2713 Camera running; collecting frames...")
    try:
        for idx in range(30):
            frame = camera.read()
            if frame is None:
                print(f"Frame {idx}: None")
            else:
                print(
                    "Frame {i}: shape={shape}, dtype={dtype}, min={minv}, max={maxv}".format(
                        i=idx,
                        shape=frame.shape,
                        dtype=frame.dtype,
                        minv=frame.min(),
                        maxv=frame.max(),
                    )
                )
            time.sleep(0.1)
    finally:
        camera.stop()
        print("\u2713 Test complete")


if __name__ == "__main__":
    test_camera()
