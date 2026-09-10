from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraPacket:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    sim_time: float


def _detect_red_object(frame_bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_1 = np.asarray([0, 70, 45], dtype=np.uint8)
    upper_1 = np.asarray([15, 255, 255], dtype=np.uint8)
    lower_2 = np.asarray([165, 70, 45], dtype=np.uint8)
    upper_2 = np.asarray([179, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_1, upper_1),
        cv2.inRange(hsv, lower_2, upper_2),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < 40.0:
        return None, None

    x, y, width, height = cv2.boundingRect(contour)
    center = (x + width // 2, y + height // 2)
    return (x, y, width, height), center


def _camera_worker(
    scene_path: str,
    camera_name: str,
    width: int,
    height: int,
    window_name: str,
    detect_red: bool,
    packet_queue: Any,
    stop_event: Any,
) -> None:
    import cv2
    import mujoco

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
    if camera_id < 0:
        stop_event.set()
        raise ValueError(f"MJCF 中不存在相机: {camera_name}")

    renderer = mujoco.Renderer(model, height=height, width=width)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    last_wall_time = time.perf_counter()
    fps_ema = 0.0
    latest_packet: CameraPacket | None = None

    try:
        while not stop_event.is_set():
            try:
                while True:
                    latest_packet = packet_queue.get_nowait()
            except queue.Empty:
                pass

            if latest_packet is None:
                time.sleep(0.005)
                continue

            data.qpos[:] = latest_packet.qpos
            data.qvel[:] = latest_packet.qvel
            data.ctrl[:] = latest_packet.ctrl
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera_name)
            frame_rgb = np.asarray(renderer.render())
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            center_x = width // 2
            center_y = height // 2
            cv2.drawMarker(
                frame_bgr,
                (center_x, center_y),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
            )

            if detect_red:
                box, target_center = _detect_red_object(frame_bgr)
                if box is not None and target_center is not None:
                    x, y, box_w, box_h = box
                    cv2.rectangle(
                        frame_bgr,
                        (x, y),
                        (x + box_w, y + box_h),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(frame_bgr, target_center, 4, (0, 255, 255), -1)
                    cv2.line(
                        frame_bgr,
                        (center_x, center_y),
                        target_center,
                        (255, 255, 0),
                        1,
                    )

            now = time.perf_counter()
            dt = max(now - last_wall_time, 1e-6)
            instant_fps = 1.0 / dt
            fps_ema = instant_fps if fps_ema <= 0.0 else 0.90 * fps_ema + 0.10 * instant_fps
            last_wall_time = now

            suction_on = bool(latest_packet.ctrl[-1] > 0.5)
            cv2.putText(
                frame_bgr,
                f"camera={camera_name}  sim={latest_packet.sim_time:6.2f}s  fps={fps_ema:4.1f}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                f"suction={'ON' if suction_on else 'OFF'}  q/ESC: close",
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255) if suction_on else (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop_event.set()
                break
    finally:
        renderer.close()
        cv2.destroyAllWindows()


class OpenCVCameraProcess:
    def __init__(
        self,
        scene_path: Path,
        camera_name: str,
        width: int,
        height: int,
        window_name: str,
        detect_red: bool,
    ) -> None:
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=1)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_camera_worker,
            args=(
                str(scene_path),
                camera_name,
                width,
                height,
                window_name,
                detect_red,
                self._queue,
                self._stop_event,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def push(self, packet: CameraPacket) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            pass

    def should_stop(self) -> bool:
        return bool(self._stop_event.is_set())

    def close(self) -> None:
        self._stop_event.set()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._queue.close()
