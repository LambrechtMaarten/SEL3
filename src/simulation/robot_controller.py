import os.path
import threading
import time

import cv2
import numpy as np

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration
from src.environment.environment import Environment


# ─── Joystick instellingen ────────────────────────────────────────────────────
PANEL_SIZE = 300
CENTER     = PANEL_SIZE // 2
RADIUS     = 120
DOT_RADIUS = 12

BG_COLOR     = (30, 30, 30)
RING_COLOR   = (70, 70, 70)
RING_COLOR2  = (50, 50, 50)
DOT_COLOR    = (0, 200, 255)
CENTER_COLOR = (100, 100, 100)
TEXT_COLOR   = (200, 200, 200)
ACCENT_COLOR = (0, 200, 255)

# ─── Shared state ──────────────────────────────────────────────────────────────
state_lock = threading.Lock()
frame_lock = threading.Lock()

_dot_x, _dot_y = CENTER, CENTER
_control_input = (0.0, 0.0)
_latest_frame = None
_running = True


# ─── Mouse input ───────────────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global _dot_x, _dot_y, _control_input

    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_LBUTTON):
        dx = x - CENTER
        dy = y - CENTER
        dist = float(np.hypot(dx, dy))

        if dist > RADIUS:
            scale = RADIUS / dist
            dx *= scale
            dy *= scale
            dist = float(RADIUS)

        new_x = int(CENTER + dx)
        new_y = int(CENTER + dy)

        speed = dist / RADIUS
        angle = float(np.arctan2(-dy, dx))

        with state_lock:
            _dot_x, _dot_y = new_x, new_y
            _control_input = (angle, speed)

    elif event == cv2.EVENT_LBUTTONUP:
        with state_lock:
            _dot_x, _dot_y = CENTER, CENTER
            _control_input = (0.0, 0.0)


# ─── Joystick rendering ───────────────────────────────────────────────────────
def draw_joystick_panel(dot_x, dot_y, speed, angle_deg):
    panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), BG_COLOR, dtype=np.uint8)

    cv2.circle(panel, (CENTER, CENTER), RADIUS, RING_COLOR, 2)
    cv2.circle(panel, (CENTER, CENTER), RADIUS // 2, RING_COLOR2, 1)

    cv2.line(panel, (CENTER - RADIUS, CENTER), (CENTER + RADIUS, CENTER), RING_COLOR2, 1)
    cv2.line(panel, (CENTER, CENTER - RADIUS), (CENTER, CENTER + RADIUS), RING_COLOR2, 1)

    cv2.line(panel, (CENTER, CENTER), (dot_x, dot_y), ACCENT_COLOR, 1)

    cv2.circle(panel, (dot_x, dot_y), DOT_RADIUS, DOT_COLOR, -1)
    cv2.circle(panel, (dot_x, dot_y), DOT_RADIUS, (255, 255, 255), 1)
    cv2.circle(panel, (CENTER, CENTER), 4, CENTER_COLOR, -1)

    cv2.putText(panel, f"Speed: {speed:.2f}", (10, PANEL_SIZE - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.putText(panel, f"Angle: {angle_deg:.1f} deg", (10, PANEL_SIZE - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.putText(panel, "Klik in cirkel | ESC = stop", (10, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    return panel


# ─── Simulation thread ─────────────────────────────────────────────────────────
def simulation_loop(env, configuration, env_state):
    global _latest_frame, _running

    DT = 1.0 / 60.0
    next_t = time.time()

    while True:
        now = time.time()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += DT

        with state_lock:
            if not _running:
                break
            control = _control_input

        actions = configuration.controller.controller.act(
            None, control, configuration, env_state
        )

        env_state = env.step(actions, env_state)
        frame = env.render(env_state)

        if frame is not None:
            with frame_lock:
                _latest_frame = np.asarray(frame, dtype=np.uint8)


# ─── Main UI loop ──────────────────────────────────────────────────────────────
def simulate_controller_joystick(output_path: str):
    global _running

    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "silent_logger"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "crossover"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "angle"),
    )

    env = Environment(configuration)

    configuration.controller.controller.read_controller(
        os.path.join(output_path, "controller")
    )

    env_state = env.reset(configuration.random.split())

    sim_thread = threading.Thread(
        target=simulation_loop,
        args=(env, configuration, env_state),
        daemon=True,
    )
    sim_thread.start()

    cv2.namedWindow("Joystick", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Joystick", 400, 400)
    cv2.setMouseCallback("Joystick", mouse_callback)

    TARGET_FPS = 60
    frame_delay = 1.0 / TARGET_FPS
    last = time.time()

    while True:
        now = time.time()
        if now - last < frame_delay:
            continue
        last = now

        with state_lock:
            dot_x, dot_y = _dot_x, _dot_y
            angle, speed = _control_input

        with frame_lock:
            frame = _latest_frame

        panel = draw_joystick_panel(
            dot_x, dot_y,
            speed,
            float(np.degrees(angle))
        )

        cv2.imshow("Joystick", panel)

        if frame is not None:
            cv2.imshow("Simulation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            with state_lock:
                _running = False
            break

    sim_thread.join(timeout=2.0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    simulate_controller_joystick("output")