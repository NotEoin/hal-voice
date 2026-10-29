"""The HAL-9000 red eye — a pygame window whose lens reacts to the interaction state.

States (driven by StateBus from the loop):
  idle      — slow, shallow "breathing" glow.
  listening — bright, steady, fully open (HAL is attending to you).
  thinking  — slow irregular flicker (HAL is… considering).
  speaking  — the lens pulses with the audio envelope (level 0..1 from the player).

Pure cosmetic, CPU-only. If pygame or a display is unavailable, the loop runs headless.
Run standalone to preview the animation:  python -m halvoice.ui_eye --demo
"""

from __future__ import annotations

import math
import sys
import time

from . import config
from .util import StateBus


class RedEye:
    def __init__(self, bus: StateBus) -> None:
        self.bus = bus
        cfg = config.load()
        self.w = int(cfg.get("ui.width", 480))
        self.h = int(cfg.get("ui.height", 480))
        self.fullscreen = bool(cfg.get("ui.fullscreen", False))
        self.color = tuple(cfg.get("ui.eye_color", [220, 30, 20]))

    def run(self) -> None:
        import pygame

        pygame.init()
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        screen = pygame.display.set_mode((self.w, self.h), flags)
        pygame.display.set_caption("HAL 9000")
        clock = pygame.time.Clock()
        cx, cy = self.w // 2, self.h // 2
        outer_r = int(min(self.w, self.h) * 0.42)
        t0 = time.time()
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
            state, level = self.bus.drain()
            t = time.time() - t0
            self._draw(pygame, screen, cx, cy, outer_r, state, level, t)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

    def _intensity(self, state: str, level: float, t: float) -> float:
        if state == "listening":
            return 1.0
        if state == "thinking":
            # irregular slow flicker
            return 0.55 + 0.18 * math.sin(t * 7) * math.sin(t * 2.3)
        if state == "speaking":
            return max(0.35, min(1.0, 0.45 + level))
        # idle: slow breathe
        return 0.30 + 0.10 * (math.sin(t * 1.2) * 0.5 + 0.5)

    def _draw(self, pygame, screen, cx, cy, outer_r, state, level, t) -> None:
        screen.fill((6, 6, 8))  # near-black faceplate
        r, g, b = self.color
        k = self._intensity(state, level, t)

        # concentric housing rings (the famous gold/grey bezel, abstracted)
        pygame.draw.circle(screen, (24, 24, 28), (cx, cy), outer_r + 14)
        pygame.draw.circle(screen, (40, 38, 34), (cx, cy), outer_r + 6)
        pygame.draw.circle(screen, (12, 12, 14), (cx, cy), outer_r)

        # outer red glow (additive-ish, drawn as fading rings)
        glow = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        for i in range(18, 0, -1):
            a = int(8 * k * i / 18)
            rad = int(outer_r * (0.55 + i / 26))
            pygame.draw.circle(glow, (r, g, b, a), (cx, cy), rad)
        screen.blit(glow, (0, 0))

        # the lens core
        core_r = int(outer_r * 0.42)
        core = (int(r * k), int(g * k), int(b * k))
        pygame.draw.circle(screen, core, (cx, cy), core_r)
        # bright hot center
        hot = (min(255, int(r * k) + 60), int(g * k) + 30, int(b * k) + 20)
        pygame.draw.circle(screen, hot, (cx, cy), int(core_r * 0.5))
        # tiny white specular highlight, offset (the "eye" feel)
        pygame.draw.circle(screen, (255, 235, 230),
                           (cx - int(core_r * 0.28), cy - int(core_r * 0.30)),
                           max(2, int(core_r * 0.10)))


def demo() -> None:
    """Cycle through the states so you can see the animation without the full loop."""
    bus = StateBus()

    def driver():
        seq = ["idle", "listening", "thinking", "speaking"]
        i = 0
        while True:
            st = seq[i % len(seq)]
            if st == "speaking":
                for _ in range(40):
                    bus.set("speaking", abs(math.sin(time.time() * 6)))
                    time.sleep(0.05)
            else:
                bus.set(st)  # type: ignore[arg-type]
                time.sleep(2.5)
            i += 1

    import threading

    threading.Thread(target=driver, daemon=True).start()
    RedEye(bus).run()


def main() -> None:
    if "--demo" in sys.argv:
        demo()
    else:
        RedEye(StateBus()).run()


if __name__ == "__main__":
    main()
