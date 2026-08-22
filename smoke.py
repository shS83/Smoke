#!/usr/bin/env python3
"""
smoke.py v0.1

Ensimmäinen askel kohti realistisempaa savusimulaatiota:
- 2D density-kenttä
- velocity-kenttä
- semi-Lagrangian advektio
- lämpimän savun kaltainen ylöspäin suuntautuva buoyancy
- pehmeä savulähde alhaalla keskellä
- kevyt sivuttaisliike, jotta pylväs ei ole täysin suora

Seuraava versio:
- paineen ratkaisu / divergence-free velocity
- vorticity confinement
- parempi turbulentti rakenne

Ohjaus:
    ESC        lopeta
    SPACE      pause
    R          reset
    Vasen hiiri: lisää savua hiiren kohdalle
"""

import math
import sys

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Asetukset
# ---------------------------------------------------------------------------

WINDOW_W = 1280
WINDOW_H = 720

SIM_W = 240
SIM_H = 135

FPS = 60

# Savun käyttäytyminen
BUOYANCY = 24.0
VELOCITY_DAMPING = 0.992
DENSITY_DISSIPATION = 0.994

# Savulähde
SOURCE_X = SIM_W * 0.5
SOURCE_Y = SIM_H - 7.0
SOURCE_RADIUS_X = 9.0
SOURCE_RADIUS_Y = 3.5
SOURCE_STRENGTH = 2.8
SOURCE_UPWARD_SPEED = 18.0

# Renderöinti
BACKGROUND = np.array([5.0, 6.0, 8.0], dtype=np.float32)
SMOKE_COLOR = np.array([195.0, 198.0, 202.0], dtype=np.float32)
ABSORPTION = 1.65


# ---------------------------------------------------------------------------
# Etukäteen lasketut ruudukot
# ---------------------------------------------------------------------------

GRID_X, GRID_Y = np.meshgrid(
    np.arange(SIM_W, dtype=np.float32),
    np.arange(SIM_H, dtype=np.float32),
)


def bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Näytteistää 2D-kentän bilineaarisesti kelluvista koordinaateista."""

    x = np.clip(x, 0.0, SIM_W - 1.001)
    y = np.clip(y, 0.0, SIM_H - 1.001)

    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, SIM_W - 1)
    y1 = np.minimum(y0 + 1, SIM_H - 1)

    sx = x - x0
    sy = y - y0

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    top = f00 * (1.0 - sx) + f10 * sx
    bottom = f01 * (1.0 - sx) + f11 * sx

    return top * (1.0 - sy) + bottom * sy


def advect(field: np.ndarray, vx: np.ndarray, vy: np.ndarray, dt: float) -> np.ndarray:
    """
    Semi-Lagrangian advektio.

    Jokaiselle solulle etsitään taaksepäin kohta, josta materiaali
    saapui nykyiseen pisteeseen.
    """

    previous_x = GRID_X - vx * dt
    previous_y = GRID_Y - vy * dt

    return bilinear_sample(field, previous_x, previous_y)


def gaussian_blob(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Pehmeä elliptinen Gauss-lähde."""

    dx = (GRID_X - cx) / rx
    dy = (GRID_Y - cy) / ry
    return np.exp(-(dx * dx + dy * dy) * 2.0).astype(np.float32)


SOURCE_MASK = gaussian_blob(
    SOURCE_X,
    SOURCE_Y,
    SOURCE_RADIUS_X,
    SOURCE_RADIUS_Y,
)


class SmokeSimulation:
    def __init__(self) -> None:
        self.density = np.zeros((SIM_H, SIM_W), dtype=np.float32)
        self.vx = np.zeros_like(self.density)
        self.vy = np.zeros_like(self.density)
        self.time = 0.0

    def reset(self) -> None:
        self.density.fill(0.0)
        self.vx.fill(0.0)
        self.vy.fill(0.0)
        self.time = 0.0

    def add_source(self, dt: float) -> None:
        """
        Lisää savua ja pystysuuntaista nopeutta alareunan lähteeseen.

        Lähteen sivuttaisnopeus elää hieman ajan mukana, jotta savupatsas
        lähtee jo v0.1:ssä luonnollisemmin vaeltelemaan.
        """

        pulse = 0.88 + 0.12 * math.sin(self.time * 2.4)
        source = SOURCE_MASK * pulse

        self.density += source * SOURCE_STRENGTH * dt * 10.0

        self.vy -= source * SOURCE_UPWARD_SPEED * dt * 5.0

        sway = (
            math.sin(self.time * 1.31) * 5.5
            + math.sin(self.time * 0.47 + 1.7) * 2.5
        )
        self.vx += source * sway * dt * 4.0

    def add_mouse_smoke(self, mouse_pos: tuple[int, int], window_size: tuple[int, int], dt: float) -> None:
        """Lisää savua hiiren kohdalle."""

        mx, my = mouse_pos
        ww, wh = window_size

        if ww <= 0 or wh <= 0:
            return

        sx = mx / ww * SIM_W
        sy = my / wh * SIM_H

        blob = gaussian_blob(sx, sy, 5.0, 5.0)

        self.density += blob * 5.0 * dt
        self.vy -= blob * 28.0 * dt

    def step(self, dt: float) -> None:
        self.time += dt

        self.add_source(dt)

        # ---------------------------------------------------------------
        # 1. Buoyancy:
        # tiheä savu saa tässä ensimmäisessä versiossa nostetta ylöspäin.
        # ---------------------------------------------------------------

        self.vy -= self.density * BUOYANCY * dt

        # ---------------------------------------------------------------
        # 2. Kevyt "tuuli", joka vaihtelee korkeuden ja ajan mukana.
        # Tämä EI vielä ole oikea turbulenssimalli.
        # ---------------------------------------------------------------

        height_wave = np.sin(GRID_Y * 0.095 + self.time * 1.15)
        second_wave = np.sin(GRID_Y * 0.041 - self.time * 0.73 + 2.1)

        self.vx += (
            height_wave * 0.55
            + second_wave * 0.30
        ).astype(np.float32) * self.density * dt * 8.0

        # ---------------------------------------------------------------
        # 3. Kuljeta velocity-kenttää sen omalla virtauksella.
        # ---------------------------------------------------------------

        old_vx = self.vx
        old_vy = self.vy

        self.vx = advect(old_vx, old_vx, old_vy, dt)
        self.vy = advect(old_vy, old_vx, old_vy, dt)

        # ---------------------------------------------------------------
        # 4. Kuljeta savun density velocity-kentässä.
        # ---------------------------------------------------------------

        self.density = advect(self.density, self.vx, self.vy, dt)

        # ---------------------------------------------------------------
        # 5. Häviöt.
        # ---------------------------------------------------------------

        self.vx *= VELOCITY_DAMPING
        self.vy *= VELOCITY_DAMPING
        self.density *= DENSITY_DISSIPATION

        # Rajaa density järkevälle alueelle.
        np.clip(self.density, 0.0, 4.0, out=self.density)

        # Alareunaan ei anneta kerääntyä ikuista savumattoa.
        self.density[-1, :] *= 0.80

    def render_array(self) -> np.ndarray:
        """
        Muuttaa density-kentän RGB-kuvaksi.

        alpha = 1 - exp(-density * absorption)

        Tämä antaa paljon luonnollisemman läpikuultavuuden kuin suora
        density * 255.
        """

        alpha = 1.0 - np.exp(-self.density * ABSORPTION)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Hieman kirkkaampi keskitiheys, mutta ei puhtaan valkoiseksi.
        brightness = np.clip(
            0.62 + self.density * 0.12,
            0.62,
            1.0,
        )

        smoke = SMOKE_COLOR[None, None, :] * brightness[:, :, None]

        rgb = (
            BACKGROUND[None, None, :] * (1.0 - alpha[:, :, None])
            + smoke * alpha[:, :, None]
        )

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # pygame.surfarray haluaa muodon (width, height, channels)
        return np.transpose(rgb, (1, 0, 2))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("smoke.py v0.1")

    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H),
        pygame.RESIZABLE,
    )

    clock = pygame.time.Clock()
    simulation = SmokeSimulation()

    paused = False
    running = True

    while running:
        # Rajoitetaan dt, jotta ikkunan liikuttelu tms. ei räjäytä simulaatiota.
        dt = min(clock.tick(FPS) / 1000.0, 1.0 / 30.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    simulation.reset()

        if not paused:
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                simulation.add_mouse_smoke(
                    pygame.mouse.get_pos(),
                    screen.get_size(),
                    dt,
                )

            simulation.step(dt)

        smoke_rgb = simulation.render_array()
        smoke_surface = pygame.surfarray.make_surface(smoke_rgb)

        scaled = pygame.transform.smoothscale(
            smoke_surface,
            screen.get_size(),
        )

        screen.blit(scaled, (0, 0))

        # Pieni debug-otsikko
        pygame.display.set_caption(
            f"smoke.py v0.1  |  {clock.get_fps():5.1f} FPS"
            + ("  |  PAUSED" if paused else "")
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
