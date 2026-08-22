#!/usr/bin/env python3
"""
smoke.py v0.2

Uutta v0.2:ssa:
- pressure projection
- divergence-kenttä
- Jacobi-iteraatioilla ratkaistava pressure-kenttä
- velocity pakotetaan lähes divergence-free-tilaan
- seinämä-/ulosvirtausrajat
- divergence RMS näkyy ikkunan otsikossa

Pohjana v0.1:
- 2D density-kenttä
- velocity-kenttä
- semi-Lagrangian advektio
- buoyancy
- pehmeä savulähde
- kevyt sivuttaisliike

Seuraava versio v0.3:
- curl / vorticity
- vorticity confinement
- oikeasti pyörteilevät savukiehkurat

Ohjaus:
    ESC         lopeta
    SPACE       pause
    R           reset
    Vasen hiiri lisää savua hiiren kohdalle
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

# Pressure projection
PRESSURE_ITERATIONS = 30

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


def bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
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


def advect(
    field: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Semi-Lagrangian advektio.

    Jokaisen ruudun nykyisestä sijainnista kuljetaan velocity-kenttää
    pitkin ajassa taaksepäin ja näytteistetään kenttä sieltä.
    """

    previous_x = GRID_X - vx * dt
    previous_y = GRID_Y - vy * dt

    return bilinear_sample(
        field,
        previous_x,
        previous_y,
    )


def gaussian_blob(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
) -> np.ndarray:
    """Pehmeä elliptinen Gauss-lähde."""

    dx = (GRID_X - cx) / rx
    dy = (GRID_Y - cy) / ry

    return np.exp(
        -(dx * dx + dy * dy) * 2.0
    ).astype(np.float32)


SOURCE_MASK = gaussian_blob(
    SOURCE_X,
    SOURCE_Y,
    SOURCE_RADIUS_X,
    SOURCE_RADIUS_Y,
)


class SmokeSimulation:
    def __init__(self) -> None:
        self.density = np.zeros(
            (SIM_H, SIM_W),
            dtype=np.float32,
        )

        self.vx = np.zeros_like(self.density)
        self.vy = np.zeros_like(self.density)

        # v0.2
        self.divergence = np.zeros_like(self.density)
        self.pressure = np.zeros_like(self.density)
        self.pressure_tmp = np.zeros_like(self.density)

        self.time = 0.0
        self.divergence_rms = 0.0

    def reset(self) -> None:
        self.density.fill(0.0)
        self.vx.fill(0.0)
        self.vy.fill(0.0)

        self.divergence.fill(0.0)
        self.pressure.fill(0.0)
        self.pressure_tmp.fill(0.0)

        self.time = 0.0
        self.divergence_rms = 0.0

    # -------------------------------------------------------------------
    # Savun lähteet
    # -------------------------------------------------------------------

    def add_source(self, dt: float) -> None:
        """
        Lisää savua ja alkunopeutta alareunan keskelle.

        Pieni ajan mukana muuttuva sway tekee lähteestä hieman elävän.
        """

        pulse = (
            0.88
            + 0.12 * math.sin(self.time * 2.4)
        )

        source = SOURCE_MASK * pulse

        self.density += (
            source
            * SOURCE_STRENGTH
            * dt
            * 10.0
        )

        self.vy -= (
            source
            * SOURCE_UPWARD_SPEED
            * dt
            * 5.0
        )

        sway = (
            math.sin(self.time * 1.31) * 5.5
            + math.sin(self.time * 0.47 + 1.7) * 2.5
        )

        self.vx += (
            source
            * sway
            * dt
            * 4.0
        )

    def add_mouse_smoke(
        self,
        mouse_pos: tuple[int, int],
        window_size: tuple[int, int],
        dt: float,
    ) -> None:
        """Lisää savua hiiren kohdalle."""

        mx, my = mouse_pos
        ww, wh = window_size

        if ww <= 0 or wh <= 0:
            return

        sx = mx / ww * SIM_W
        sy = my / wh * SIM_H

        blob = gaussian_blob(
            sx,
            sy,
            5.0,
            5.0,
        )

        self.density += blob * 5.0 * dt
        self.vy -= blob * 28.0 * dt

    # -------------------------------------------------------------------
    # Rajat
    # -------------------------------------------------------------------

    def apply_velocity_boundaries(self) -> None:
        """
        Sivut ja alareuna käyttäytyvät seinäminä.

        Yläreuna on avoin: ylöspäin liikkuva savu saa poistua
        simulaatioalueelta.
        """

        # Sivuseinillä ei normaalisuuntaista X-nopeutta.
        self.vx[:, 0] = 0.0
        self.vx[:, -1] = 0.0

        # Sivuseinillä pystynopeus jatkuu pehmeästi.
        self.vy[:, 0] = self.vy[:, 1]
        self.vy[:, -1] = self.vy[:, -2]

        # Alareuna suljettu.
        self.vy[-1, :] = 0.0
        self.vx[-1, :] = self.vx[-2, :]

        # Yläreuna avoin.
        self.vx[0, :] = self.vx[1, :]

        # Negatiivinen vy = ylöspäin.
        # Ulosvirtaus sallitaan, sisäänpäin tuleva virtaus estetään.
        self.vy[0, :] = np.minimum(
            self.vy[1, :],
            0.0,
        )

    def apply_pressure_boundaries(
        self,
        pressure: np.ndarray,
    ) -> None:
        """
        Paineelle Neumann-tyyppinen reunaehto:
        reunan paine kopioidaan viereisestä solusta.
        """

        pressure[:, 0] = pressure[:, 1]
        pressure[:, -1] = pressure[:, -2]

        pressure[0, :] = pressure[1, :]
        pressure[-1, :] = pressure[-2, :]

    # -------------------------------------------------------------------
    # v0.2: Pressure projection
    # -------------------------------------------------------------------

    def compute_divergence(self) -> None:
        """
        Laskee velocity-kentän divergenssin keskitettynä erotuksena.

        divergence > 0:
            virtaus pyrkii laajenemaan

        divergence < 0:
            virtaus pyrkii pakkautumaan

        Incompressible-fluidissa tavoitteena on divergence ~= 0.
        """

        self.divergence.fill(0.0)

        self.divergence[1:-1, 1:-1] = 0.5 * (
            self.vx[1:-1, 2:]
            - self.vx[1:-1, :-2]
            + self.vy[2:, 1:-1]
            - self.vy[:-2, 1:-1]
        )

    def solve_pressure(self) -> None:
        """
        Ratkaisee Poisson-yhtälöä:

            Laplacian(p) = divergence

        Jacobi-iteraatioilla.

        Tämä on v0.2:n varsinainen uusi osa.
        """

        self.pressure.fill(0.0)
        self.pressure_tmp.fill(0.0)

        p = self.pressure
        p_new = self.pressure_tmp
        div = self.divergence

        for _ in range(PRESSURE_ITERATIONS):
            p_new[1:-1, 1:-1] = 0.25 * (
                p[1:-1, :-2]
                + p[1:-1, 2:]
                + p[:-2, 1:-1]
                + p[2:, 1:-1]
                - div[1:-1, 1:-1]
            )

            self.apply_pressure_boundaries(p_new)

            p, p_new = p_new, p

        # Jacobi swapparit voivat jättää lopputuloksen kumpaan tahansa
        # taulukkoon riippuen iteraatioiden määrästä.
        if p is not self.pressure:
            self.pressure[:, :] = p

    def subtract_pressure_gradient(self) -> None:
        """
        Poistaa pressure-gradientin velocity-kentästä.

            u_new = u - grad(p)
        """

        p = self.pressure

        self.vx[1:-1, 1:-1] -= 0.5 * (
            p[1:-1, 2:]
            - p[1:-1, :-2]
        )

        self.vy[1:-1, 1:-1] -= 0.5 * (
            p[2:, 1:-1]
            - p[:-2, 1:-1]
        )

    def project_velocity(self) -> None:
        """
        Koko pressure projection:

        velocity
            -> divergence
            -> pressure solve
            -> subtract grad(p)
            -> lähes divergence-free velocity
        """

        self.apply_velocity_boundaries()

        self.compute_divergence()
        self.solve_pressure()
        self.subtract_pressure_gradient()

        self.apply_velocity_boundaries()

        # Mittaa projektion jälkeinen divergenssi debugia varten.
        self.compute_divergence()

        interior = self.divergence[1:-1, 1:-1]

        self.divergence_rms = float(
            np.sqrt(
                np.mean(interior * interior)
            )
        )

    # -------------------------------------------------------------------
    # Simulaatio
    # -------------------------------------------------------------------

    def step(self, dt: float) -> None:
        self.time += dt

        # 1. Lisää savulähde.
        self.add_source(dt)

        # 2. Buoyancy.
        # Negatiivinen Y-nopeus tarkoittaa ylöspäin.
        self.vy -= (
            self.density
            * BUOYANCY
            * dt
        )

        # 3. Kevyt keinotekoinen sivuttaisliike.
        # Tämä poistuu / muuttuu v0.3:ssa, kun mukaan tulee vorticity.
        height_wave = np.sin(
            GRID_Y * 0.095
            + self.time * 1.15
        )

        second_wave = np.sin(
            GRID_Y * 0.041
            - self.time * 0.73
            + 2.1
        )

        self.vx += (
            height_wave * 0.55
            + second_wave * 0.30
        ).astype(np.float32) * self.density * dt * 8.0

        # 4. Advect velocity omalla velocity-kentällään.
        old_vx = self.vx.copy()
        old_vy = self.vy.copy()

        self.vx = advect(
            old_vx,
            old_vx,
            old_vy,
            dt,
        )

        self.vy = advect(
            old_vy,
            old_vx,
            old_vy,
            dt,
        )

        # 5. UUSI v0.2:
        # pakota velocity lähes divergence-free-tilaan.
        self.project_velocity()

        # 6. Advect smoke density projektioidulla velocity-kentällä.
        self.density = advect(
            self.density,
            self.vx,
            self.vy,
            dt,
        )

        # 7. Häviöt.
        self.vx *= VELOCITY_DAMPING
        self.vy *= VELOCITY_DAMPING

        self.density *= DENSITY_DISSIPATION

        # Rajaa density järkevälle alueelle.
        np.clip(
            self.density,
            0.0,
            4.0,
            out=self.density,
        )

        # Alareunaan ei anneta kerääntyä ikuista savumattoa.
        self.density[-1, :] *= 0.80

    # -------------------------------------------------------------------
    # Renderöinti
    # -------------------------------------------------------------------

    def render_array(self) -> np.ndarray:
        """
        Muuttaa density-kentän RGB-kuvaksi.

        Optinen tiheys:

            alpha = 1 - exp(-density * absorption)

        Ohut savu jää läpikuultavaksi ja paksu savu peittyy luonnollisesti.
        """

        alpha = (
            1.0
            - np.exp(
                -self.density * ABSORPTION
            )
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0,
        )

        brightness = np.clip(
            0.62
            + self.density * 0.12,
            0.62,
            1.0,
        )

        smoke = (
            SMOKE_COLOR[None, None, :]
            * brightness[:, :, None]
        )

        rgb = (
            BACKGROUND[None, None, :]
            * (1.0 - alpha[:, :, None])
            + smoke
            * alpha[:, :, None]
        )

        rgb = np.clip(
            rgb,
            0,
            255,
        ).astype(np.uint8)

        # pygame.surfarray käyttää järjestystä width, height, channels.
        return np.transpose(
            rgb,
            (1, 0, 2),
        )


def main() -> None:
    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H),
        pygame.RESIZABLE,
    )

    clock = pygame.time.Clock()
    simulation = SmokeSimulation()

    paused = False
    running = True

    while running:
        # Estää suuren timestepin ikkunan siirtelyn / breakpointin jälkeen.
        dt = min(
            clock.tick(FPS) / 1000.0,
            1.0 / 30.0,
        )

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
            if pygame.mouse.get_pressed(
                num_buttons=3
            )[0]:
                simulation.add_mouse_smoke(
                    pygame.mouse.get_pos(),
                    screen.get_size(),
                    dt,
                )

            simulation.step(dt)

        smoke_rgb = simulation.render_array()

        smoke_surface = pygame.surfarray.make_surface(
            smoke_rgb
        )

        scaled = pygame.transform.smoothscale(
            smoke_surface,
            screen.get_size(),
        )

        screen.blit(
            scaled,
            (0, 0),
        )

        pygame.display.set_caption(
            "smoke.py v0.2"
            f"  |  {clock.get_fps():5.1f} FPS"
            f"  |  div RMS {simulation.divergence_rms:.4f}"
            + ("  |  PAUSED" if paused else "")
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
