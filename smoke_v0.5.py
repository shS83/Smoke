#!/usr/bin/env python3
"""
smoke.py v0.5

Pohjana v0.3:n realistisemmaksi koettu flat-renderöinti.

Uutta v0.5:ssa:
- erillinen temperature-kenttä
- temperature advektoituu velocity-kentän mukana
- kuuma kaasu nousee voimakkaammin
- lämpötila jäähtyy ajan myötä
- kuuma lähde syöttää savun lisäksi lämpöä
- lämpötila vaikuttaa buoyancyyn
- T-näppäimellä temperature-debugnäkymä
- flat smoke rendering on oletus

Mukana aiemmista:
- density
- velocity
- semi-Lagrangian advektio
- pressure projection
- divergence-free velocity
- curl
- vorticity confinement
- avoin yläreuna
- hiirellä lisättävä savu

Ohjaus:
    ESC          lopeta
    SPACE        pause
    R            reset
    T            temperature debug päälle/pois
    Vasen hiiri  lisää savua + lämpöä
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

# ---------------------------------------------------------------------------
# Savun käyttäytyminen
# ---------------------------------------------------------------------------

VELOCITY_DAMPING = 0.992
DENSITY_DISSIPATION = 0.994

# V0.5: lämpötila
AMBIENT_TEMPERATURE = 0.0

TEMPERATURE_DISSIPATION = 0.991

# Kuinka voimakkaasti lämpötila nostaa kaasua.
THERMAL_BUOYANCY = 31.0

# Density tekee savusta hieman raskaampaa.
SMOKE_WEIGHT = 5.0

# ---------------------------------------------------------------------------
# Pressure projection
# ---------------------------------------------------------------------------

PRESSURE_ITERATIONS = 30

# ---------------------------------------------------------------------------
# Pyörteisyys
# ---------------------------------------------------------------------------

VORTICITY_STRENGTH = 22.0
VORTICITY_EPSILON = 1.0e-5

# ---------------------------------------------------------------------------
# Savulähde
# ---------------------------------------------------------------------------

SOURCE_X = SIM_W * 0.5
SOURCE_Y = SIM_H - 7.0

SOURCE_RADIUS_X = 9.0
SOURCE_RADIUS_Y = 3.5

SOURCE_STRENGTH = 2.8
SOURCE_UPWARD_SPEED = 12.0

# V0.5
SOURCE_TEMPERATURE = 3.0
SOURCE_HEAT_STRENGTH = 7.0

# ---------------------------------------------------------------------------
# Renderöinti — flat smoke
# ---------------------------------------------------------------------------

BACKGROUND = np.array(
    [5.0, 6.0, 8.0],
    dtype=np.float32,
)

SMOKE_COLOR = np.array(
    [195.0, 198.0, 202.0],
    dtype=np.float32,
)

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
    """Näytteistää 2D-kentän bilineaarisesti."""

    x = np.clip(
        x,
        0.0,
        SIM_W - 1.001,
    )

    y = np.clip(
        y,
        0.0,
        SIM_H - 1.001,
    )

    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)

    x1 = np.minimum(
        x0 + 1,
        SIM_W - 1,
    )

    y1 = np.minimum(
        y0 + 1,
        SIM_H - 1,
    )

    sx = x - x0
    sy = y - y0

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    top = (
        f00 * (1.0 - sx)
        + f10 * sx
    )

    bottom = (
        f01 * (1.0 - sx)
        + f11 * sx
    )

    return (
        top * (1.0 - sy)
        + bottom * sy
    )


def advect(
    field: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Semi-Lagrangian advektio."""

    previous_x = (
        GRID_X
        - vx * dt
    )

    previous_y = (
        GRID_Y
        - vy * dt
    )

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

    dx = (
        GRID_X - cx
    ) / rx

    dy = (
        GRID_Y - cy
    ) / ry

    return np.exp(
        -(dx * dx + dy * dy)
        * 2.0
    ).astype(np.float32)


SOURCE_MASK = gaussian_blob(
    SOURCE_X,
    SOURCE_Y,
    SOURCE_RADIUS_X,
    SOURCE_RADIUS_Y,
)


class SmokeSimulation:
    def __init__(self) -> None:
        shape = (
            SIM_H,
            SIM_W,
        )

        self.density = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.temperature = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.vx = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.vy = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.divergence = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.pressure = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.pressure_tmp = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.curl = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.time = 0.0

        self.divergence_rms = 0.0
        self.curl_rms = 0.0

        self.show_temperature = False

    def reset(self) -> None:
        self.density.fill(0.0)
        self.temperature.fill(
            AMBIENT_TEMPERATURE
        )

        self.vx.fill(0.0)
        self.vy.fill(0.0)

        self.divergence.fill(0.0)
        self.pressure.fill(0.0)
        self.pressure_tmp.fill(0.0)

        self.curl.fill(0.0)

        self.time = 0.0

        self.divergence_rms = 0.0
        self.curl_rms = 0.0

    # -------------------------------------------------------------------
    # Lähteet
    # -------------------------------------------------------------------

    def add_source(
        self,
        dt: float,
    ) -> None:
        """
        Lisää savua ja lämpöä.

        V0.5:ssa suoraa ylöspäin työntävää nopeutta on vähennetty:
        suurin osa nosteesta tulee nyt temperature-kentästä.
        """

        pulse = (
            0.88
            + 0.12
            * math.sin(
                self.time * 2.4
            )
        )

        source = (
            SOURCE_MASK
            * pulse
        )

        # Savu
        self.density += (
            source
            * SOURCE_STRENGTH
            * dt
            * 10.0
        )

        # Lämpö
        self.temperature += (
            source
            * SOURCE_HEAT_STRENGTH
            * dt
        )

        np.clip(
            self.temperature,
            AMBIENT_TEMPERATURE,
            SOURCE_TEMPERATURE,
            out=self.temperature,
        )

        # Pieni alkupotku ylöspäin.
        self.vy -= (
            source
            * SOURCE_UPWARD_SPEED
            * dt
            * 3.0
        )

        # Pieni epäsymmetria pyörteiden siemeneksi.
        sway = (
            math.sin(
                self.time * 0.83
            ) * 2.0
            + math.sin(
                self.time * 0.31
                + 1.7
            ) * 1.0
        )

        self.vx += (
            source
            * sway
            * dt
            * 3.0
        )

    def add_mouse_smoke(
        self,
        mouse_pos: tuple[int, int],
        window_size: tuple[int, int],
        dt: float,
    ) -> None:
        """Hiirellä lisätään sekä savua että lämpöä."""

        mx, my = mouse_pos
        ww, wh = window_size

        if ww <= 0 or wh <= 0:
            return

        sx = (
            mx
            / ww
            * SIM_W
        )

        sy = (
            my
            / wh
            * SIM_H
        )

        blob = gaussian_blob(
            sx,
            sy,
            5.0,
            5.0,
        )

        self.density += (
            blob
            * 5.0
            * dt
        )

        self.temperature += (
            blob
            * 3.8
            * dt
        )

        self.vy -= (
            blob
            * 12.0
            * dt
        )

        self.vx += (
            blob
            * math.sin(
                self.time * 3.1
            )
            * 6.0
            * dt
        )

    # -------------------------------------------------------------------
    # Rajat
    # -------------------------------------------------------------------

    def apply_velocity_boundaries(
        self,
    ) -> None:
        self.vx[:, 0] = 0.0
        self.vx[:, -1] = 0.0

        self.vy[:, 0] = (
            self.vy[:, 1]
        )

        self.vy[:, -1] = (
            self.vy[:, -2]
        )

        self.vy[-1, :] = 0.0

        self.vx[-1, :] = (
            self.vx[-2, :]
        )

        self.vx[0, :] = (
            self.vx[1, :]
        )

        # Negatiivinen vy = ylöspäin.
        self.vy[0, :] = np.minimum(
            self.vy[1, :],
            0.0,
        )

    def apply_pressure_boundaries(
        self,
        pressure: np.ndarray,
    ) -> None:
        pressure[:, 0] = (
            pressure[:, 1]
        )

        pressure[:, -1] = (
            pressure[:, -2]
        )

        pressure[0, :] = (
            pressure[1, :]
        )

        pressure[-1, :] = (
            pressure[-2, :]
        )

    # -------------------------------------------------------------------
    # Pressure projection
    # -------------------------------------------------------------------

    def compute_divergence(
        self,
    ) -> None:
        self.divergence.fill(
            0.0
        )

        self.divergence[
            1:-1,
            1:-1,
        ] = 0.5 * (
            self.vx[
                1:-1,
                2:,
            ]
            - self.vx[
                1:-1,
                :-2,
            ]
            + self.vy[
                2:,
                1:-1,
            ]
            - self.vy[
                :-2,
                1:-1,
            ]
        )

    def solve_pressure(
        self,
    ) -> None:
        self.pressure.fill(
            0.0
        )

        self.pressure_tmp.fill(
            0.0
        )

        p = self.pressure
        p_new = self.pressure_tmp
        div = self.divergence

        for _ in range(
            PRESSURE_ITERATIONS
        ):
            p_new[
                1:-1,
                1:-1,
            ] = 0.25 * (
                p[
                    1:-1,
                    :-2,
                ]
                + p[
                    1:-1,
                    2:,
                ]
                + p[
                    :-2,
                    1:-1,
                ]
                + p[
                    2:,
                    1:-1,
                ]
                - div[
                    1:-1,
                    1:-1,
                ]
            )

            self.apply_pressure_boundaries(
                p_new
            )

            p, p_new = (
                p_new,
                p,
            )

        if p is not self.pressure:
            self.pressure[:, :] = p

    def subtract_pressure_gradient(
        self,
    ) -> None:
        p = self.pressure

        self.vx[
            1:-1,
            1:-1,
        ] -= 0.5 * (
            p[
                1:-1,
                2:,
            ]
            - p[
                1:-1,
                :-2,
            ]
        )

        self.vy[
            1:-1,
            1:-1,
        ] -= 0.5 * (
            p[
                2:,
                1:-1,
            ]
            - p[
                :-2,
                1:-1,
            ]
        )

    def project_velocity(
        self,
    ) -> None:
        self.apply_velocity_boundaries()

        self.compute_divergence()
        self.solve_pressure()
        self.subtract_pressure_gradient()

        self.apply_velocity_boundaries()

        self.compute_divergence()

        interior = (
            self.divergence[
                1:-1,
                1:-1,
            ]
        )

        self.divergence_rms = float(
            np.sqrt(
                np.mean(
                    interior
                    * interior
                )
            )
        )

    # -------------------------------------------------------------------
    # Curl + vorticity confinement
    # -------------------------------------------------------------------

    def compute_curl(
        self,
    ) -> None:
        self.curl.fill(
            0.0
        )

        self.curl[
            1:-1,
            1:-1,
        ] = 0.5 * (
            self.vy[
                1:-1,
                2:,
            ]
            - self.vy[
                1:-1,
                :-2,
            ]
            - self.vx[
                2:,
                1:-1,
            ]
            + self.vx[
                :-2,
                1:-1,
            ]
        )

        interior = (
            self.curl[
                1:-1,
                1:-1,
            ]
        )

        self.curl_rms = float(
            np.sqrt(
                np.mean(
                    interior
                    * interior
                )
            )
        )

    def apply_vorticity_confinement(
        self,
        dt: float,
    ) -> None:
        self.compute_curl()

        abs_curl = np.abs(
            self.curl
        )

        grad_x = np.zeros_like(
            self.curl
        )

        grad_y = np.zeros_like(
            self.curl
        )

        grad_x[
            1:-1,
            1:-1,
        ] = 0.5 * (
            abs_curl[
                1:-1,
                2:,
            ]
            - abs_curl[
                1:-1,
                :-2,
            ]
        )

        grad_y[
            1:-1,
            1:-1,
        ] = 0.5 * (
            abs_curl[
                2:,
                1:-1,
            ]
            - abs_curl[
                :-2,
                1:-1,
            ]
        )

        magnitude = np.sqrt(
            grad_x * grad_x
            + grad_y * grad_y
        )

        inv_magnitude = (
            1.0
            / (
                magnitude
                + VORTICITY_EPSILON
            )
        )

        nx = (
            grad_x
            * inv_magnitude
        )

        ny = (
            grad_y
            * inv_magnitude
        )

        omega = self.curl

        force_x = np.clip(
            ny
            * omega
            * VORTICITY_STRENGTH,
            -80.0,
            80.0,
        )

        force_y = np.clip(
            -nx
            * omega
            * VORTICITY_STRENGTH,
            -80.0,
            80.0,
        )

        smoke_weight = np.clip(
            0.15
            + self.density
            * 0.85,
            0.15,
            1.0,
        )

        self.vx += (
            force_x
            * smoke_weight
            * dt
        )

        self.vy += (
            force_y
            * smoke_weight
            * dt
        )

    # -------------------------------------------------------------------
    # V0.5: thermal buoyancy
    # -------------------------------------------------------------------

    def apply_thermal_buoyancy(
        self,
        dt: float,
    ) -> None:
        """
        Kuuma kaasu nousee.

        Negatiivinen Y = ylöspäin.

        Voima muodostuu:
            lämpötila -> ylöspäin
            density   -> hieman alaspäin/raskaammaksi

        Näin savun nousu ei enää perustu pelkästään suoraan
        density-arvoon kuten vanhemmissa versioissa.
        """

        temperature_delta = (
            self.temperature
            - AMBIENT_TEMPERATURE
        )

        buoyancy_force = (
            temperature_delta
            * THERMAL_BUOYANCY
            - self.density
            * SMOKE_WEIGHT
        )

        self.vy -= (
            buoyancy_force
            * dt
        )

    # -------------------------------------------------------------------
    # Simulaatio
    # -------------------------------------------------------------------

    def step(
        self,
        dt: float,
    ) -> None:
        self.time += dt

        # 1. Lähde:
        # density + temperature
        self.add_source(
            dt
        )

        # 2. V0.5:
        # lämpötila määrää suuren osan nosteesta.
        self.apply_thermal_buoyancy(
            dt
        )

        # 3. Pieni symmetrianrikkoja.
        tiny_wave = np.sin(
            GRID_Y * 0.073
            + self.time * 0.61
        )

        self.vx += (
            tiny_wave.astype(
                np.float32
            )
            * self.density
            * dt
            * 0.35
        )

        # 4. Advect velocity.
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

        # 5. Pyörteisyys.
        self.apply_vorticity_confinement(
            dt
        )

        # 6. Pressure projection.
        self.project_velocity()

        # 7. Advect density.
        self.density = advect(
            self.density,
            self.vx,
            self.vy,
            dt,
        )

        # 8. V0.5:
        # Advect temperature samalla virtauksella.
        self.temperature = advect(
            self.temperature,
            self.vx,
            self.vy,
            dt,
        )

        # 9. Häviöt.
        self.vx *= (
            VELOCITY_DAMPING
        )

        self.vy *= (
            VELOCITY_DAMPING
        )

        self.density *= (
            DENSITY_DISSIPATION
        )

        self.temperature *= (
            TEMPERATURE_DISSIPATION
        )

        # Lämpötila palautuu hiljalleen ympäristön tasolle.
        self.temperature += (
            AMBIENT_TEMPERATURE
            * (
                1.0
                - TEMPERATURE_DISSIPATION
            )
        )

        np.clip(
            self.density,
            0.0,
            4.0,
            out=self.density,
        )

        np.clip(
            self.temperature,
            AMBIENT_TEMPERATURE,
            SOURCE_TEMPERATURE,
            out=self.temperature,
        )

        # Pohjalle ei saa syntyä ikuista savumattoa.
        self.density[
            -1,
            :,
        ] *= 0.80

        # Lämpökin häviää nopeasti aivan pohjareunasta.
        self.temperature[
            -1,
            :,
        ] *= 0.82

    # -------------------------------------------------------------------
    # Renderöinti
    # -------------------------------------------------------------------

    def render_smoke(
        self,
    ) -> np.ndarray:
        """
        Flat rendering, kuten v0.3:ssa.

        Tämä valittiin v0.4:n volumetrista renderöintiä
        realistisemman näköiseksi savulle.
        """

        alpha = (
            1.0
            - np.exp(
                -self.density
                * ABSORPTION
            )
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0,
        )

        brightness = np.clip(
            0.62
            + self.density
            * 0.12,
            0.62,
            1.0,
        )

        smoke = (
            SMOKE_COLOR[
                None,
                None,
                :,
            ]
            * brightness[
                :,
                :,
                None,
            ]
        )

        rgb = (
            BACKGROUND[
                None,
                None,
                :,
            ]
            * (
                1.0
                - alpha[
                    :,
                    :,
                    None,
                ]
            )
            + smoke
            * alpha[
                :,
                :,
                None,
            ]
        )

        return np.clip(
            rgb,
            0,
            255,
        ).astype(
            np.uint8
        )

    def render_temperature(
        self,
    ) -> np.ndarray:
        """
        Debug-näkymä temperature-kentästä.

        Musta -> viileä
        punainen -> lämmin
        keltainen -> kuuma
        valkoinen -> erittäin kuuma
        """

        t = np.clip(
            self.temperature
            / SOURCE_TEMPERATURE,
            0.0,
            1.0,
        )

        r = np.clip(
            t * 3.0,
            0.0,
            1.0,
        )

        g = np.clip(
            (t - 0.25)
            * 2.2,
            0.0,
            1.0,
        )

        b = np.clip(
            (t - 0.72)
            * 3.5,
            0.0,
            1.0,
        )

        rgb = np.stack(
            (
                r,
                g,
                b,
            ),
            axis=2,
        )

        rgb = (
            rgb
            * 255.0
        )

        return np.clip(
            rgb,
            0,
            255,
        ).astype(
            np.uint8
        )

    def render_array(
        self,
    ) -> np.ndarray:
        if self.show_temperature:
            rgb = (
                self.render_temperature()
            )
        else:
            rgb = (
                self.render_smoke()
            )

        return np.transpose(
            rgb,
            (
                1,
                0,
                2,
            ),
        )


def main() -> None:
    pygame.init()

    screen = pygame.display.set_mode(
        (
            WINDOW_W,
            WINDOW_H,
        ),
        pygame.RESIZABLE,
    )

    clock = pygame.time.Clock()

    simulation = SmokeSimulation()

    paused = False
    running = True

    while running:
        dt = min(
            clock.tick(
                FPS
            )
            / 1000.0,
            1.0 / 30.0,
        )

        for event in pygame.event.get():
            if (
                event.type
                == pygame.QUIT
            ):
                running = False

            elif (
                event.type
                == pygame.KEYDOWN
            ):
                if (
                    event.key
                    == pygame.K_ESCAPE
                ):
                    running = False

                elif (
                    event.key
                    == pygame.K_SPACE
                ):
                    paused = (
                        not paused
                    )

                elif (
                    event.key
                    == pygame.K_r
                ):
                    simulation.reset()

                elif (
                    event.key
                    == pygame.K_t
                ):
                    simulation.show_temperature = (
                        not simulation.show_temperature
                    )

        if not paused:
            if pygame.mouse.get_pressed(
                num_buttons=3
            )[0]:
                simulation.add_mouse_smoke(
                    pygame.mouse.get_pos(),
                    screen.get_size(),
                    dt,
                )

            simulation.step(
                dt
            )

        smoke_rgb = (
            simulation.render_array()
        )

        smoke_surface = (
            pygame.surfarray.make_surface(
                smoke_rgb
            )
        )

        scaled = (
            pygame.transform.smoothscale(
                smoke_surface,
                screen.get_size(),
            )
        )

        screen.blit(
            scaled,
            (
                0,
                0,
            ),
        )

        mode = (
            "TEMP"
            if simulation.show_temperature
            else "SMOKE"
        )

        max_temp = float(
            np.max(
                simulation.temperature
            )
        )

        pygame.display.set_caption(
            "smoke.py v0.5"
            f"  |  {clock.get_fps():5.1f} FPS"
            f"  |  div {simulation.divergence_rms:.4f}"
            f"  |  curl {simulation.curl_rms:.4f}"
            f"  |  temp {max_temp:.2f}"
            f"  |  {mode}"
            + (
                "  |  PAUSED"
                if paused
                else ""
            )
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
