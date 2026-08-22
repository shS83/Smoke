#!/usr/bin/env python3
"""
fire_smoke.py v0.8.1 GPU

Ensimmäinen varsinainen fire + smoke -versio.

Uutta:
- fuel-kenttä
- palaminen kuluttaa fuelia
- palaminen tuottaa lämpöä
- palaminen tuottaa savua
- liekin ulkoasu syntyy temperature + fuel + burn-kentistä
- savu käyttää edelleen v0.5:n flat-renderöintiä
- liekki kompositoidaan savun päälle additiivisesti / emissiivisesti

Ohjaus:
    ESC          lopeta
    SPACE        pause
    R            reset
    F            fire päälle/pois
    T            temperature debug
    G            fuel debug
    Vasen hiiri  lisää polttoainetta + lämpöä + hieman savua
"""

import math
import sys

import numpy as np
import pygame

try:
    import moderngl
except ImportError as exc:
    raise SystemExit(
        "fire_smoke v0.8 requires ModernGL: pip install moderngl"
    ) from exc


# ---------------------------------------------------------------------------
# Ikkuna / simulaatio
# ---------------------------------------------------------------------------

WINDOW_W = 1280
WINDOW_H = 720

SIM_W = 240
SIM_H = 135

FPS = 60


# ---------------------------------------------------------------------------
# Fluid
# ---------------------------------------------------------------------------

VELOCITY_DAMPING = 0.992
DENSITY_DISSIPATION = 0.994

PRESSURE_ITERATIONS = 30

VORTICITY_STRENGTH = 22.0
VORTICITY_EPSILON = 1.0e-5


# ---------------------------------------------------------------------------
# Lämpö / buoyancy
# ---------------------------------------------------------------------------

AMBIENT_TEMPERATURE = 0.0
TEMPERATURE_DISSIPATION = 0.990

THERMAL_BUOYANCY = 30.0
SMOKE_WEIGHT = 4.5


# ---------------------------------------------------------------------------
# Polttoaine / palaminen
# ---------------------------------------------------------------------------

FUEL_DISSIPATION = 0.996

# Fuelia kuluu sekunnissa lämpötilasta riippuen.
BURN_RATE = 1.85

# Palaminen alkaa kunnolla tämän lämpötilan yläpuolella.
IGNITION_TEMPERATURE = 0.32

# Palamisesta syntyvä lämpö.
HEAT_RELEASE = 2.45

# Palamisesta syntyvä smoke density.
SMOKE_YIELD = 0.38

# Polttoaine ei saa kasvaa loputtomasti.
MAX_FUEL = 2.0

# Lämpötilan maksimi.
MAX_TEMPERATURE = 3.2


# ---------------------------------------------------------------------------
# Lähde
# ---------------------------------------------------------------------------

SOURCE_X = SIM_W * 0.5
SOURCE_Y = SIM_H - 7.0

SOURCE_RADIUS_X = 8.0
SOURCE_RADIUS_Y = 3.2

SOURCE_FUEL_STRENGTH = 3.8
SOURCE_HEAT_STRENGTH = 4.5
SOURCE_SMOKE_STRENGTH = 0.32

SOURCE_UPWARD_SPEED = 10.0


# ---------------------------------------------------------------------------
# Smoke rendering
# ---------------------------------------------------------------------------

BACKGROUND = np.array(
    [4.0, 5.0, 7.0],
    dtype=np.float32,
)

SMOKE_COLOR = np.array(
    [195.0, 198.0, 202.0],
    dtype=np.float32,
)

SMOKE_ABSORPTION = 1.65


# ---------------------------------------------------------------------------
# Fire rendering
# ---------------------------------------------------------------------------

# Liekki näkyy vain tarpeeksi lämpimänä.
FIRE_TEMP_MIN = 0.18

# Liekin alpha / emissio.
FIRE_ALPHA_GAIN = 2.05
FIRE_GLOW_GAIN = 0.72

# V0.7: kasvatetaan liekin näkyvää aluetta renderöinnissä ilman,
# että savun tai fluidin fysiikkaa muutetaan.
FLAME_DILATION_STEPS = 2
FLAME_DILATION_MIX = 0.72



# ---------------------------------------------------------------------------
# Grid
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


class FireSmokeSimulation:
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

        self.fuel = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.burn = np.zeros(
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

        # v0.8: reusable work buffers to reduce per-frame allocations.
        self.grad_x = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.grad_y = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.old_vx = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.old_vy = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.time = 0.0

        self.divergence_rms = 0.0
        self.curl_rms = 0.0

        self.show_fire = True
        self.debug_mode = "normal"

    def reset(self) -> None:
        self.density.fill(0.0)
        self.temperature.fill(0.0)
        self.fuel.fill(0.0)
        self.burn.fill(0.0)

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
        pulse = (
            0.90
            + 0.10
            * math.sin(
                self.time * 3.2
            )
        )

        source = (
            SOURCE_MASK
            * pulse
        )

        # Fuel
        self.fuel += (
            source
            * SOURCE_FUEL_STRENGTH
            * dt
        )

        # Sytytyslämpö
        self.temperature += (
            source
            * SOURCE_HEAT_STRENGTH
            * dt
        )

        # Lähteestä tulee vähän valmista savua.
        self.density += (
            source
            * SOURCE_SMOKE_STRENGTH
            * dt
        )

        # Pieni alkupotku.
        self.vy -= (
            source
            * SOURCE_UPWARD_SPEED
            * dt
        )

        # Epäsymmetria, jotta liekki ei pysy täydellisenä pylväänä.
        sway = (
            math.sin(
                self.time * 0.91
            ) * 2.2
            + math.sin(
                self.time * 0.37
                + 1.3
            ) * 1.0
        )

        self.vx += (
            source
            * sway
            * dt
            * 2.8
        )

        np.clip(
            self.fuel,
            0.0,
            MAX_FUEL,
            out=self.fuel,
        )

        np.clip(
            self.temperature,
            0.0,
            MAX_TEMPERATURE,
            out=self.temperature,
        )

    def add_mouse_source(
        self,
        mouse_pos: tuple[int, int],
        window_size: tuple[int, int],
        dt: float,
    ) -> None:
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

        self.fuel += (
            blob
            * 2.2
            * dt
        )

        self.temperature += (
            blob
            * 3.0
            * dt
        )

        self.density += (
            blob
            * 0.35
            * dt
        )

        self.vy -= (
            blob
            * 10.0
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
    # Palaminen
    # -------------------------------------------------------------------

    def combust(
        self,
        dt: float,
    ) -> None:
        """
        Yksinkertainen combustion-malli:

        - fuel palaa vain kun lämpötila ylittää ignition-rajan
        - fuel vähenee
        - temperature kasvaa
        - density kasvaa => syntyy savua
        """

        ignition = np.clip(
            (
                self.temperature
                - IGNITION_TEMPERATURE
            )
            / (
                1.0
                - IGNITION_TEMPERATURE
            ),
            0.0,
            1.0,
        )

        possible_burn = (
            self.fuel
            * ignition
            * BURN_RATE
            * dt
        )

        burned = np.minimum(
            self.fuel,
            possible_burn,
        )

        self.fuel -= burned

        self.temperature += (
            burned
            * HEAT_RELEASE
        )

        self.density += (
            burned
            * SMOKE_YIELD
        )

        # Burn-field renderöintiä varten:
        # ei pelkkä fuel tai temperature, vaan juuri tämän framen palaminen.
        if dt > 0.0:
            self.burn = (
                burned
                / dt
            ).astype(np.float32)
        else:
            self.burn.fill(0.0)

        np.clip(
            self.temperature,
            0.0,
            MAX_TEMPERATURE,
            out=self.temperature,
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
        self.pressure.fill(0.0)
        self.pressure_tmp.fill(0.0)

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
    # Curl / vorticity
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

        grad_x = self.grad_x
        grad_y = self.grad_y

        grad_x.fill(0.0)
        grad_y.fill(0.0)

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

        activity = np.clip(
            0.12
            + self.density * 0.55
            + self.fuel * 0.33,
            0.12,
            1.0,
        )

        self.vx += (
            force_x
            * activity
            * dt
        )

        self.vy += (
            force_y
            * activity
            * dt
        )

    # -------------------------------------------------------------------
    # Buoyancy
    # -------------------------------------------------------------------

    def apply_thermal_buoyancy(
        self,
        dt: float,
    ) -> None:
        buoyancy_force = (
            self.temperature
            * THERMAL_BUOYANCY
            - self.density
            * SMOKE_WEIGHT
        )

        self.vy -= (
            buoyancy_force
            * dt
        )

    # -------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------

    def step(
        self,
        dt: float,
    ) -> None:
        self.time += dt

        # 1. Lisää polttoainetta ja sytytyslämpöä.
        self.add_source(
            dt
        )

        # 2. Palaminen ennen buoyancya.
        self.combust(
            dt
        )

        # 3. Kuuma kaasu nousee.
        self.apply_thermal_buoyancy(
            dt
        )

        # 4. Hyvin pieni symmetrian rikkoja.
        tiny_wave = np.sin(
            GRID_Y * 0.073
            + self.time * 0.61
        )

        self.vx += (
            tiny_wave.astype(
                np.float32
            )
            * (
                self.density
                + self.fuel * 0.45
            )
            * dt
            * 0.30
        )

        # 5. Advect velocity.
        np.copyto(
            self.old_vx,
            self.vx,
        )

        np.copyto(
            self.old_vy,
            self.vy,
        )

        self.vx = advect(
            self.old_vx,
            self.old_vx,
            self.old_vy,
            dt,
        )

        self.vy = advect(
            self.old_vy,
            self.old_vx,
            self.old_vy,
            dt,
        )

        # 6. Pyörteet.
        self.apply_vorticity_confinement(
            dt
        )

        # 7. Pressure projection.
        self.project_velocity()

        # 8. Advect scalar-kentät samalla velocityllä.
        self.density = advect(
            self.density,
            self.vx,
            self.vy,
            dt,
        )

        self.temperature = advect(
            self.temperature,
            self.vx,
            self.vy,
            dt,
        )

        self.fuel = advect(
            self.fuel,
            self.vx,
            self.vy,
            dt,
        )

        # Burn visual kulkee mukana kevyesti, mutta häviää nopeasti.
        self.burn = advect(
            self.burn,
            self.vx,
            self.vy,
            dt,
        )

        self.burn *= 0.90

        # 9. Dissipation / cooling.
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

        self.fuel *= (
            FUEL_DISSIPATION
        )

        np.clip(
            self.density,
            0.0,
            4.0,
            out=self.density,
        )

        np.clip(
            self.temperature,
            0.0,
            MAX_TEMPERATURE,
            out=self.temperature,
        )

        np.clip(
            self.fuel,
            0.0,
            MAX_FUEL,
            out=self.fuel,
        )

        # Pohjareunan siivous.
        self.density[-1, :] *= 0.80
        self.temperature[-1, :] *= 0.84
        self.fuel[-1, :] *= 0.82

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def render_smoke_base(
        self,
    ) -> np.ndarray:
        """
        Sama flat smoke -periaate kuin v0.5:ssa.
        """

        alpha = (
            1.0
            - np.exp(
                -self.density
                * SMOKE_ABSORPTION
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

        return rgb

    def fire_palette(
        self,
        heat: np.ndarray,
    ) -> np.ndarray:
        """
        V0.7:n punaisempi liekkipaletti:

        tumma punainen -> kirkas punainen -> punaoranssi ->
        oranssi -> keltainen -> lämmin valkoinen.

        Punainen pysyy mukana selvästi pidempään kuin v0.6:ssa.
        """

        h = np.clip(
            heat,
            0.0,
            1.0,
        )

        # Punainen syttyy hyvin aikaisin ja pysyy vahvana koko alueella.
        r = np.clip(
            h * 4.6,
            0.0,
            1.0,
        )

        # Vihreä tulee mukaan myöhemmin, jotta liekki ei muutu
        # liian nopeasti keltaiseksi.
        g = np.clip(
            (h - 0.30)
            * 1.75,
            0.0,
            1.0,
        )

        # Sininen vain aivan kuumimpaan ytimeen.
        b = np.clip(
            (h - 0.79)
            * 3.8,
            0.0,
            1.0,
        )

        # Keskialueelle hieman puna-oranssia korostusta.
        red_mid = np.clip(
            1.0 - np.abs(h - 0.42) / 0.42,
            0.0,
            1.0,
        )

        r = np.clip(
            r + red_mid * 0.16,
            0.0,
            1.0,
        )

        # Vain aivan kuumin osa lähestyy lämmintä valkoista.
        white = np.clip(
            (h - 0.91)
            / 0.09,
            0.0,
            1.0,
        )

        r = np.clip(
            r + white * 0.04,
            0.0,
            1.0,
        )

        g = np.clip(
            g + white * 0.14,
            0.0,
            1.0,
        )

        b = np.clip(
            b + white * 0.34,
            0.0,
            1.0,
        )

        return np.stack(
            (
                r,
                g,
                b,
            ),
            axis=2,
        )

    def expand_flame_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Kevyt max-filter-tyyppinen laajennus NumPyllä.

        Tämä kasvattaa vain näkyvän liekin aluetta renderöinnissä.
        Fluid, fuel, temperature ja savu pysyvät täysin samoina.
        """

        expanded = mask.copy()

        for _ in range(FLAME_DILATION_STEPS):
            neighbors = (
                expanded,
                np.roll(expanded, 1, axis=0),
                np.roll(expanded, -1, axis=0),
                np.roll(expanded, 1, axis=1),
                np.roll(expanded, -1, axis=1),
                np.roll(np.roll(expanded, 1, axis=0), 1, axis=1),
                np.roll(np.roll(expanded, 1, axis=0), -1, axis=1),
                np.roll(np.roll(expanded, -1, axis=0), 1, axis=1),
                np.roll(np.roll(expanded, -1, axis=0), -1, axis=1),
            )

            expanded = np.maximum.reduce(neighbors)

            # Estä wrap-around reunoilta.
            expanded[0, :] = 0.0
            expanded[-1, :] = 0.0
            expanded[:, 0] = 0.0
            expanded[:, -1] = 0.0

        return expanded

    def render_fire_smoke(
        self,
    ) -> np.ndarray:
        rgb = self.render_smoke_base()

        if not self.show_fire:
            return np.clip(
                rgb,
                0,
                255,
            ).astype(np.uint8)

        # Normalisoitu lämpö.
        heat = np.clip(
            (
                self.temperature
                - FIRE_TEMP_MIN
            )
            / (
                MAX_TEMPERATURE
                - FIRE_TEMP_MIN
            ),
            0.0,
            1.0,
        )

        # Liekki vaatii sekä lämpöä että joko fuelia tai aktiivista palamista.
        fuel_term = np.clip(
            self.fuel
            / 0.9,
            0.0,
            1.0,
        )

        burn_term = np.clip(
            self.burn
            / 1.6,
            0.0,
            1.0,
        )

        flame_mass = np.clip(
            fuel_term * 0.72
            + burn_term * 0.68,
            0.0,
            1.0,
        )

        flame_base = np.clip(
            heat
            * flame_mass
            * FIRE_ALPHA_GAIN,
            0.0,
            1.0,
        )

        # V0.7: kasvatetaan näkyvää liekkialuetta vain renderöinnissä.
        flame_expanded = self.expand_flame_mask(
            flame_base
        )

        flame = np.clip(
            flame_base * (1.0 - FLAME_DILATION_MIX)
            + flame_expanded * FLAME_DILATION_MIX,
            0.0,
            1.0,
        )

        # Hieman terävämpi ydin säilyttää keskustan kirkkaana,
        # vaikka ulkoreunaa kasvatetaan.
        core = np.power(
            flame_base,
            0.70,
        )

        fire_rgb = (
            self.fire_palette(
                np.clip(
                    heat * 0.98
                    + burn_term * 0.14,
                    0.0,
                    1.0,
                )
            )
            * 255.0
        )

        # Alpha-composite.
        rgb = (
            rgb
            * (
                1.0
                - flame[
                    :,
                    :,
                    None,
                ]
                * 0.88
            )
            + fire_rgb
            * flame[
                :,
                :,
                None,
            ]
        )

        # Emissive glow: pieni lisävalo, ei erillistä blur-passia.
        glow = np.clip(
            (
                heat
                * 0.55
                + burn_term * 0.45
            )
            * FIRE_GLOW_GAIN,
            0.0,
            0.55,
        )

        glow_rgb = np.stack(
            (
                glow * 255.0,
                glow * 86.0,
                glow * 10.0,
            ),
            axis=2,
        )

        rgb += (
            glow_rgb
            * core[
                :,
                :,
                None,
            ]
        )

        return np.clip(
            rgb,
            0,
            255,
        ).astype(np.uint8)

    def render_temperature(
        self,
    ) -> np.ndarray:
        t = np.clip(
            self.temperature
            / MAX_TEMPERATURE,
            0.0,
            1.0,
        )

        return (
            self.fire_palette(t)
            * 255.0
        ).astype(np.uint8)

    def render_fuel(
        self,
    ) -> np.ndarray:
        f = np.clip(
            self.fuel
            / MAX_FUEL,
            0.0,
            1.0,
        )

        # sinivioletti debug
        rgb = np.stack(
            (
                f * 80.0,
                f * 120.0,
                f * 255.0,
            ),
            axis=2,
        )

        return np.clip(
            rgb,
            0,
            255,
        ).astype(np.uint8)

    def render_array(
        self,
    ) -> np.ndarray:
        if self.debug_mode == "temperature":
            rgb = self.render_temperature()

        elif self.debug_mode == "fuel":
            rgb = self.render_fuel()

        else:
            rgb = self.render_fire_smoke()

        return np.transpose(
            rgb,
            (
                1,
                0,
                2,
            ),
        )


# ---------------------------------------------------------------------------
# V0.8: GPU renderer
# ---------------------------------------------------------------------------

VERTEX_SHADER = """
#version 330

in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""


FRAGMENT_SHADER = """
#version 330

uniform sampler2D density_tex;
uniform sampler2D temperature_tex;
uniform sampler2D fuel_tex;
uniform sampler2D burn_tex;

uniform vec2 texel_size;

uniform int debug_mode;
uniform bool show_fire;
uniform bool volumetric_enabled;
uniform float volumetric_strength;

in vec2 v_uv;
out vec4 fragColor;


float sat(float x) {
    return clamp(x, 0.0, 1.0);
}


vec3 fire_palette(float h) {
    h = sat(h);

    // v0.7:n punaisempi paletti.
    float r = sat(h * 4.6);
    float g = sat((h - 0.30) * 1.75);
    float b = sat((h - 0.79) * 3.8);

    float red_mid = sat(
        1.0 - abs(h - 0.42) / 0.42
    );

    r = sat(r + red_mid * 0.16);

    float white = sat(
        (h - 0.91) / 0.09
    );

    r = sat(r + white * 0.04);
    g = sat(g + white * 0.14);
    b = sat(b + white * 0.34);

    return vec3(r, g, b);
}


float sample_density(vec2 uv) {
    return texture(density_tex, uv).r;
}


float flame_at(vec2 uv) {
    float temp = texture(
        temperature_tex,
        uv
    ).r;

    float fuel = texture(
        fuel_tex,
        uv
    ).r;

    float burn = texture(
        burn_tex,
        uv
    ).r;

    const float FIRE_TEMP_MIN = 0.18;
    const float MAX_TEMPERATURE = 3.2;
    const float FIRE_ALPHA_GAIN = 2.05;

    float heat = sat(
        (temp - FIRE_TEMP_MIN)
        / (
            MAX_TEMPERATURE
            - FIRE_TEMP_MIN
        )
    );

    float fuel_term = sat(
        fuel / 0.9
    );

    float burn_term = sat(
        burn / 1.6
    );

    float flame_mass = sat(
        fuel_term * 0.72
        + burn_term * 0.68
    );

    return sat(
        heat
        * flame_mass
        * FIRE_ALPHA_GAIN
    );
}


float expanded_flame(vec2 uv) {
    float m = flame_at(uv);

    vec2 t = texel_size;

    // GPU-versio v0.7:n kahden dilation-stepin max-filteristä.
    // Ensimmäinen rengas.
    m = max(m, flame_at(uv + vec2( t.x, 0.0)));
    m = max(m, flame_at(uv + vec2(-t.x, 0.0)));
    m = max(m, flame_at(uv + vec2(0.0,  t.y)));
    m = max(m, flame_at(uv + vec2(0.0, -t.y)));

    m = max(m, flame_at(uv + vec2( t.x,  t.y)));
    m = max(m, flame_at(uv + vec2(-t.x,  t.y)));
    m = max(m, flame_at(uv + vec2( t.x, -t.y)));
    m = max(m, flame_at(uv + vec2(-t.x, -t.y)));

    // Toinen rengas.
    vec2 t2 = t * 2.0;

    m = max(m, flame_at(uv + vec2( t2.x, 0.0)));
    m = max(m, flame_at(uv + vec2(-t2.x, 0.0)));
    m = max(m, flame_at(uv + vec2(0.0,  t2.y)));
    m = max(m, flame_at(uv + vec2(0.0, -t2.y)));

    m = max(m, flame_at(uv + vec2( t2.x,  t2.y)));
    m = max(m, flame_at(uv + vec2(-t2.x,  t2.y)));
    m = max(m, flame_at(uv + vec2( t2.x, -t2.y)));
    m = max(m, flame_at(uv + vec2(-t2.x, -t2.y)));

    return m;
}


vec3 flat_smoke(
    float d,
    out float alpha
) {
    const vec3 BG = vec3(
        4.0 / 255.0,
        5.0 / 255.0,
        7.0 / 255.0
    );

    const vec3 SMOKE = vec3(
        195.0 / 255.0,
        198.0 / 255.0,
        202.0 / 255.0
    );

    alpha = sat(
        1.0 - exp(
            -d * 1.65
        )
    );

    float brightness = clamp(
        0.62 + d * 0.12,
        0.62,
        1.0
    );

    vec3 smoke = (
        SMOKE * brightness
    );

    return mix(
        BG,
        smoke,
        alpha
    );
}


vec3 volumetric_smoke(
    float d,
    float alpha
) {
    const vec3 BG = vec3(
        4.0 / 255.0,
        5.0 / 255.0,
        7.0 / 255.0
    );

    const vec3 DARK = vec3(
        67.0 / 255.0,
        70.0 / 255.0,
        75.0 / 255.0
    );

    const vec3 MID = vec3(
        150.0 / 255.0,
        154.0 / 255.0,
        160.0 / 255.0
    );

    const vec3 LIGHT = vec3(
        225.0 / 255.0,
        226.0 / 255.0,
        228.0 / 255.0
    );

    vec2 t = texel_size;

    float d_l = sample_density(
        v_uv - vec2(t.x, 0.0)
    );

    float d_r = sample_density(
        v_uv + vec2(t.x, 0.0)
    );

    float d_u = sample_density(
        v_uv - vec2(0.0, t.y)
    );

    float d_d = sample_density(
        v_uv + vec2(0.0, t.y)
    );

    float gx = 0.5 * (
        d_r - d_l
    );

    float gy = 0.5 * (
        d_d - d_u
    );

    vec3 n = normalize(
        vec3(
            -gx * 3.2,
            -gy * 3.2,
            1.0
        )
    );

    vec3 light_dir = normalize(
        vec3(
            -0.72,
            -0.58,
            0.38
        )
    );

    float diffuse = sat(
        dot(n, light_dir)
        * 0.5
        + 0.5
    );

    float edge = sat(
        length(
            vec2(gx, gy)
        ) * 2.8
    );

    float thickness = sat(
        d / 2.2
    );

    float interior_light = (
        1.0
        - thickness * 0.52
    );

    // Halpa self-shadow kuten v0.4:ssa,
    // mutta kaikki tehdään nyt fragment shaderissa.
    float s1 = sample_density(
        v_uv
        + vec2(-2.0, -1.0)
        * t
    );

    float s2 = sample_density(
        v_uv
        + vec2(-4.0, -2.0)
        * t
    );

    float s3 = sample_density(
        v_uv
        + vec2(-7.0, -4.0)
        * t
    );

    float s4 = sample_density(
        v_uv
        + vec2(-10.0, -6.0)
        * t
    );

    float occlusion = (
        s1 * 0.40
        + s2 * 0.28
        + s3 * 0.20
        + s4 * 0.12
    );

    float shadow = clamp(
        exp(
            -occlusion * 0.48
        ),
        0.38,
        1.0
    );

    float light = (
        0.30
        + diffuse * 0.62
        + edge * 0.58
    );

    light *= interior_light;
    light *= shadow;

    light = clamp(
        light,
        0.18,
        1.25
    );

    float thin_mix = (
        1.0 - thickness
    );

    vec3 base = mix(
        DARK,
        MID,
        thin_mix
    );

    float highlight_mix = clamp(
        edge
        * diffuse
        * 0.72,
        0.0,
        0.78
    );

    vec3 smoke = mix(
        base,
        LIGHT,
        highlight_mix
    );

    smoke *= light;

    smoke.b += (
        thin_mix
        * (5.0 / 255.0)
    );

    return mix(
        BG,
        smoke,
        alpha
    );
}


void main() {
    float d = texture(
        density_tex,
        v_uv
    ).r;

    float temp = texture(
        temperature_tex,
        v_uv
    ).r;

    float fuel = texture(
        fuel_tex,
        v_uv
    ).r;

    float burn = texture(
        burn_tex,
        v_uv
    ).r;

    // Debug: temperature
    if (debug_mode == 1) {
        float h = sat(
            temp / 3.2
        );

        fragColor = vec4(
            fire_palette(h),
            1.0
        );

        return;
    }

    // Debug: fuel
    if (debug_mode == 2) {
        float f = sat(
            fuel / 2.0
        );

        fragColor = vec4(
            f * vec3(
                80.0 / 255.0,
                120.0 / 255.0,
                1.0
            ),
            1.0
        );

        return;
    }

    float smoke_alpha;

    vec3 flat_color = flat_smoke(
        d,
        smoke_alpha
    );

    vec3 rgb = flat_color;

    // V0.4:n idea palautetaan miedompana:
    // flat on aina pohja, volumetric on vain lisäkerros.
    if (volumetric_enabled) {
        vec3 vol = volumetric_smoke(
            d,
            smoke_alpha
        );

        rgb = mix(
            flat_color,
            vol,
            volumetric_strength
        );
    }

    if (show_fire) {
        float heat = sat(
            (temp - 0.18)
            / (3.2 - 0.18)
        );

        float burn_term = sat(
            burn / 1.6
        );

        float flame_base = flame_at(
            v_uv
        );

        float flame_expanded = expanded_flame(
            v_uv
        );

        float flame = sat(
            flame_base * 0.28
            + flame_expanded * 0.72
        );

        float palette_heat = sat(
            heat * 0.98
            + burn_term * 0.14
        );

        vec3 fire_rgb = (
            fire_palette(
                palette_heat
            )
        );

        rgb = (
            rgb
            * (
                1.0
                - flame * 0.88
            )
            + fire_rgb
            * flame
        );

        // GPU glow: leveämpi kuin v0.7:ssa ilman np.roll/max-filteriä.
        vec2 t = texel_size;

        float glow_field = flame_expanded;

        glow_field = max(
            glow_field,
            flame_at(
                v_uv
                + vec2(3.0, 0.0) * t
            )
        );

        glow_field = max(
            glow_field,
            flame_at(
                v_uv
                + vec2(-3.0, 0.0) * t
            )
        );

        glow_field = max(
            glow_field,
            flame_at(
                v_uv
                + vec2(0.0, 3.0) * t
            )
        );

        glow_field = max(
            glow_field,
            flame_at(
                v_uv
                + vec2(0.0, -3.0) * t
            )
        );

        glow_field = max(
            glow_field,
            flame_at(
                v_uv
                + vec2(5.0, 2.0) * t
            )
        );

        glow_field = max(
            glow_field,
            flame_at(
                v_uv
                + vec2(-5.0, 2.0) * t
            )
        );

        float core = pow(
            max(flame_base, 0.0001),
            0.70
        );

        float glow = sat(
            (
                heat * 0.55
                + burn_term * 0.45
            ) * 0.72
        );

        glow *= (
            0.45
            + glow_field * 0.55
        );

        vec3 glow_rgb = vec3(
            glow,
            glow * (86.0 / 255.0),
            glow * (10.0 / 255.0)
        );

        // Lähialueen punertava emissio.
        rgb += (
            glow_rgb
            * (
                0.35
                + core * 0.75
            )
        );
    }

    fragColor = vec4(
        clamp(
            rgb,
            0.0,
            1.0
        ),
        1.0
    );
}
"""


class GPURenderer:
    """
    Uploads the four small scalar fields to GPU and performs:
    - smoke compositing
    - subtle volumetric shading
    - flame dilation
    - fire palette
    - emissive glow
    - final window scaling

    Fluid simulation itself is still CPU-side in v0.8.
    """

    def __init__(self) -> None:
        self.ctx = moderngl.create_context(
            require=330
        )

        self.ctx.disable(
            moderngl.DEPTH_TEST
        )

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        vertices = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 0.0,

                -1.0,  1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 1.0,
                 1.0,  1.0, 1.0, 0.0,
            ],
            dtype=np.float32,
        )

        self.vbo = self.ctx.buffer(
            vertices.tobytes()
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    "2f 2f",
                    "in_pos",
                    "in_uv",
                )
            ],
        )

        self.density_tex = self._make_texture()
        self.temperature_tex = self._make_texture()
        self.fuel_tex = self._make_texture()
        self.burn_tex = self._make_texture()

        self.program[
            "density_tex"
        ].value = 0

        self.program[
            "temperature_tex"
        ].value = 1

        self.program[
            "fuel_tex"
        ].value = 2

        self.program[
            "burn_tex"
        ].value = 3

        self.program[
            "texel_size"
        ].value = (
            1.0 / SIM_W,
            1.0 / SIM_H,
        )

        self.density_tex.use(
            location=0
        )

        self.temperature_tex.use(
            location=1
        )

        self.fuel_tex.use(
            location=2
        )

        self.burn_tex.use(
            location=3
        )

    def _make_texture(
        self,
    ):
        texture = self.ctx.texture(
            (
                SIM_W,
                SIM_H,
            ),
            components=1,
            dtype="f4",
        )

        texture.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR,
        )

        texture.repeat_x = False
        texture.repeat_y = False

        return texture

    def upload(
        self,
        simulation: FireSmokeSimulation,
    ) -> None:
        # Arrays are already float32 and normally C-contiguous.
        self.density_tex.write(
            np.ascontiguousarray(
                simulation.density
            )
        )

        self.temperature_tex.write(
            np.ascontiguousarray(
                simulation.temperature
            )
        )

        self.fuel_tex.write(
            np.ascontiguousarray(
                simulation.fuel
            )
        )

        self.burn_tex.write(
            np.ascontiguousarray(
                simulation.burn
            )
        )

    def render(
        self,
        simulation: FireSmokeSimulation,
        volumetric_enabled: bool,
        volumetric_strength: float,
    ) -> None:
        width, height = (
            pygame.display.get_window_size()
        )

        self.ctx.viewport = (
            0,
            0,
            max(width, 1),
            max(height, 1),
        )

        if simulation.debug_mode == "temperature":
            debug_mode = 1

        elif simulation.debug_mode == "fuel":
            debug_mode = 2

        else:
            debug_mode = 0

        self.program[
            "debug_mode"
        ].value = debug_mode

        self.program[
            "show_fire"
        ].value = simulation.show_fire

        self.program[
            "volumetric_enabled"
        ].value = volumetric_enabled

        self.program[
            "volumetric_strength"
        ].value = volumetric_strength

        self.ctx.clear(
            4.0 / 255.0,
            5.0 / 255.0,
            7.0 / 255.0,
            1.0,
        )

        self.vao.render(
            mode=moderngl.TRIANGLES
        )


def main() -> None:
    pygame.init()

    # Request an OpenGL 3.3 core context for ModernGL.
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_MAJOR_VERSION,
        3,
    )

    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_MINOR_VERSION,
        3,
    )

    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )

    pygame.display.gl_set_attribute(
        pygame.GL_DOUBLEBUFFER,
        1,
    )

    pygame.display.set_mode(
        (
            WINDOW_W,
            WINDOW_H,
        ),
        pygame.OPENGL
        | pygame.DOUBLEBUF
        | pygame.RESIZABLE,
    )

    clock = pygame.time.Clock()

    simulation = FireSmokeSimulation()
    renderer = GPURenderer()

    paused = False
    running = True

    # V0.4:n volumetric oli liian voimakas savulle.
    # Tässä se on tarkoituksella vain hienovarainen lisä flat-savuun.
    volumetric_enabled = True
    volumetric_strength = 0.38

    while running:
        dt = min(
            clock.tick(
                FPS
            )
            / 1000.0,
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

                elif event.key == pygame.K_f:
                    simulation.show_fire = (
                        not simulation.show_fire
                    )

                elif event.key == pygame.K_t:
                    if (
                        simulation.debug_mode
                        == "temperature"
                    ):
                        simulation.debug_mode = "normal"
                    else:
                        simulation.debug_mode = "temperature"

                elif event.key == pygame.K_g:
                    if (
                        simulation.debug_mode
                        == "fuel"
                    ):
                        simulation.debug_mode = "normal"
                    else:
                        simulation.debug_mode = "fuel"

                elif event.key == pygame.K_l:
                    volumetric_enabled = (
                        not volumetric_enabled
                    )

                elif event.key == pygame.K_LEFTBRACKET:
                    volumetric_strength = max(
                        0.0,
                        volumetric_strength - 0.05,
                    )

                elif event.key == pygame.K_RIGHTBRACKET:
                    volumetric_strength = min(
                        1.0,
                        volumetric_strength + 0.05,
                    )

        if not paused:
            if pygame.mouse.get_pressed(
                num_buttons=3
            )[0]:
                simulation.add_mouse_source(
                    pygame.mouse.get_pos(),
                    pygame.display.get_window_size(),
                    dt,
                )

            simulation.step(
                dt
            )

        # Only four 240x135 float textures are transferred each frame.
        renderer.upload(
            simulation
        )

        renderer.render(
            simulation,
            volumetric_enabled,
            volumetric_strength,
        )

        pygame.display.flip()

        max_temp = float(
            np.max(
                simulation.temperature
            )
        )

        max_fuel = float(
            np.max(
                simulation.fuel
            )
        )

        fire_name = (
            "FIRE"
            if simulation.show_fire
            else "SMOKE"
        )

        vol_name = (
            f"VOL {volumetric_strength:.2f}"
            if volumetric_enabled
            else "FLAT"
        )

        pygame.display.set_caption(
            "fire_smoke.py v0.8.1 GPU"
            f"  |  {clock.get_fps():5.1f} FPS"
            f"  |  div {simulation.divergence_rms:.4f}"
            f"  |  curl {simulation.curl_rms:.4f}"
            f"  |  temp {max_temp:.2f}"
            f"  |  fuel {max_fuel:.2f}"
            f"  |  {fire_name}"
            f"  |  {vol_name}"
            + (
                "  |  PAUSED"
                if paused
                else ""
            )
        )

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
