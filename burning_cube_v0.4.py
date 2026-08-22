#!/usr/bin/env python3
"""
burning_cube.py v0.4

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
from pathlib import Path

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
SMOKE_YIELD = 0.20

# Polttoaine ei saa kasvaa loputtomasti.
MAX_FUEL = 2.0

# Lämpötilan maksimi.
MAX_TEMPERATURE = 3.2


# ---------------------------------------------------------------------------
# Lähde
# ---------------------------------------------------------------------------

SOURCE_X = SIM_W * 0.5
SOURCE_Y = SIM_H * 0.84

SOURCE_RADIUS_X = 10.0
SOURCE_RADIUS_Y = 3.8

SOURCE_FUEL_STRENGTH = 4.9
SOURCE_HEAT_STRENGTH = 5.8
SOURCE_SMOKE_STRENGTH = 0.06

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
    # Keep interpolation math strictly float32.
    # Without this, subtracting int32 indices from float32 coordinates
    # promotes sx/sy to float64 in NumPy and doubles GPU upload size.
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    x = np.clip(
        x,
        np.float32(0.0),
        np.float32(SIM_W - 1.001),
    )

    y = np.clip(
        y,
        np.float32(0.0),
        np.float32(SIM_H - 1.001),
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

    sx = (
        x
        - x0.astype(
            np.float32
        )
    )

    sy = (
        y
        - y0.astype(
            np.float32
        )
    )

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    one = np.float32(1.0)

    top = (
        f00 * (one - sx)
        + f10 * sx
    )

    bottom = (
        f01 * (one - sx)
        + f11 * sx
    )

    result = (
        top * (
            np.float32(1.0)
            - sy
        )
        + bottom * sy
    )

    return result.astype(
        np.float32,
        copy=False,
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


# ---------------------------------------------------------------------------
# Burning Cube v0.4 — screen-space cube ignition masks
#
# The 3D cube occupies roughly x=0.30..0.70 and y=0.35..0.74 after
# projection. These soft masks make the 2D combustion field behave as if
# the cube itself starts emitting heat/fuel once the fire below reaches it.
# ---------------------------------------------------------------------------

CUBE_SIM_X = SIM_W * 0.50
CUBE_SIM_Y = SIM_H * 0.55

CUBE_CONTACT_MASK = gaussian_blob(
    CUBE_SIM_X,
    SIM_H * 0.73,
    14.0,
    5.0,
)

CUBE_LEFT_MASK = gaussian_blob(
    SIM_W * 0.36,
    CUBE_SIM_Y,
    4.6,
    23.0,
)

CUBE_RIGHT_MASK = gaussian_blob(
    SIM_W * 0.64,
    CUBE_SIM_Y,
    4.6,
    23.0,
)

CUBE_TOP_MASK = gaussian_blob(
    CUBE_SIM_X,
    SIM_H * 0.36,
    18.0,
    4.6,
)

CUBE_BOTTOM_MASK = gaussian_blob(
    CUBE_SIM_X,
    SIM_H * 0.73,
    18.0,
    4.8,
)

CUBE_CENTER_MASK = gaussian_blob(
    CUBE_SIM_X,
    CUBE_SIM_Y,
    19.0,
    20.0,
)

# Basic silhouette used for charring / surface activity.
CUBE_ENVELOPE_MASK = np.clip(
    CUBE_LEFT_MASK
    + CUBE_RIGHT_MASK
    + CUBE_TOP_MASK * 0.92
    + CUBE_BOTTOM_MASK * 0.84
    + CUBE_CENTER_MASK * 0.16,
    0.0,
    1.0,
).astype(np.float32)

# A more unified "flame sheet" source.
# Instead of several clearly separate sources, this behaves like one flame
# body that wraps around the cube after the lower part has ignited.
CUBE_FLAME_CORE_MASK = gaussian_blob(
    CUBE_SIM_X,
    CUBE_SIM_Y,
    16.0,
    24.0,
)

CUBE_FLAME_SHEET_MASK = np.clip(
    CUBE_ENVELOPE_MASK * 0.92
    + CUBE_FLAME_CORE_MASK * 0.34,
    0.0,
    1.0,
).astype(np.float32)

# Vertical weighting: more energetic/fuel-rich at the bottom, more
# smoke-prone at the top.
CUBE_TOPNESS = np.clip(
    (SIM_H * 0.76 - GRID_Y) / (SIM_H * 0.42),
    0.0,
    1.0,
).astype(np.float32)

CUBE_BOTTOMNESS = (
    1.0 - CUBE_TOPNESS
).astype(np.float32)

CUBE_FLAME_SOURCE_MASK = np.clip(
    CUBE_FLAME_SHEET_MASK
    * (
        0.58
        + CUBE_BOTTOMNESS * 0.52
    ),
    0.0,
    1.0,
).astype(np.float32)

CUBE_SMOKE_TRANSITION_MASK = np.clip(
    CUBE_TOP_MASK * 0.90
    + CUBE_TOPNESS * CUBE_FLAME_SHEET_MASK * 0.42
    + CUBE_CENTER_MASK * CUBE_TOPNESS * 0.12,
    0.0,
    1.0,
).astype(np.float32)

CUBE_TOP_SMOKE_MASK = gaussian_blob(
    CUBE_SIM_X,
    SIM_H * 0.31,
    16.0,
    5.4,
)

CUBE_TOP_LIFT_MASK = gaussian_blob(
    CUBE_SIM_X,
    SIM_H * 0.36,
    19.0,
    7.6,
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

        # Burning Cube v0.4:
        # 0 = cube not yet burning, 1 = fully engulfed.
        self.cube_ignition = 0.0

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
        self.cube_ignition = 0.0

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

    def update_cube_ignition(
        self,
        dt: float,
    ) -> None:
        """
        The flame below the cube must first heat the cube's lower region.

        Once contact temperature is high enough the fire spreads over the
        cube over a few seconds. After ignition has clearly started it
        continues spreading even if a turbulent frame briefly lowers the
        contact temperature.
        """

        weight_sum = float(
            np.sum(
                CUBE_CONTACT_MASK
            )
        )

        contact_heat = float(
            np.sum(
                self.temperature
                * CUBE_CONTACT_MASK
            )
            / max(
                weight_sum,
                1.0e-6,
            )
        )

        if contact_heat > 0.10:
            spread_speed = (
                0.52
                + min(
                    contact_heat,
                    1.6,
                ) * 0.22
            )

            self.cube_ignition = min(
                1.0,
                self.cube_ignition
                + spread_speed * dt,
            )

        elif self.cube_ignition > 0.12:
            # Once the surface is properly lit, fire keeps crawling upward.
            self.cube_ignition = min(
                1.0,
                self.cube_ignition
                + 0.12 * dt,
            )

    def add_cube_surface_source(
        self,
        dt: float,
    ) -> None:
        """
        v0.3: a more unified flame sheet around the cube.

        Instead of many visually separate local sources, the fire forms one
        coherent upward-wrapping body after the lower cube ignites. As the
        flame reaches the top, the upper region gradually transitions from
        fuel-rich flame into smoke.
        """

        ignition = self.cube_ignition

        if ignition <= 0.001:
            return

        flicker = (
            0.84
            + 0.09
            * math.sin(
                self.time * 4.2
            )
            + 0.07
            * math.sin(
                self.time * 7.6
                + 1.2
            )
        )

        flame_source = (
            CUBE_FLAME_SOURCE_MASK
            * ignition
            * flicker
        )

        # One surrounding flame source clinging to the cube and climbing up.
        self.fuel += (
            flame_source
            * 1.95
            * dt
        )

        self.temperature += (
            flame_source
            * 2.05
            * dt
        )

        self.vy -= (
            flame_source
            * (
                6.3
                + ignition * 1.7
            )
            * dt
        )

        # Gentle side roll to make the flame envelope move around the cube.
        side_wave = (
            math.sin(
                self.time * 2.8
            ) * 1.9
        )

        self.vx += (
            (
                CUBE_LEFT_MASK
                - CUBE_RIGHT_MASK
            )
            * side_wave
            * ignition
            * 0.90
            * dt
        )

        # Transition near the top:
        # existing hot fuel is gradually converted into smoke, so the
        # top feels like flame turning into smoke rather than a hard cutoff.
        top_transition = (
            CUBE_SMOKE_TRANSITION_MASK
            * ignition
            * (
                0.84
                + 0.16
                * math.sin(
                    self.time * 2.0
                    + 0.7
                )
            )
        )

        fuel_to_smoke = np.minimum(
            self.fuel,
            top_transition
            * 0.72
            * dt,
        )

        self.fuel -= fuel_to_smoke

        self.density += (
            fuel_to_smoke
            * 1.55
        )

        # Additional smoke concentrated just above the cube.
        top_smoke = (
            CUBE_TOP_SMOKE_MASK
            * ignition
            * (
                0.78
                + 0.22
                * math.sin(
                    self.time * 1.9
                    + 0.9
                )
            )
        )

        self.density += (
            top_smoke
            * 0.92
            * dt
        )

        # The upper plume keeps rising and cooling toward smoke.
        self.temperature += (
            CUBE_TOP_LIFT_MASK
            * ignition
            * 0.26
            * dt
        )

        self.vy -= (
            CUBE_TOP_LIFT_MASK
            * ignition
            * 7.2
            * dt
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

        # 1. Ignition flame below the cube.
        self.add_source(
            dt
        )

        # 2. Heat reaching the lower cube starts surface ignition.
        self.update_cube_ignition(
            dt
        )

        # 3. The burning cube becomes a distributed fuel/heat source.
        self.add_cube_surface_source(
            dt
        )

        # 4. Combust after all sources have been injected.
        self.combust(
            dt
        )

        # 5. Kuuma kaasu nousee.
        self.apply_thermal_buoyancy(
            dt
        )

        # 6. Hyvin pieni symmetrian rikkoja.
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

        # 7. Advect velocity.
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

        # 8. Pyörteet.
        self.apply_vorticity_confinement(
            dt
        )

        # 9. Pressure projection.
        self.project_velocity()

        # 10. Advect scalar-kentät samalla velocityllä.
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

        # 11. Dissipation / cooling.
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
# Burning Cube v0.4 — ModernGL renderer
# ---------------------------------------------------------------------------

CUBE_VERTEX_SHADER = """
#version 330

in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 mvp;
uniform mat4 model;

out vec3 v_normal;
out vec2 v_uv;

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_normal = normalize(mat3(model) * in_normal);
    v_uv = in_uv;
}
"""


CUBE_FRAGMENT_SHADER = """
#version 330

uniform sampler2D face_tex;
uniform sampler2D density_tex;
uniform sampler2D burn_tex;

in vec3 v_normal;
in vec2 v_uv;

out vec4 fragColor;


float sat(float x) {
    return clamp(x, 0.0, 1.0);
}


void main() {
    vec3 base = texture(
        face_tex,
        v_uv
    ).rgb;

    vec2 sim_uv = vec2(
        v_uv.x,
        1.0 - v_uv.y
    );

    float density = texture(
        density_tex,
        sim_uv
    ).r;

    float burn = texture(
        burn_tex,
        sim_uv
    ).r;

    // Cube surface may char and darken,
    // but it should not display a separate flame inside the texture.
    float char_amount = sat(
        density * 0.16
        + burn * 0.24
    );

    vec3 charred = mix(
        base,
        base * 0.28,
        char_amount
    );

    vec3 light_dir = normalize(
        vec3(-0.45, 0.70, 0.65)
    );

    float diffuse = (
        0.34
        + max(
            dot(
                normalize(v_normal),
                light_dir
            ),
            0.0
        ) * 0.66
    );

    vec3 color = charred * diffuse;

    fragColor = vec4(
        clamp(color, 0.0, 1.0),
        1.0
    );
}
"""


FLAME_SHELL_VERTEX_SHADER = """
#version 330

in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 mvp;
uniform mat4 model;

out vec3 v_normal;
out vec2 v_uv;

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_normal = normalize(mat3(model) * in_normal);
    v_uv = in_uv;
}
"""


FLAME_SHELL_FRAGMENT_SHADER = """
#version 330

uniform sampler2D temperature_tex;
uniform sampler2D fuel_tex;
uniform sampler2D burn_tex;

uniform float alpha_scale;

in vec3 v_normal;
in vec2 v_uv;

out vec4 fragColor;


float sat(float x) {
    return clamp(x, 0.0, 1.0);
}


vec3 fire_palette(float h) {
    h = sat(h);

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


void main() {
    vec2 sim_uv = vec2(
        v_uv.x,
        1.0 - v_uv.y
    );

    float temp = texture(
        temperature_tex,
        sim_uv
    ).r;

    float fuel = texture(
        fuel_tex,
        sim_uv
    ).r;

    float burn = texture(
        burn_tex,
        sim_uv
    ).r;

    float heat = sat(
        (temp - 0.18)
        / (3.2 - 0.18)
    );

    float fuel_term = sat(
        fuel / 0.9
    );

    float burn_term = sat(
        burn / 1.6
    );

    float flame = sat(
        heat
        * (
            fuel_term * 0.72
            + burn_term * 0.78
        )
        * 2.15
    );

    // Shape the flame to the cube:
    // stronger around the outer parts of each face,
    // weaker in the middle so it reads as an envelope rather than
    // as fire passing through the cube.
    float edge_dist = max(
        abs(v_uv.x - 0.5),
        abs(v_uv.y - 0.5)
    );

    float edge_ring = smoothstep(
        0.16,
        0.48,
        edge_dist
    );

    float center_soft = smoothstep(
        0.00,
        0.26,
        edge_dist
    );

    float top_fade = 1.0 - smoothstep(
        0.72,
        1.00,
        v_uv.y
    ) * 0.32;

    flame *= (
        0.25
        + edge_ring * 0.75
    );

    flame *= (
        0.35
        + center_soft * 0.65
    );

    flame *= top_fade;
    flame *= alpha_scale;

    if (flame < 0.01) {
        discard;
    }

    vec3 normal_vec = normalize(v_normal);
    vec3 viewish_light = normalize(
        vec3(-0.25, 0.85, 0.55)
    );

    float facing = (
        0.40
        + max(
            dot(normal_vec, viewish_light),
            0.0
        ) * 0.60
    );

    float palette_heat = sat(
        heat * 0.98
        + burn_term * 0.14
    );

    vec3 fire_rgb = fire_palette(
        palette_heat
    );

    // Warm glow around the shell.
    float glow = sat(
        (
            heat * 0.56
            + burn_term * 0.44
        ) * 0.78
    );

    vec3 glow_rgb = vec3(
        glow,
        glow * (86.0 / 255.0),
        glow * (10.0 / 255.0)
    );

    vec3 color = fire_rgb * facing;
    color += glow_rgb * 0.42;

    fragColor = vec4(
        clamp(color, 0.0, 1.0) * flame,
        flame
    );
}
"""


LINE_VERTEX_SHADER = """
#version 330

in vec3 in_pos;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""


LINE_FRAGMENT_SHADER = """
#version 330

out vec4 fragColor;

void main() {
    fragColor = vec4(
        0.92,
        0.93,
        0.96,
        0.88
    );
}
"""


OVERLAY_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(
        in_pos,
        0.0,
        1.0
    );

    v_uv = in_uv;
}
"""


OVERLAY_FRAGMENT_SHADER = """
#version 330

uniform sampler2D density_tex;
uniform sampler2D temperature_tex;
uniform sampler2D fuel_tex;
uniform sampler2D burn_tex;

uniform vec2 texel_size;

uniform int debug_mode;
uniform int pass_mode;
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
    return texture(
        density_tex,
        clamp(
            uv,
            vec2(0.0),
            vec2(1.0)
        )
    ).r;
}


float flame_at(vec2 uv) {
    uv = clamp(
        uv,
        vec2(0.0),
        vec2(1.0)
    );

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

    float heat = sat(
        (temp - 0.18)
        / (3.2 - 0.18)
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
        * 2.05
    );
}


float expanded_flame(vec2 uv) {
    vec2 t = texel_size;

    float m = flame_at(uv);

    m = max(m, flame_at(uv + vec2( t.x, 0.0)));
    m = max(m, flame_at(uv + vec2(-t.x, 0.0)));
    m = max(m, flame_at(uv + vec2(0.0,  t.y)));
    m = max(m, flame_at(uv + vec2(0.0, -t.y)));

    m = max(m, flame_at(uv + vec2( t.x,  t.y)));
    m = max(m, flame_at(uv + vec2(-t.x,  t.y)));
    m = max(m, flame_at(uv + vec2( t.x, -t.y)));
    m = max(m, flame_at(uv + vec2(-t.x, -t.y)));

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


vec3 flat_smoke_color(
    float d
) {
    vec3 smoke = vec3(
        195.0 / 255.0,
        198.0 / 255.0,
        202.0 / 255.0
    );

    float brightness = clamp(
        0.62 + d * 0.12,
        0.62,
        1.0
    );

    return smoke * brightness;
}


vec3 volumetric_smoke_color(
    float d
) {
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

    vec3 normal_vec = normalize(
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
        dot(
            normal_vec,
            light_dir
        ) * 0.5
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

    float lighting = (
        0.30
        + diffuse * 0.62
        + edge * 0.58
    );

    lighting *= interior_light;
    lighting *= shadow;

    lighting = clamp(
        lighting,
        0.18,
        1.25
    );

    float thin_mix = (
        1.0 - thickness
    );

    vec3 smoke = mix(
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

    smoke = mix(
        smoke,
        LIGHT,
        highlight_mix
    );

    smoke *= lighting;

    smoke.b += (
        thin_mix
        * (5.0 / 255.0)
    );

    return smoke;
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

    float smoke_alpha = sat(
        1.0 - exp(
            -d * 1.65
        )
    );

    // pass_mode 0 = debug/fullscreen support
    // pass_mode 2 = smoke plume rising from the top of the cube
    if (pass_mode == 2) {
        float dx = (v_uv.x - 0.50) / 0.22;
        float dy = (v_uv.y - 0.27) / 0.20;

        float plume_mask = exp(
            -(dx * dx + dy * dy) * 1.8
        );

        plume_mask *= (
            0.70
            + smoothstep(
                0.18,
                0.44,
                1.0 - v_uv.y
            ) * 0.30
        );

        smoke_alpha *= plume_mask;
    }

    vec3 smoke_flat = flat_smoke_color(
        d
    );

    vec3 smoke_color = smoke_flat;

    if (volumetric_enabled) {
        vec3 smoke_vol = volumetric_smoke_color(
            d
        );

        smoke_color = mix(
            smoke_flat,
            smoke_vol,
            volumetric_strength
        );
    }

    // Premultiplied-alpha accumulator.
    vec3 premul = (
        smoke_color
        * smoke_alpha
    );

    float alpha = smoke_alpha;

    if (show_fire && pass_mode != 2) {
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

        float fire_alpha = (
            flame
            * 0.94
        );

        vec3 fire_rgb = fire_palette(
            sat(
                heat * 0.98
                + burn_term * 0.14
            )
        );

        premul = (
            premul
            * (1.0 - fire_alpha)
            + fire_rgb
            * fire_alpha
        );

        alpha = (
            alpha
            + fire_alpha
            * (1.0 - alpha)
        );

        // Wide red/orange glow, still entirely GPU-side.
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

        float glow = sat(
            (
                heat * 0.55
                + burn_term * 0.45
            ) * 0.72
        );

        glow *= (
            0.25
            + glow_field * 0.75
        );

        vec3 glow_rgb = vec3(
            glow,
            glow * (86.0 / 255.0),
            glow * (10.0 / 255.0)
        );

        premul += (
            glow_rgb
            * glow_field
            * 0.50
        );

        alpha = max(
            alpha,
            glow_field
            * 0.12
        );
    }

    if (alpha < 0.002) {
        discard;
    }

    fragColor = vec4(
        premul,
        alpha
    );
}
"""


def perspective_matrix(
    fov_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    f = (
        1.0
        / math.tan(
            math.radians(
                fov_degrees
            ) * 0.5
        )
    )

    out = np.zeros(
        (4, 4),
        dtype=np.float32,
    )

    out[0, 0] = (
        f / aspect
    )

    out[1, 1] = f

    out[2, 2] = (
        (far + near)
        / (near - far)
    )

    out[2, 3] = (
        (2.0 * far * near)
        / (near - far)
    )

    out[3, 2] = -1.0

    return out


def translation_matrix(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    out = np.eye(
        4,
        dtype=np.float32,
    )

    out[0, 3] = x
    out[1, 3] = y
    out[2, 3] = z

    return out


def scale_matrix(
    scale: float,
) -> np.ndarray:
    out = np.eye(
        4,
        dtype=np.float32,
    )

    out[0, 0] = scale
    out[1, 1] = scale
    out[2, 2] = scale

    return out


def rotation_x_matrix(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_y_matrix(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def create_cube_vertices() -> np.ndarray:
    """
    36 vertices:
        position xyz
        normal xyz
        uv
    """

    faces = [
        # Front
        (
            (0.0, 0.0, 1.0),
            [
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ],
        ),
        # Back
        (
            (0.0, 0.0, -1.0),
            [
                (1.0, -1.0, -1.0),
                (-1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (1.0, 1.0, -1.0),
            ],
        ),
        # Left
        (
            (-1.0, 0.0, 0.0),
            [
                (-1.0, -1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (-1.0, 1.0, -1.0),
            ],
        ),
        # Right
        (
            (1.0, 0.0, 0.0),
            [
                (1.0, -1.0, 1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0),
            ],
        ),
        # Top
        (
            (0.0, 1.0, 0.0),
            [
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
            ],
        ),
        # Bottom
        (
            (0.0, -1.0, 0.0),
            [
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, -1.0, 1.0),
                (-1.0, -1.0, 1.0),
            ],
        ),
    ]

    uv = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]

    triangle_order = (
        0, 1, 2,
        0, 2, 3,
    )

    data: list[float] = []

    for normal, positions in faces:
        for i in triangle_order:
            px, py, pz = (
                positions[i]
            )

            nx, ny, nz = normal
            u, v = uv[i]

            data.extend(
                (
                    px, py, pz,
                    nx, ny, nz,
                    u, v,
                )
            )

    return np.array(
        data,
        dtype=np.float32,
    )


def create_edge_vertices() -> np.ndarray:
    vertices = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    data: list[float] = []

    for a, b in edges:
        data.extend(
            vertices[a]
        )

        data.extend(
            vertices[b]
        )

    return np.array(
        data,
        dtype=np.float32,
    )


class BurningCubeRenderer:
    def __init__(self) -> None:
        self.ctx = moderngl.create_context(
            require=330
        )

        self.cube_program = self.ctx.program(
            vertex_shader=CUBE_VERTEX_SHADER,
            fragment_shader=CUBE_FRAGMENT_SHADER,
        )

        self.flame_program = self.ctx.program(
            vertex_shader=FLAME_SHELL_VERTEX_SHADER,
            fragment_shader=FLAME_SHELL_FRAGMENT_SHADER,
        )

        self.line_program = self.ctx.program(
            vertex_shader=LINE_VERTEX_SHADER,
            fragment_shader=LINE_FRAGMENT_SHADER,
        )

        self.overlay_program = self.ctx.program(
            vertex_shader=OVERLAY_VERTEX_SHADER,
            fragment_shader=OVERLAY_FRAGMENT_SHADER,
        )

        self._create_simulation_textures()
        self._create_cube()
        self._create_overlay_quad()

        self.angle_x = 0.0
        self.angle_y = 0.0

        self.show_edges = True
        self.rotate_cube = True

    # ------------------------------------------------------------------
    # Textures
    # ------------------------------------------------------------------

    def _make_scalar_texture(
        self,
    ):
        tex = self.ctx.texture(
            (
                SIM_W,
                SIM_H,
            ),
            components=1,
            dtype="f4",
        )

        tex.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR,
        )

        tex.repeat_x = False
        tex.repeat_y = False

        return tex

    def _create_simulation_textures(
        self,
    ) -> None:
        self.density_tex = (
            self._make_scalar_texture()
        )

        self.temperature_tex = (
            self._make_scalar_texture()
        )

        self.fuel_tex = (
            self._make_scalar_texture()
        )

        self.burn_tex = (
            self._make_scalar_texture()
        )

        for program in (
            self.overlay_program,
            self.cube_program,
            self.flame_program,
        ):
            if "density_tex" in program:
                program[
                    "density_tex"
                ].value = 0

            if "temperature_tex" in program:
                program[
                    "temperature_tex"
                ].value = 1

            if "fuel_tex" in program:
                program[
                    "fuel_tex"
                ].value = 2

            if "burn_tex" in program:
                program[
                    "burn_tex"
                ].value = 3

        self.overlay_program[
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

    def _load_face_texture(
        self,
    ):
        texture_path = (
            Path(__file__).resolve().with_name(
                "face.png"
            )
        )

        if texture_path.exists():
            image = pygame.image.load(
                str(texture_path)
            ).convert_alpha()

        else:
            # Fallback so the demo still runs if face.png was not copied.
            image = pygame.Surface(
                (256, 256),
                pygame.SRCALPHA,
            )

            image.fill(
                (35, 35, 42, 255)
            )

            for y in range(0, 256, 32):
                for x in range(0, 256, 32):
                    if (
                        (x // 32 + y // 32)
                        % 2
                    ):
                        pygame.draw.rect(
                            image,
                            (100, 25, 22, 255),
                            (
                                x,
                                y,
                                32,
                                32,
                            ),
                        )

            pygame.draw.rect(
                image,
                (230, 230, 235, 255),
                image.get_rect(),
                5,
            )

            print(
                "WARNING: face.png not found next to script; "
                "using generated fallback texture."
            )

        width, height = (
            image.get_size()
        )

        data = pygame.image.tobytes(
            image,
            "RGBA",
            True,
        )

        tex = self.ctx.texture(
            (
                width,
                height,
            ),
            components=4,
            data=data,
        )

        tex.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR,
        )

        tex.repeat_x = False
        tex.repeat_y = False

        return tex

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _create_cube(
        self,
    ) -> None:
        cube_vertices = (
            create_cube_vertices()
        )

        self.cube_vbo = self.ctx.buffer(
            cube_vertices.tobytes()
        )

        self.cube_vao = self.ctx.vertex_array(
            self.cube_program,
            [
                (
                    self.cube_vbo,
                    "3f 3f 2f",
                    "in_pos",
                    "in_normal",
                    "in_uv",
                )
            ],
        )

        self.flame_vao = self.ctx.vertex_array(
            self.flame_program,
            [
                (
                    self.cube_vbo,
                    "3f 3f 2f",
                    "in_pos",
                    "in_normal",
                    "in_uv",
                )
            ],
        )

        edge_vertices = (
            create_edge_vertices()
        )

        self.edge_vbo = self.ctx.buffer(
            edge_vertices.tobytes()
        )

        self.edge_vao = self.ctx.vertex_array(
            self.line_program,
            [
                (
                    self.edge_vbo,
                    "3f",
                    "in_pos",
                )
            ],
        )

        self.face_tex = (
            self._load_face_texture()
        )

        self.cube_program[
            "face_tex"
        ].value = 4

        self.face_tex.use(
            location=4
        )

    def _create_overlay_quad(
        self,
    ) -> None:
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

        self.overlay_vbo = (
            self.ctx.buffer(
                vertices.tobytes()
            )
        )

        self.overlay_vao = (
            self.ctx.vertex_array(
                self.overlay_program,
                [
                    (
                        self.overlay_vbo,
                        "2f 2f",
                        "in_pos",
                        "in_uv",
                    )
                ],
            )
        )

    # ------------------------------------------------------------------
    # Per-frame work
    # ------------------------------------------------------------------

    def upload(
        self,
        simulation: FireSmokeSimulation,
    ) -> None:
        self.density_tex.write(
            np.ascontiguousarray(
                simulation.density,
                dtype=np.float32,
            )
        )

        self.temperature_tex.write(
            np.ascontiguousarray(
                simulation.temperature,
                dtype=np.float32,
            )
        )

        self.fuel_tex.write(
            np.ascontiguousarray(
                simulation.fuel,
                dtype=np.float32,
            )
        )

        self.burn_tex.write(
            np.ascontiguousarray(
                simulation.burn,
                dtype=np.float32,
            )
        )

    def update_rotation(
        self,
        dt: float,
    ) -> None:
        if not self.rotate_cube:
            return

        # Same basic rotation rates as the original cube demo.
        self.angle_x += (
            math.radians(25.0)
            * dt
        )

        self.angle_y += (
            math.radians(40.0)
            * dt
        )

    def _matrices(
        self,
        edge_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        width, height = (
            pygame.display.get_window_size()
        )

        aspect = (
            max(width, 1)
            / max(height, 1)
        )

        projection = perspective_matrix(
            50.0,
            aspect,
            0.1,
            100.0,
        )

        model = (
            translation_matrix(
                0.0,
                -0.20,
                -5.3,
            )
            @ rotation_y_matrix(
                self.angle_y
            )
            @ rotation_x_matrix(
                self.angle_x
            )
            @ scale_matrix(
                0.78
                * edge_scale
            )
        )

        mvp = (
            projection
            @ model
        )

        return model, mvp

    @staticmethod
    def _write_mat4(
        uniform,
        matrix: np.ndarray,
    ) -> None:
        # GLSL matrices are column-major.
        uniform.write(
            np.ascontiguousarray(
                matrix.T,
                dtype=np.float32,
            ).tobytes()
        )

    def _render_flame_shell(
        self,
        alpha_scale: float,
        edge_scale: float,
        depth_test: bool,
    ) -> None:
        model, mvp = self._matrices(
            edge_scale=edge_scale
        )

        self._write_mat4(
            self.flame_program[
                "model"
            ],
            model,
        )

        self._write_mat4(
            self.flame_program[
                "mvp"
            ],
            mvp,
        )

        self.flame_program[
            "alpha_scale"
        ].value = alpha_scale

        self.ctx.enable(
            moderngl.BLEND
        )

        self.ctx.blend_func = (
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        if depth_test:
            self.ctx.enable(
                moderngl.DEPTH_TEST
                | moderngl.CULL_FACE
            )
        else:
            self.ctx.disable(
                moderngl.DEPTH_TEST
            )
            self.ctx.disable(
                moderngl.CULL_FACE
            )

        self.flame_vao.render(
            mode=moderngl.TRIANGLES
        )

    def _set_overlay_uniforms(
        self,
        simulation: FireSmokeSimulation,
        volumetric_enabled: bool,
        volumetric_strength: float,
        pass_mode: int,
    ) -> None:
        if simulation.debug_mode == "temperature":
            debug_mode = 1
        elif simulation.debug_mode == "fuel":
            debug_mode = 2
        else:
            debug_mode = 0

        self.overlay_program[
            "debug_mode"
        ].value = debug_mode

        self.overlay_program[
            "pass_mode"
        ].value = pass_mode

        self.overlay_program[
            "show_fire"
        ].value = simulation.show_fire

        self.overlay_program[
            "volumetric_enabled"
        ].value = volumetric_enabled

        self.overlay_program[
            "volumetric_strength"
        ].value = volumetric_strength

    def _render_overlay(
        self,
        simulation: FireSmokeSimulation,
        volumetric_enabled: bool,
        volumetric_strength: float,
        pass_mode: int,
    ) -> None:
        self._set_overlay_uniforms(
            simulation,
            volumetric_enabled,
            volumetric_strength,
            pass_mode,
        )

        self.ctx.disable(
            moderngl.DEPTH_TEST
            | moderngl.CULL_FACE
        )

        self.ctx.enable(
            moderngl.BLEND
        )

        self.ctx.blend_func = (
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        self.overlay_vao.render(
            mode=moderngl.TRIANGLES
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

        self.ctx.clear(
            4.0 / 255.0,
            5.0 / 255.0,
            8.0 / 255.0,
            1.0,
            depth=1.0,
        )

        # Debug views still use the full-screen overlay.
        if simulation.debug_mode != "normal":
            self._render_overlay(
                simulation,
                volumetric_enabled,
                volumetric_strength,
                pass_mode=0,
            )
            return

        # --------------------------------------------------------------
        # 1. Flame shell BEHIND the cube.
        #
        # Rendered first and then partially hidden by the solid cube,
        # so the flame no longer appears to pass directly through it.
        # --------------------------------------------------------------
        if simulation.show_fire:
            self._render_flame_shell(
                alpha_scale=0.88,
                edge_scale=1.18,
                depth_test=False,
            )

        # --------------------------------------------------------------
        # 2. Solid textured cube.
        # --------------------------------------------------------------
        self.ctx.disable(
            moderngl.BLEND
        )

        self.ctx.enable(
            moderngl.DEPTH_TEST
            | moderngl.CULL_FACE
        )

        model, mvp = (
            self._matrices()
        )

        self._write_mat4(
            self.cube_program[
                "model"
            ],
            model,
        )

        self._write_mat4(
            self.cube_program[
                "mvp"
            ],
            mvp,
        )

        self.face_tex.use(
            location=4
        )

        self.cube_vao.render(
            mode=moderngl.TRIANGLES
        )

        # --------------------------------------------------------------
        # 3. White cube edges from the original demo.
        # --------------------------------------------------------------
        if self.show_edges:
            self.ctx.disable(
                moderngl.CULL_FACE
            )

            _, edge_mvp = (
                self._matrices(
                    edge_scale=1.008
                )
            )

            self._write_mat4(
                self.line_program[
                    "mvp"
                ],
                edge_mvp,
            )

            try:
                self.ctx.line_width = 1.5
            except Exception:
                pass

            self.edge_vao.render(
                mode=moderngl.LINES
            )

        # --------------------------------------------------------------
        # 4. Flame shell IN FRONT of the cube.
        #
        # This is what creates the visible "wrapped around the cube"
        # envelope instead of a flat overlay passing through the mesh.
        # --------------------------------------------------------------
        if simulation.show_fire:
            self._render_flame_shell(
                alpha_scale=0.98,
                edge_scale=1.10,
                depth_test=True,
            )

        # --------------------------------------------------------------
        # 5. Smoke plume only, rising from the top of the cube.
        # --------------------------------------------------------------
        self._render_overlay(
            simulation,
            volumetric_enabled,
            volumetric_strength,
            pass_mode=2,
        )


def main() -> None:
    pygame.init()

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

    pygame.display.gl_set_attribute(
        pygame.GL_DEPTH_SIZE,
        24,
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
    renderer = BurningCubeRenderer()

    paused = False
    running = True

    volumetric_enabled = True
    volumetric_strength = 0.30

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

                elif event.key == pygame.K_e:
                    renderer.show_edges = (
                        not renderer.show_edges
                    )

                elif event.key == pygame.K_p:
                    renderer.rotate_cube = (
                        not renderer.rotate_cube
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

            renderer.update_rotation(
                dt
            )

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
            "Burning Cube v0.4"
            f"  |  {clock.get_fps():5.1f} FPS"
            f"  |  div {simulation.divergence_rms:.4f}"
            f"  |  curl {simulation.curl_rms:.4f}"
            f"  |  temp {max_temp:.2f}"
            f"  |  fuel {max_fuel:.2f}"
            f"  |  ignition {simulation.cube_ignition * 100.0:4.0f}%"
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
