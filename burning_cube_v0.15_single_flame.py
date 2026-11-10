#!/usr/bin/env python3
"""
burning_cube.py v0.15 single-flame

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
SOURCE_Y = SIM_H * 0.89

SOURCE_RADIUS_X = 11.5
SOURCE_RADIUS_Y = 4.4

SOURCE_FUEL_STRENGTH = 4.9
SOURCE_HEAT_STRENGTH = 5.8
SOURCE_SMOKE_STRENGTH = 0.06

SOURCE_UPWARD_SPEED = 11.5


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

# ---------------------------------------------------------------------------
# Burning Cube v0.9: 2D obstacle physics
# ---------------------------------------------------------------------------

# Kuinka voimakkaasti virtaus työnnetään pois kuution pinnasta.
OBSTACLE_REPEL = 29.0

# Auttaa kuumaa kaasua liukumaan kuution sivuja pitkin ylöspäin.
OBSTACLE_LIFT = 17.0

# Kun liekki osuu kuution alapintaan, virtaus jaetaan vasemmalle/oikealle.
OBSTACLE_SPLIT = 34.0

# Kuution yläreunassa sama fuel muuttuu asteittain savuksi.
TOP_FUEL_TO_SMOKE = 1.18
TOP_SMOKE_YIELD = 1.58
TOP_COOLING = 1.10

# ---------------------------------------------------------------------------
# v0.10: the cube itself is combustible
# ---------------------------------------------------------------------------

# Surface needs real flame heat before it starts pyrolysing.
CUBE_SURFACE_IGNITION_TEMP = 0.28

# How quickly an already burning patch ignites neighbouring cube surface.
CUBE_SURFACE_SPREAD = 3.8

# Burning cube surface releases fresh fuel into the adjacent fluid.
CUBE_FUEL_RELEASE = 3.30

# Extra heat released by the burning material.
CUBE_HEAT_RELEASE = 2.60

# Small smoke yield directly from the burning material.
CUBE_SMOKE_RELEASE = 0.10

# Makes the flame cling to and climb the burning surface.
CUBE_FLAME_LIFT = 16.0

# Surface burn decays only very slowly while it remains on the cube.
CUBE_SURFACE_DECAY = 0.996

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
# Burning Cube v0.15 single-flame — screen-space cube ignition masks
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

        # v0.9: projected rotating cube acts as a real obstacle
        # in the 2D fluid field.
        self.obstacle = np.zeros(
            shape,
            dtype=bool,
        )

        self.obstacle_float = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.obstacle_potential = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.obstacle_nx = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.obstacle_ny = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.obstacle_near = np.zeros(
            shape,
            dtype=bool,
        )

        self.obstacle_bottom_zone = np.zeros(
            shape,
            dtype=bool,
        )

        self.obstacle_top_zone = np.zeros(
            shape,
            dtype=bool,
        )

        # v0.10: how strongly each current projected surface patch burns.
        # This is NOT a second independent flame effect: it is combustible
        # material that releases fuel into the same single fluid simulation.
        self.surface_burn = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.surface_burn_tmp = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.obstacle_center_x = SIM_W * 0.5

        self.time = 0.0

        self.divergence_rms = 0.0
        self.curl_rms = 0.0

        self.show_fire = True
        self.debug_mode = "normal"

        # Burning Cube v0.15 single-flame:
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

        self.obstacle.fill(False)
        self.obstacle_float.fill(0.0)
        self.obstacle_potential.fill(0.0)
        self.obstacle_nx.fill(0.0)
        self.obstacle_ny.fill(0.0)
        self.obstacle_near.fill(False)
        self.obstacle_bottom_zone.fill(False)
        self.obstacle_top_zone.fill(False)
        self.surface_burn.fill(0.0)
        self.surface_burn_tmp.fill(0.0)
        self.obstacle_center_x = SIM_W * 0.5

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

    def set_obstacle(
        self,
        mask: np.ndarray,
    ) -> None:
        """
        Set the current projected cube silhouette as a fluid obstacle.

        The mask comes from the actual rotating 3D cube projected into the
        same 240x135 coordinate system used by the fire simulation.
        """

        if mask.shape != self.obstacle.shape:
            raise ValueError(
                "Obstacle mask has wrong shape"
            )

        self.obstacle[:, :] = mask

        obs = self.obstacle_float
        obs[:, :] = mask.astype(
            np.float32
        )

        # A tiny one-cell inflation prevents semi-Lagrangian sampling from
        # leaking through sharp moving edges.
        inflated = (
            mask
            | np.roll(mask, 1, axis=0)
            | np.roll(mask, -1, axis=0)
            | np.roll(mask, 1, axis=1)
            | np.roll(mask, -1, axis=1)
        )

        inflated[0, :] = False
        inflated[-1, :] = False
        inflated[:, 0] = False
        inflated[:, -1] = False

        self.obstacle[:, :] = inflated
        obs[:, :] = inflated.astype(
            np.float32
        )

        # Build a cheap smooth obstacle potential. Its gradient points
        # INTO the cube and is used to deflect approaching velocity.
        potential = self.obstacle_potential

        potential[:, :] = (
            obs
            + 0.48
            * (
                np.roll(obs, 1, axis=0)
                + np.roll(obs, -1, axis=0)
                + np.roll(obs, 1, axis=1)
                + np.roll(obs, -1, axis=1)
            )
            + 0.20
            * (
                np.roll(obs, 2, axis=0)
                + np.roll(obs, -2, axis=0)
                + np.roll(obs, 2, axis=1)
                + np.roll(obs, -2, axis=1)
            )
        )

        gx = self.obstacle_nx
        gy = self.obstacle_ny

        gx.fill(0.0)
        gy.fill(0.0)

        gx[:, 1:-1] = 0.5 * (
            potential[:, 2:]
            - potential[:, :-2]
        )

        gy[1:-1, :] = 0.5 * (
            potential[2:, :]
            - potential[:-2, :]
        )

        magnitude = np.sqrt(
            gx * gx
            + gy * gy
        )

        near = (
            (~self.obstacle)
            & (magnitude > 1.0e-5)
        )

        self.obstacle_near[:, :] = near

        # Keep persistent material ignition attached to the CURRENT
        # projected surface. A small dilation allows it to survive normal
        # frame-to-frame rotation without leaving trails behind the cube.
        surface_keep = (
            near
            | np.roll(near, 1, axis=0)
            | np.roll(near, -1, axis=0)
            | np.roll(near, 1, axis=1)
            | np.roll(near, -1, axis=1)
        )

        self.surface_burn *= surface_keep.astype(
            np.float32
        )

        inv = np.zeros_like(
            magnitude,
            dtype=np.float32,
        )

        inv[near] = (
            1.0
            / magnitude[near]
        )

        gx *= inv
        gy *= inv

        # Fluid cells directly BELOW the projected cube.
        below = (
            (~self.obstacle)
            & np.roll(
                self.obstacle,
                1,
                axis=0,
            )
        )

        # Widen contact zone slightly.
        self.obstacle_bottom_zone[:, :] = (
            below
            | np.roll(
                below,
                1,
                axis=0,
            )
            | np.roll(
                below,
                2,
                axis=0,
            )
        )

        # Fluid cells directly ABOVE the projected cube.
        above = (
            (~self.obstacle)
            & np.roll(
                self.obstacle,
                -1,
                axis=0,
            )
        )

        self.obstacle_top_zone[:, :] = (
            above
            | np.roll(
                above,
                -1,
                axis=0,
            )
            | np.roll(
                above,
                -2,
                axis=0,
            )
            | np.roll(
                above,
                -3,
                axis=0,
            )
        )

        ys, xs = np.nonzero(
            self.obstacle
        )

        if xs.size:
            self.obstacle_center_x = float(
                np.mean(xs)
            )

    def update_cube_ignition(
        self,
        dt: float,
    ) -> None:
        """
        The cube starts ignition only when the ONE flame below actually
        heats fluid cells immediately under the current projected cube.
        """

        contact = self.obstacle_bottom_zone

        count = int(
            np.count_nonzero(
                contact
            )
        )

        if count <= 0:
            return

        contact_heat = float(
            np.mean(
                self.temperature[
                    contact
                ]
            )
        )

        contact_fire = float(
            np.mean(
                self.burn[
                    contact
                ]
            )
        )

        contact_energy = (
            contact_heat
            + contact_fire * 0.20
        )

        if contact_energy > 0.08:
            spread_speed = (
                0.30
                + min(
                    contact_energy,
                    1.8,
                ) * 0.24
            )

            self.cube_ignition = min(
                1.0,
                self.cube_ignition
                + spread_speed * dt,
            )

        elif self.cube_ignition > 0.18:
            self.cube_ignition = min(
                1.0,
                self.cube_ignition
                + 0.045 * dt,
            )

    def burn_cube_surface(
        self,
        dt: float,
    ) -> None:
        """
        Make the cube itself act as combustible material.

        Important distinction:
        - the original flame below is still the ignition source;
        - the cube does not get a separate rendered flame effect;
        - heated surface patches RELEASE FUEL into the SAME fluid field.

        Thus the flame physically strengthens while touching the cube,
        climbs over its projected edges, and dies naturally after leaving
        the fuel-producing surface.
        """

        surface = self.obstacle_near

        if not np.any(surface):
            return

        # Local fire contact ignites only the surface actually being heated.
        local_heat = np.clip(
            (
                self.temperature
                - CUBE_SURFACE_IGNITION_TEMP
            )
            / 1.05,
            0.0,
            1.0,
        )

        local_fire = np.clip(
            self.burn / 1.5,
            0.0,
            1.0,
        )

        contact = np.clip(
            local_heat
            + local_fire * 0.48,
            0.0,
            1.0,
        )

        contact *= surface

        burn = self.surface_burn

        # Directly contacted material ignites quickly.
        burn += (
            contact
            * 2.35
            * dt
        )

        # Fire spreads over the contiguous projected surface.
        # Neighbour max gives us an inexpensive surface flame-front.
        neighbour = self.surface_burn_tmp

        neighbour[:, :] = np.maximum.reduce(
            (
                np.roll(burn, 1, axis=0),
                np.roll(burn, -1, axis=0),
                np.roll(burn, 1, axis=1),
                np.roll(burn, -1, axis=1),
                np.roll(
                    np.roll(
                        burn,
                        1,
                        axis=0,
                    ),
                    1,
                    axis=1,
                ),
                np.roll(
                    np.roll(
                        burn,
                        1,
                        axis=0,
                    ),
                    -1,
                    axis=1,
                ),
                np.roll(
                    np.roll(
                        burn,
                        -1,
                        axis=0,
                    ),
                    1,
                    axis=1,
                ),
                np.roll(
                    np.roll(
                        burn,
                        -1,
                        axis=0,
                    ),
                    -1,
                    axis=1,
                ),
            )
        )

        # Slight upward preference makes ignition crawl naturally from the
        # initial lower contact toward the cube's upper edges.
        upward_seed = np.roll(
            burn,
            -1,
            axis=0,
        )

        spread_target = np.maximum(
            neighbour * 0.92,
            upward_seed,
        )

        spread_target *= surface

        spread = np.maximum(
            spread_target
            - burn,
            0.0,
        )

        burn += (
            spread
            * CUBE_SURFACE_SPREAD
            * dt
        )

        burn *= CUBE_SURFACE_DECAY

        # Burning material must stay attached to the actual cube surface.
        burn *= surface.astype(
            np.float32
        )

        np.clip(
            burn,
            0.0,
            1.0,
            out=burn,
        )

        # Global ignition is now simply a useful debug/progress indication.
        burning_cells = burn[
            surface
        ]

        if burning_cells.size:
            surface_level = float(
                np.mean(
                    burning_cells
                )
            )

            self.cube_ignition = max(
                self.cube_ignition,
                min(
                    1.0,
                    surface_level * 1.85,
                ),
            )

        # ---------------------------------------------------------------
        # Pyrolysis:
        # burning solid material emits fuel and heat into adjacent fluid.
        # ---------------------------------------------------------------

        material_fire = (
            burn
            * surface
        )

        # Stronger fuel production in active flame contact; already-burning
        # surface continues feeding the flame after the initial contact.
        contact_boost = (
            0.72
            + contact * 0.58
        )

        released_fuel = (
            material_fire
            * contact_boost
            * CUBE_FUEL_RELEASE
            * dt
        )

        self.fuel += released_fuel

        self.temperature += (
            material_fire
            * CUBE_HEAT_RELEASE
            * dt
        )

        self.density += (
            material_fire
            * CUBE_SMOKE_RELEASE
            * dt
        )

        # The freshly released hot fuel sticks to the surface long enough
        # to wrap over the cube instead of immediately peeling away.
        self.vy -= (
            material_fire
            * CUBE_FLAME_LIFT
            * dt
        )

        # Along the lower half, help the burning material feed the same
        # left/right split that already deflects the original flame.
        split_sign = np.sign(
            GRID_X
            - self.obstacle_center_x
        ).astype(
            np.float32
        )

        self.vx += (
            split_sign
            * material_fire
            * 5.5
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

    def apply_obstacle_velocity(
        self,
        dt: float,
    ) -> None:
        """
        Deflect velocity around the cube instead of letting the flame cross it.

        This is deliberately inexpensive rather than a full solid-aware
        Navier-Stokes solve, but the important behavior is physical:
        one flame hits an obstacle, splits around it, climbs its sides and
        rejoins above it.
        """

        obs = self.obstacle
        near = self.obstacle_near

        # Nothing moves inside the solid cube.
        self.vx[obs] = 0.0
        self.vy[obs] = 0.0

        if not np.any(near):
            return

        nx = self.obstacle_nx
        ny = self.obstacle_ny

        # Positive dot = velocity pointing INTO the obstacle.
        dot = (
            self.vx * nx
            + self.vy * ny
        )

        approach = np.maximum(
            dot,
            0.0,
        )

        approach *= near

        # Remove the inward normal component -> velocity becomes tangential.
        self.vx -= (
            nx
            * approach
            * 1.08
        )

        self.vy -= (
            ny
            * approach
            * 1.08
        )

        activity = np.clip(
            self.temperature / 1.2
            + self.fuel / 1.0
            + self.density * 0.16
            + self.surface_burn * 0.55,
            0.0,
            1.0,
        )

        activity *= near

        # Push away from the solid surface.
        self.vx -= (
            nx
            * OBSTACLE_REPEL
            * activity
            * dt
        )

        self.vy -= (
            ny
            * OBSTACLE_REPEL
            * activity
            * dt
        )

        # Hot gas naturally wants to climb while touching the cube.
        self.vy -= (
            OBSTACLE_LIFT
            * activity
            * dt
        )

        # At the lower face, split the one incoming flame left/right.
        bottom = self.obstacle_bottom_zone

        split_sign = np.sign(
            GRID_X
            - self.obstacle_center_x
        ).astype(
            np.float32
        )

        split_activity = np.clip(
            self.temperature / 1.0
            + self.fuel / 0.9,
            0.0,
            1.0,
        )

        self.vx += (
            split_sign
            * bottom
            * split_activity
            * OBSTACLE_SPLIT
            * dt
        )

        self.vy[obs] = 0.0
        self.vx[obs] = 0.0

    def apply_obstacle_scalars(
        self,
        dt: float,
    ) -> None:
        """
        The cube contains no fluid. At its upper edge the remaining hot fuel
        gradually becomes smoke, giving a continuous fire -> smoke transition.
        """

        obs = self.obstacle

        self.density[obs] = 0.0
        self.temperature[obs] = 0.0
        self.fuel[obs] = 0.0
        self.burn[obs] = 0.0

        if self.cube_ignition <= 0.001:
            return

        top = self.obstacle_top_zone

        conversion = np.minimum(
            self.fuel,
            top
            * self.cube_ignition
            * TOP_FUEL_TO_SMOKE
            * dt,
        )

        self.fuel -= conversion

        self.density += (
            conversion
            * TOP_SMOKE_YIELD
        )

        cool = np.clip(
            top
            * self.cube_ignition
            * TOP_COOLING
            * dt,
            0.0,
            0.20,
        )

        self.temperature *= (
            1.0 - cool
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

        # Pressure projection is not fully solid-aware, so enforce the
        # moving cube boundary again afterwards.
        self.apply_obstacle_velocity(
            0.0
        )

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

        # 2. The original lower flame must first reach the cube.
        self.update_cube_ignition(
            dt
        )

        # 3. Once heated, the cube becomes combustible material.
        # It releases fuel into the SAME fluid field only along the actual
        # projected surface touched/spread by the fire.
        self.burn_cube_surface(
            dt
        )

        # 4. Combust all fuel: original source + fuel released by the cube.
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

        # 9. The projected rotating cube is now a real flow obstacle.
        self.apply_obstacle_velocity(
            dt
        )

        # 10. Pressure projection.
        self.project_velocity()

        # 11. Advect scalar-kentät samalla velocityllä.
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

        # 12. Remove fluid inside the cube and convert hot fuel to smoke
        # as it clears the upper edge.
        self.apply_obstacle_scalars(
            dt
        )

        # 13. Dissipation / cooling.
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
# Burning Cube v0.15 single-flame — ModernGL renderer
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

in vec3 v_normal;
in vec2 v_uv;

out vec4 fragColor;


void main() {
    vec3 base = texture(
        face_tex,
        v_uv
    ).rgb;

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

    fragColor = vec4(
        clamp(
            base * diffuse,
            0.0,
            1.0
        ),
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
uniform float cube_ignition;
uniform float simulation_time;

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
    // IMPORTANT:
    // The combustion simulation is screen-space and the cube occupies
    // only a small central region of it. Sampling the complete 0..1
    // texture with face UVs made almost the whole shell sample empty
    // fluid cells in v0.4.
    //
    // Map every cube face specifically into the active cube/fire region.
    vec2 sim_uv = vec2(
        mix(
            0.29,
            0.71,
            v_uv.x
        ),
        mix(
            0.74,
            0.34,
            v_uv.y
        )
    );

    vec2 sample_dx = vec2(
        2.0 / 240.0,
        0.0
    );

    vec2 sample_dy = vec2(
        0.0,
        2.0 / 135.0
    );

    float temp = texture(
        temperature_tex,
        sim_uv
    ).r;

    temp = max(
        temp,
        texture(
            temperature_tex,
            sim_uv + sample_dx
        ).r
    );

    temp = max(
        temp,
        texture(
            temperature_tex,
            sim_uv - sample_dx
        ).r
    );

    temp = max(
        temp,
        texture(
            temperature_tex,
            sim_uv + sample_dy
        ).r
    );

    temp = max(
        temp,
        texture(
            temperature_tex,
            sim_uv - sample_dy
        ).r
    );

    float fuel = texture(
        fuel_tex,
        sim_uv
    ).r;

    fuel = max(
        fuel,
        texture(
            fuel_tex,
            sim_uv + sample_dx
        ).r
    );

    fuel = max(
        fuel,
        texture(
            fuel_tex,
            sim_uv - sample_dx
        ).r
    );

    float burn = texture(
        burn_tex,
        sim_uv
    ).r;

    burn = max(
        burn,
        texture(
            burn_tex,
            sim_uv + sample_dy
        ).r
    );

    burn = max(
        burn,
        texture(
            burn_tex,
            sim_uv - sample_dy
        ).r
    );

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

    float physical_flame = sat(
        heat
        * (
            fuel_term * 0.72
            + burn_term * 0.78
        )
        * 2.30
    );

    // Shape the fire as a shell following the cube surface.
    float edge_dist = max(
        abs(v_uv.x - 0.5),
        abs(v_uv.y - 0.5)
    );

    float edge_ring = smoothstep(
        0.13,
        0.46,
        edge_dist
    );

    // Slight animated breakup keeps the shell from looking like a
    // perfectly solid glowing box.
    float flicker = (
        0.78
        + 0.12
        * sin(
            simulation_time * 5.1
            + v_uv.x * 13.0
            + v_uv.y * 7.0
        )
        + 0.10
        * sin(
            simulation_time * 8.7
            - v_uv.x * 9.0
            + v_uv.y * 15.0
        )
    );

    // Once the cube is ignited, maintain a real visible flame envelope.
    // The live fuel/temperature field still drives most of the variation,
    // but a turbulent frame can no longer make the shell vanish.
    float ignition_shell = (
        cube_ignition
        * (
            0.18
            + edge_ring * 0.62
        )
        * (
            0.50
            + heat * 0.34
            + burn_term * 0.16
        )
        * flicker
    );

    float flame = max(
        physical_flame,
        ignition_shell
    );

    // The top still burns, but slightly less strongly because this is
    // where flame gradually becomes the smoke plume.
    float top_transition = (
        1.0
        - smoothstep(
            0.76,
            1.00,
            v_uv.y
        ) * 0.20
    );

    flame *= top_transition;
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



CUBE_MASK_VERTEX_SHADER = """
#version 330

in vec3 in_pos;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(
        in_pos,
        1.0
    );
}
"""


CUBE_MASK_FRAGMENT_SHADER = """
#version 330

out vec4 fragColor;

void main() {
    fragColor = vec4(
        1.0,
        1.0,
        1.0,
        1.0
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
uniform sampler2D cube_mask_tex;

uniform vec2 texel_size;

uniform int debug_mode;
uniform int pass_mode;
uniform bool show_fire;
uniform bool volumetric_enabled;
uniform float volumetric_strength;
uniform float cube_ignition;
uniform float simulation_time;
uniform vec4 cube_bounds;

in vec2 v_uv;
out vec4 fragColor;


float sat(float x) {
    return clamp(x, 0.0, 1.0);
}


float hash21(vec2 p) {
    p = fract(
        p * vec2(
            123.34,
            456.21
        )
    );

    p += dot(
        p,
        p + 45.32
    );

    return fract(
        p.x * p.y
    );
}


float value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    f = f * f * (
        3.0 - 2.0 * f
    );

    float a = hash21(
        i
    );

    float b = hash21(
        i + vec2(1.0, 0.0)
    );

    float c = hash21(
        i + vec2(0.0, 1.0)
    );

    float d = hash21(
        i + vec2(1.0, 1.0)
    );

    return mix(
        mix(a, b, f.x),
        mix(c, d, f.x),
        f.y
    );
}


float flame_noise(vec2 uv) {
    vec2 p = vec2(
        uv.x * 7.0,
        uv.y * 5.2
        - simulation_time * 1.15
    );

    float n = 0.0;

    n += value_noise(
        p
    ) * 0.55;

    n += value_noise(
        p * 2.05
        + vec2(
            simulation_time * 0.18,
            4.7
        )
    ) * 0.28;

    n += value_noise(
        p * 4.10
        - vec2(
            simulation_time * 0.24,
            2.1
        )
    ) * 0.17;

    return n;
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



float source_flame_energy() {
    // Sample only the ONE actual flame source below the cube.
    // These points cover the widened source around x=0.5, y≈0.84.
    float e = 0.0;

    e = max(
        e,
        expanded_flame(
            vec2(0.50, 0.89)
        )
    );

    e = max(
        e,
        expanded_flame(
            vec2(0.46, 0.89)
        )
    );

    e = max(
        e,
        expanded_flame(
            vec2(0.54, 0.89)
        )
    );

    e = max(
        e,
        expanded_flame(
            vec2(0.50, 0.85)
        )
    );

    e = max(
        e,
        expanded_flame(
            vec2(0.43, 0.91)
        )
    );

    e = max(
        e,
        expanded_flame(
            vec2(0.57, 0.91)
        )
    );

    return sat(e);
}


float pulled_flame_from_below(vec2 uv) {
    // Sample the same physical flame field at points progressively lower
    // and closer to the source center. This visually bends the one flame
    // around the obstacle without creating any new simulation source.
    float e = 0.0;

    vec2 p1 = vec2(
        mix(
            uv.x,
            0.50,
            0.38
        ),
        min(
            uv.y + 0.12,
            0.98
        )
    );

    vec2 p2 = vec2(
        mix(
            uv.x,
            0.50,
            0.56
        ),
        min(
            uv.y + 0.22,
            0.98
        )
    );

    vec2 p3 = vec2(
        mix(
            uv.x,
            0.50,
            0.72
        ),
        min(
            uv.y + 0.34,
            0.98
        )
    );

    vec2 p4 = vec2(
        mix(
            uv.x,
            0.50,
            0.86
        ),
        min(
            uv.y + 0.46,
            0.98
        )
    );

    e = max(
        e,
        expanded_flame(p1)
    );

    e = max(
        e,
        expanded_flame(p2) * 0.92
    );

    e = max(
        e,
        expanded_flame(p3) * 0.82
    );

    e = max(
        e,
        expanded_flame(p4) * 0.70
    );

    return sat(e);
}


float cube_mask_at(vec2 uv) {
    uv = clamp(
        uv,
        vec2(0.0),
        vec2(1.0)
    );

    return texture(
        cube_mask_tex,
        uv
    ).r;
}


float cube_outer_rim(vec2 uv) {
    float center = cube_mask_at(
        uv
    );

    vec2 t = texel_size;

    float dilated = center;

    // Several rings make a soft but clearly visible flame corridor
    // outside the projected cube silhouette.
    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 2.0,  0.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-2.0,  0.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 0.0,  2.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 0.0, -2.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 4.0,  0.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-4.0,  0.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 0.0,  4.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 0.0, -4.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 6.0,  2.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-6.0,  2.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 6.0, -2.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-6.0, -2.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 9.0,  0.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-9.0,  0.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 0.0,  9.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 0.0, -9.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 12.0,  4.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-12.0,  4.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2( 12.0, -4.0) * t
        )
    );

    dilated = max(
        dilated,
        cube_mask_at(
            uv + vec2(-12.0, -4.0) * t
        )
    );

    return clamp(
        dilated - center,
        0.0,
        1.0
    );
}


vec2 engulf_source_uv(
    vec2 uv,
    float phase
) {
    float left = cube_bounds.x;
    float right = cube_bounds.y;
    float top = cube_bounds.z;
    float bottom = cube_bounds.w;

    float width = max(
        right - left,
        0.01
    );

    float height = max(
        bottom - top,
        0.01
    );

    // Position inside the cube's SCREEN-SPACE bounding rectangle.
    // This is only for stretching the source flame. No 3D silhouette
    // information is used.
    float local_x = sat(
        (uv.x - left)
        / width
    );

    float local_y = sat(
        (uv.y - top)
        / height
    );

    float n = flame_noise(
        vec2(
            local_x,
            local_y
        )
        + vec2(
            phase * 0.13,
            -phase * 0.08
        )
    );

    // Stretch the actual narrow lower flame across the complete cube width.
    //
    // left side of cube -> left part of the real flame
    // right side       -> right part of the real flame
    //
    // The sampled source remains narrow in source space; stretching happens
    // solely because the destination area is much wider.
    float source_x = mix(
        0.425,
        0.575,
        local_x
    );

    source_x += (
        (n - 0.50) * 0.030
        + sin(
            simulation_time * 3.1
            + local_y * 12.0
            + phase
        ) * 0.010
    );

    // Bottom of cube samples the bright base of the source flame.
    // Top of cube samples the upper flame tongues.
    float source_y = mix(
        0.79,
        0.965,
        local_y
    );

    source_y += (
        (n - 0.50) * 0.018
        + sin(
            simulation_time * 3.8
            + local_x * 16.0
            + phase * 1.6
        ) * 0.007
    );

    return clamp(
        vec2(
            source_x,
            source_y
        ),
        vec2(0.0),
        vec2(1.0)
    );
}


float stretched_source_flame(
    vec2 uv,
    out vec2 hottest_uv
) {
    vec2 s0 = engulf_source_uv(
        uv,
        0.0
    );

    vec2 s1 = engulf_source_uv(
        uv,
        1.7
    );

    vec2 s2 = engulf_source_uv(
        uv,
        3.4
    );

    float f0 = expanded_flame(
        s0
    );

    float f1 = expanded_flame(
        s1
    ) * 0.90;

    float f2 = expanded_flame(
        s2
    ) * 0.82;

    float f = f0;
    hottest_uv = s0;

    if (f1 > f) {
        f = f1;
        hottest_uv = s1;
    }

    if (f2 > f) {
        f = f2;
        hottest_uv = s2;
    }

    return sat(f);
}


float cube_width_region(
    vec2 uv,
    float widen
) {
    float left = cube_bounds.x;
    float right = cube_bounds.y;

    float cube_width = max(
        right - left,
        0.01
    );

    float center = (
        left + right
    ) * 0.5;

    // Approximate width of the original lower flame in screen UV.
    float source_half_width = 0.075;

    float cube_half_width = (
        cube_width * 0.5
        + cube_width * 0.055
    );

    float half_width = mix(
        source_half_width,
        cube_half_width,
        sat(widen)
    );

    float n = flame_noise(
        vec2(
            uv.x * 1.7,
            uv.y * 0.9
        )
    );

    // Flame edges remain organic while widening.
    float edge_warp = (
        (n - 0.5)
        * cube_width
        * 0.035
        * sat(widen)
    );

    float soft = mix(
        0.010,
        max(
            cube_width * 0.028,
            0.007
        ),
        sat(widen)
    );

    float left_edge = (
        center
        - half_width
        + edge_warp
    );

    float right_edge = (
        center
        + half_width
        + edge_warp
    );

    float left_mask = smoothstep(
        left_edge - soft,
        left_edge + soft,
        uv.x
    );

    float right_mask = (
        1.0
        - smoothstep(
            right_edge - soft,
            right_edge + soft,
            uv.x
        )
    );

    return sat(
        left_mask * right_mask
    );
}


float cube_flame_column_region(
    vec2 uv
) {
    float left = cube_bounds.x;
    float right = cube_bounds.y;
    float top = cube_bounds.z;
    float bottom = cube_bounds.w;

    float width = max(
        right - left,
        0.01
    );

    // Allow flame to lick slightly outside the left/right cube edges.
    float side_margin = (
        width * 0.055
    );

    left -= side_margin;
    right += side_margin;

    float n = flame_noise(
        vec2(
            uv.x * 1.4,
            uv.y * 0.8
        )
    );

    // Turbulent side edges so even the maximum width does not form
    // a clean rectangular boundary.
    float edge_warp = (
        (n - 0.5)
        * width
        * 0.045
    );

    float soft = max(
        width * 0.030,
        0.006
    );

    float left_mask = smoothstep(
        left - soft + edge_warp,
        left + soft + edge_warp,
        uv.x
    );

    float right_mask = (
        1.0
        - smoothstep(
            right - soft + edge_warp,
            right + soft + edge_warp,
            uv.x
        )
    );

    // Only vertical range matters here. The flame is NOT shaped like the
    // cube; it is simply a wide 2D flame column passing through it.
    float bottom_mask = (
        1.0
        - smoothstep(
            bottom + 0.045,
            bottom + 0.080,
            uv.y
        )
    );

    float top_mask = smoothstep(
        top - 0.075,
        top - 0.015,
        uv.y
    );

    return sat(
        left_mask
        * right_mask
        * bottom_mask
        * top_mask
    );
}


float cube_front_flame_region(vec2 uv) {
    float body = cube_mask_at(
        uv
    );

    float rim = cube_outer_rim(
        uv
    );

    // Entire projected cube plus flame licking somewhat beyond its edges.
    return sat(
        body
        + rim * 0.92
    );
}


float cube_top_connection(vec2 uv) {
    // Outside the cube, immediately above its current projected top edge.
    float center = cube_mask_at(
        uv
    );

    if (center > 0.5) {
        return 0.0;
    }

    vec2 t = texel_size;

    float below = 0.0;

    below = max(
        below,
        cube_mask_at(
            uv + vec2(0.0, 2.0) * t
        )
    );

    below = max(
        below,
        cube_mask_at(
            uv + vec2(0.0, 4.0) * t
        )
    );

    below = max(
        below,
        cube_mask_at(
            uv + vec2(0.0, 7.0) * t
        )
    );

    below = max(
        below,
        cube_mask_at(
            uv + vec2(0.0, 11.0) * t
        )
    );

    return below;
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

    // pass_mode 0 = the ONE physical fire/smoke field behind the cube
    // pass_mode 1 = legacy rim-only flame mode
    // pass_mode 2 = smoke connection immediately above the cube
    // pass_mode 3 = ONE widened 2D flame across the whole cube front
    if (
        pass_mode == 1
        || pass_mode == 3
        || pass_mode == 4
    ) {
        smoke_alpha = 0.0;
    }

    if (pass_mode == 2) {
        float top_connection = cube_top_connection(
            v_uv
        );

        float soft_top = smoothstep(
            0.10,
            0.90,
            top_connection
        );

        smoke_alpha *= (
            soft_top
            * (
                0.30
                + cube_ignition * 0.34
            )
        );
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

    if (
        show_fire
        && pass_mode != 0
        && pass_mode != 2
    ) {
        float heat = sat(
            (temp - 0.18)
            / (3.2 - 0.18)
        );

        float burn_term = sat(
            burn / 1.6
        );

        // Normally color comes from local heat/burn.
        // In the silhouette-wrap pass these are replaced with values
        // derived from the ONE flame source below the cube.
        float render_heat = heat;
        float render_burn = burn_term;

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

        // Fire naturally loses emissive flame above the cube and becomes
        // mainly smoke. v_uv.y = 0 is the top of the screen.
        float upper_flame_fade = smoothstep(
            0.18,
            0.36,
            v_uv.y
        );

        flame *= upper_flame_fade;

        if (pass_mode == 1) {
            float rim = cube_outer_rim(
                v_uv
            );

            float source_energy = source_flame_energy();

            float pulled_energy = pulled_flame_from_below(
                v_uv
            );

            // ONE source only:
            // the active flame below supplies both energy and color.
            float wrap_energy = max(
                pulled_energy,
                source_energy * 0.66
            );

            // Ignition progresses from the cube's lower silhouette toward
            // the top. It no longer appears uniformly around the whole
            // cube the instant contact happens.
            float height_from_bottom = sat(
                (0.80 - v_uv.y)
                / 0.58
            );

            float climb_limit = (
                0.08
                + cube_ignition * 1.02
            );

            float ignition_climb = (
                1.0
                - smoothstep(
                    climb_limit,
                    climb_limit + 0.13,
                    height_from_bottom
                )
            );

            // Bottom is strongest; upper edge gradually turns to smoke.
            float vertical_wrap = mix(
                0.42,
                1.00,
                smoothstep(
                    0.28,
                    0.79,
                    v_uv.y
                )
            );

            float lower_contact = smoothstep(
                0.54,
                0.84,
                v_uv.y
            );

            vertical_wrap *= (
                0.80
                + lower_contact * 0.20
            );

            float flicker = (
                0.84
                + 0.09
                * sin(
                    simulation_time * 5.3
                    + v_uv.x * 19.0
                    + v_uv.y * 6.0
                )
                + 0.07
                * sin(
                    simulation_time * 8.1
                    - v_uv.y * 23.0
                    + v_uv.x * 8.0
                )
            );

            float tongue = (
                0.74
                + 0.26
                * sat(
                    0.5
                    + 0.5
                    * sin(
                        v_uv.x * 31.0
                        + v_uv.y * 17.0
                        + simulation_time * 4.7
                    )
                )
            );

            flame = sat(
                wrap_energy
                * rim
                * ignition_climb
                * vertical_wrap
                * flicker
                * tongue
                * 1.58
            );

            // CRITICAL v0.8 FIX:
            // v0.7 used LOCAL heat here. Around the cube rim local heat is
            // often zero, so fire_rgb became black while alpha remained
            // non-zero -> a black ghost cube with premultiplied blending.
            //
            // Use the actual lower flame's energy for the rim palette.
            render_heat = sat(
                0.34
                + source_energy * 0.58
                + pulled_energy * 0.18
            );

            render_burn = sat(
                0.18
                + source_energy * 0.42
            );
        }

        if (pass_mode == 3) {
            float region = cube_flame_column_region(
                v_uv
            );

            vec2 source_uv;

            // Sample the exact same physical flame that exists below.
            // It is only stretched horizontally and vertically.
            float sampled_flame = stretched_source_flame(
                v_uv,
                source_uv
            );

            float left = cube_bounds.x;
            float right = cube_bounds.y;
            float top = cube_bounds.z;
            float bottom = cube_bounds.w;

            float height = max(
                bottom - top,
                0.01
            );

            float local_y = sat(
                (v_uv.y - top)
                / height
            );

            // Ignition grows bottom -> top. At full ignition the 2D flame
            // simply spans the entire cube height.
            float height_from_bottom = (
                1.0 - local_y
            );

            float climb_limit = (
                0.08
                + cube_ignition * 1.12
            );

            float climb = (
                1.0
                - smoothstep(
                    climb_limit,
                    climb_limit + 0.16,
                    height_from_bottom
                )
            );

            float n = flame_noise(
                vec2(
                    (v_uv.x - left)
                    / max(right - left, 0.01),
                    local_y
                )
            );

            // Retain open holes and rising tongues from the real flame.
            float tongue_wave = (
                0.5
                + 0.5
                * sin(
                    v_uv.x * 33.0
                    - simulation_time * 5.0
                    + n * 5.0
                )
            );

            float breakup = smoothstep(
                0.28,
                0.68,
                n * 0.58
                + tongue_wave * 0.42
            );

            // Strong enough to reach the top, but begins thinning there
            // so the physical smoke plume can take over.
            float top_transition = mix(
                0.72,
                1.00,
                smoothstep(
                    0.18,
                    0.62,
                    local_y
                )
            );

            flame = clamp(
                sampled_flame
                * region
                * climb
                * (
                    0.48
                    + breakup * 0.52
                )
                * top_transition
                * 1.30,
                0.0,
                0.86
            );

            // Exact colour comes from the same lower-flame sample.
            float sampled_temp = texture(
                temperature_tex,
                source_uv
            ).r;

            float sampled_burn = texture(
                burn_tex,
                source_uv
            ).r;

            render_heat = clamp(
                (
                    sampled_temp - 0.18
                )
                / (
                    3.2 - 0.18
                ),
                0.16,
                0.88
            );

            render_burn = clamp(
                sampled_burn / 1.6,
                0.08,
                0.74
            );
        }

        if (pass_mode == 4) {
            float left = cube_bounds.x;
            float right = cube_bounds.y;
            float top = cube_bounds.z;
            float bottom = cube_bounds.w;

            float cube_width = max(
                right - left,
                0.01
            );

            float cube_height = max(
                bottom - top,
                0.01
            );

            // ------------------------------------------------------------
            // ONE FLAME, ONE CONTINUOUS COORDINATE TRANSFORM
            // ------------------------------------------------------------
            //
            // Far below the cube:
            //     morph = 0 -> source_uv == v_uv
            //
            // At the lower cube edge:
            //     morph grows smoothly
            //
            // Through the cube:
            //     morph = 1 -> the SAME lower flame is stretched to the
            //     cube width and height.
            //
            // There is no second fire texture or second flame source.
            // ------------------------------------------------------------

            float morph = (
                1.0
                - smoothstep(
                    bottom - 0.035,
                    bottom + 0.115,
                    v_uv.y
                )
            );

            // Keep full morph while crossing the cube and slightly above it.
            float above_fade = smoothstep(
                top - 0.115,
                top + 0.020,
                v_uv.y
            );

            // Local vertical position through the cube.
            float local_y = sat(
                (v_uv.y - top)
                / cube_height
            );

            // A SINGLE source coordinate. No multi-sample/max copies.
            vec2 stretched_uv = engulf_source_uv(
                v_uv,
                0.0
            );

            // Exact original fire below, continuously transforming into
            // the wide version. This removes the visual seam between
            // "source flame" and "cube flame".
            vec2 continuous_uv = mix(
                v_uv,
                stretched_uv,
                morph
            );

            float continuous_flame = expanded_flame(
                continuous_uv
            );

            // Width changes continuously with the SAME morph factor.
            float width_mask = cube_width_region(
                v_uv,
                morph
            );

            // Original narrow fire below should not be clipped by the
            // cube-width mask before widening starts.
            float width_gate = mix(
                1.0,
                width_mask,
                morph
            );

            // Ignition controls only how high the widened part may climb.
            // It does NOT create another flame.
            float height_from_bottom = (
                1.0 - local_y
            );

            float climb_limit = (
                0.10
                + cube_ignition * 1.10
            );

            float climb = (
                1.0
                - smoothstep(
                    climb_limit,
                    climb_limit + 0.15,
                    height_from_bottom
                )
            );

            // Below the cube, ignition gating is irrelevant.
            float ignition_gate = mix(
                1.0,
                climb,
                morph
            );

            float n = flame_noise(
                vec2(
                    (v_uv.x - left)
                    / cube_width,
                    v_uv.y
                )
            );

            // Gentle domain breakup only; never a second set of tongues.
            float turbulence = (
                0.84
                + 0.16
                * smoothstep(
                    0.22,
                    0.78,
                    n
                )
            );

            // Let flame reach a little beyond the top before smoke wins.
            float top_exit = mix(
                1.0,
                above_fade,
                morph
            );

            flame = clamp(
                continuous_flame
                * width_gate
                * ignition_gate
                * turbulence
                * top_exit
                * 1.12,
                0.0,
                0.90
            );

            // Colour comes from the exact SAME continuous sample position.
            float sampled_temp = texture(
                temperature_tex,
                continuous_uv
            ).r;

            float sampled_burn = texture(
                burn_tex,
                continuous_uv
            ).r;

            render_heat = clamp(
                (
                    sampled_temp - 0.18
                )
                / (
                    3.2 - 0.18
                ),
                0.10,
                0.92
            );

            render_burn = clamp(
                sampled_burn / 1.6,
                0.05,
                0.78
            );
        }

        float fire_alpha_scale = (
            pass_mode == 4
            ? 0.88
            : (
                pass_mode == 3
                ? 0.76
                : 0.94
            )
        );

        float fire_alpha = (
            flame
            * fire_alpha_scale
        );

        vec3 fire_rgb = fire_palette(
            sat(
                render_heat * 0.98
                + render_burn * 0.14
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
                render_heat * 0.55
                + render_burn * 0.45
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

        float glow_mask = 1.0;
        float effective_glow_field = glow_field;

        if (pass_mode == 1) {
            glow_mask = (
                cube_outer_rim(
                    v_uv
                )
                * cube_ignition
            );

            effective_glow_field = max(
                flame,
                source_flame_energy() * 0.34
            );
        }

        float glow_strength = 0.50;
        float glow_alpha_strength = 0.12;

        if (pass_mode == 3) {
            glow_mask = (
                cube_flame_column_region(
                    v_uv
                )
                * smoothstep(
                    0.06,
                    0.56,
                    flame
                )
            );

            effective_glow_field = flame;

            glow_strength = 0.15;
            glow_alpha_strength = 0.028;
        }

        if (pass_mode == 4) {
            // The glow is generated from the very same visible flame.
            // No extra mask, ring, side flame or front flame is added.
            glow_mask = smoothstep(
                0.04,
                0.50,
                flame
            );

            effective_glow_field = flame;

            glow_strength = 0.18;
            glow_alpha_strength = 0.032;
        }

        premul += (
            glow_rgb
            * effective_glow_field
            * glow_strength
            * glow_mask
        );

        alpha = max(
            alpha,
            effective_glow_field
            * glow_alpha_strength
            * glow_mask
        );
    }

    if (
        pass_mode == 3
        || pass_mode == 4
    ) {
        vec3 safe_limit = vec3(
            max(alpha * 1.85, 0.08)
        );

        premul = min(
            premul,
            safe_limit
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



def rasterize_triangle_into_mask(
    mask: np.ndarray,
    points: np.ndarray,
) -> None:
    """
    Rasterize one projected triangle directly into the 240x135 boolean mask.
    """

    min_x = max(
        0,
        int(
            math.floor(
                float(
                    np.min(
                        points[:, 0]
                    )
                )
            )
        ),
    )

    max_x = min(
        SIM_W - 1,
        int(
            math.ceil(
                float(
                    np.max(
                        points[:, 0]
                    )
                )
            )
        ),
    )

    min_y = max(
        0,
        int(
            math.floor(
                float(
                    np.min(
                        points[:, 1]
                    )
                )
            )
        ),
    )

    max_y = min(
        SIM_H - 1,
        int(
            math.ceil(
                float(
                    np.max(
                        points[:, 1]
                    )
                )
            )
        ),
    )

    if (
        max_x < min_x
        or max_y < min_y
    ):
        return

    x = (
        GRID_X[
            min_y:max_y + 1,
            min_x:max_x + 1,
        ]
        + 0.5
    )

    y = (
        GRID_Y[
            min_y:max_y + 1,
            min_x:max_x + 1,
        ]
        + 0.5
    )

    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]

    denom = (
        (y1 - y2)
        * (x0 - x2)
        + (x2 - x1)
        * (y0 - y2)
    )

    if abs(
        float(denom)
    ) < 1.0e-8:
        return

    a = (
        (y1 - y2)
        * (x - x2)
        + (x2 - x1)
        * (y - y2)
    ) / denom

    b = (
        (y2 - y0)
        * (x - x2)
        + (x0 - x2)
        * (y - y2)
    ) / denom

    c = (
        1.0
        - a
        - b
    )

    inside = (
        (a >= -0.001)
        & (b >= -0.001)
        & (c >= -0.001)
    )

    mask[
        min_y:max_y + 1,
        min_x:max_x + 1,
    ] |= inside


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

        self.mask_program = self.ctx.program(
            vertex_shader=CUBE_MASK_VERTEX_SHADER,
            fragment_shader=CUBE_MASK_FRAGMENT_SHADER,
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

        # Dynamic projected silhouette of the rotating cube.
        self.cube_mask_tex = self.ctx.texture(
            (
                SIM_W,
                SIM_H,
            ),
            components=1,
            dtype="f1",
        )

        self.cube_mask_tex.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR,
        )

        self.cube_mask_tex.repeat_x = False
        self.cube_mask_tex.repeat_y = False

        self.cube_mask_depth = self.ctx.depth_renderbuffer(
            (
                SIM_W,
                SIM_H,
            )
        )

        self.cube_mask_fbo = self.ctx.framebuffer(
            color_attachments=[
                self.cube_mask_tex
            ],
            depth_attachment=self.cube_mask_depth,
        )

        for program in (
            self.overlay_program,
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

        self.overlay_program[
            "cube_mask_tex"
        ].value = 5

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

        self.cube_mask_tex.use(
            location=5
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

        self.cube_positions = (
            cube_vertices
            .reshape(-1, 8)[:, 0:3]
            .copy()
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

        cube_positions = (
            cube_vertices
            .reshape(-1, 8)[:, 0:3]
            .copy()
        )

        self.mask_vbo = self.ctx.buffer(
            cube_positions.astype(
                np.float32
            ).tobytes()
        )

        self.mask_vao = self.ctx.vertex_array(
            self.mask_program,
            [
                (
                    self.mask_vbo,
                    "3f",
                    "in_pos",
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

        # v0.11:
        # upload the CURRENT projected cube obstacle as a 2D silhouette.
        # This drives one unified front-flame pass across the whole cube,
        # not separate flames on individual faces.
        cube_mask_u8 = np.ascontiguousarray(
            simulation.obstacle.astype(
                np.uint8
            ) * 255
        )

        self.cube_mask_tex.write(
            cube_mask_u8
        )

    def build_obstacle_mask(
        self,
    ) -> np.ndarray:
        """
        Project the actual rotating cube into the fire simulation grid.

        No GPU readback and no shader mask are needed. The exact projected
        3D silhouette becomes a boolean solid obstacle for the fluid.
        """

        _, mvp = self._matrices()

        positions = self.cube_positions

        ones = np.ones(
            (
                positions.shape[0],
                1,
            ),
            dtype=np.float32,
        )

        hom = np.concatenate(
            (
                positions,
                ones,
            ),
            axis=1,
        )

        clip = (
            mvp
            @ hom.T
        ).T

        w = clip[:, 3]

        safe_w = np.where(
            np.abs(w) < 1.0e-6,
            1.0e-6,
            w,
        )

        ndc = (
            clip[:, 0:3]
            / safe_w[:, None]
        )

        # NDC -> simulation coordinates.
        screen_x = (
            ndc[:, 0] * 0.5
            + 0.5
        ) * (
            SIM_W - 1
        )

        # Simulation row 0 is top.
        screen_y = (
            1.0
            - (
                ndc[:, 1] * 0.5
                + 0.5
            )
        ) * (
            SIM_H - 1
        )

        projected = np.stack(
            (
                screen_x,
                screen_y,
            ),
            axis=1,
        ).astype(
            np.float32
        )

        mask = np.zeros(
            (
                SIM_H,
                SIM_W,
            ),
            dtype=bool,
        )

        # create_cube_vertices() is triangle-expanded: every 3 vertices
        # already form one face triangle.
        for i in range(
            0,
            projected.shape[0],
            3,
        ):
            rasterize_triangle_into_mask(
                mask,
                projected[
                    i:i + 3
                ],
            )

        return mask

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
                0.66
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

    def _render_cube_mask(
        self,
    ) -> None:
        """
        Render the real rotating 3D cube silhouette into a small GPU mask.
        The 2D flame shader uses this exact projected shape as an obstacle.
        """

        self.cube_mask_fbo.use()

        self.ctx.viewport = (
            0,
            0,
            SIM_W,
            SIM_H,
        )

        self.cube_mask_fbo.clear(
            0.0,
            0.0,
            0.0,
            0.0,
            depth=1.0,
        )

        self.ctx.disable(
            moderngl.BLEND
        )

        self.ctx.enable(
            moderngl.DEPTH_TEST
            | moderngl.CULL_FACE
        )

        _, mvp = self._matrices()

        self._write_mat4(
            self.mask_program[
                "mvp"
            ],
            mvp,
        )

        self.mask_vao.render(
            mode=moderngl.TRIANGLES
        )

        self.ctx.screen.use()

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

        self.overlay_program[
            "cube_ignition"
        ].value = simulation.cube_ignition

        self.overlay_program[
            "simulation_time"
        ].value = simulation.time

        ys, xs = np.nonzero(
            simulation.obstacle
        )

        if xs.size:
            left = float(
                np.min(xs)
            ) / float(
                SIM_W - 1
            )

            right = float(
                np.max(xs)
            ) / float(
                SIM_W - 1
            )

            top = float(
                np.min(ys)
            ) / float(
                SIM_H - 1
            )

            bottom = float(
                np.max(ys)
            ) / float(
                SIM_H - 1
            )

        else:
            left = 0.34
            right = 0.66
            top = 0.28
            bottom = 0.75

        self.overlay_program[
            "cube_bounds"
        ].value = (
            left,
            right,
            top,
            bottom,
        )

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

        self.ctx.screen.use()

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

        # --------------------------------------------------------------
        # 1. Smoke background only.
        #
        # Fire is deliberately suppressed in pass 0. The one and only
        # visible flame is rendered later in pass 4, continuously from
        # the original source through the burning cube.
        # --------------------------------------------------------------
        self._render_overlay(
            simulation,
            volumetric_enabled,
            volumetric_strength,
            pass_mode=0,
        )

        if simulation.debug_mode != "normal":
            return

        # --------------------------------------------------------------
        # 2. Draw the real 3D cube over the fluid.
        #
        # Because the simulation contains no fluid inside the projected
        # obstacle, flames should naturally be visible only below, beside
        # and above the cube rather than through its center.
        # --------------------------------------------------------------
        self.ctx.disable(
            moderngl.BLEND
        )

        self.ctx.enable(
            moderngl.DEPTH_TEST
            | moderngl.CULL_FACE
        )

        model, mvp = self._matrices()

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

        if self.show_edges:
            self.ctx.disable(
                moderngl.CULL_FACE
            )

            _, edge_mvp = self._matrices(
                edge_scale=1.008
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
        # 3. THE ONE visible flame.
        #
        # This pass includes the unchanged original source below the cube.
        # As that same flame reaches the cube, its sampling coordinates
        # continuously widen to the cube's screen width. Nothing is added
        # on top of it; there is no second flame element.
        # --------------------------------------------------------------
        if simulation.show_fire:
            self._render_overlay(
                simulation,
                volumetric_enabled,
                volumetric_strength,
                pass_mode=4,
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
            renderer.update_rotation(
                dt
            )

            simulation.set_obstacle(
                renderer.build_obstacle_mask()
            )

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
        else:
            # Keep the obstacle aligned even after resizing while paused.
            simulation.set_obstacle(
                renderer.build_obstacle_mask()
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
            "Burning Cube v0.15 single-flame"
            f"  |  {clock.get_fps():5.1f} FPS"
            f"  |  div {simulation.divergence_rms:.4f}"
            f"  |  curl {simulation.curl_rms:.4f}"
            f"  |  temp {max_temp:.2f}"
            f"  |  fuel {max_fuel:.2f}"
            f"  |  ignition {simulation.cube_ignition * 100.0:4.0f}%"
            f"  |  surface {float(np.max(simulation.surface_burn)) * 100.0:3.0f}%"
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
