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
    // No wrapped ghost flames at edges.
    if (uv.x <= texel_size.x || uv.x >= 1.0 - texel_size.x ||
        uv.y <= texel_size.y || uv.y >= 1.0 - texel_size.y) return 0.0;
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

    // Larger fire: a soft third ring extends the burning silhouette.
    vec2 t3 = t * 3.0;
    m = max(m, flame_at(uv + vec2( t3.x, 0.0)) * 0.84);
    m = max(m, flame_at(uv + vec2(-t3.x, 0.0)) * 0.84);
    m = max(m, flame_at(uv + vec2(0.0,  t3.y)) * 0.84);
    m = max(m, flame_at(uv + vec2(0.0, -t3.y)) * 0.84);
    m = max(m, flame_at(uv + vec2( t3.x,  t3.y)) * 0.72);
    m = max(m, flame_at(uv + vec2(-t3.x,  t3.y)) * 0.72);
    m = max(m, flame_at(uv + vec2( t3.x, -t3.y)) * 0.72);
    m = max(m, flame_at(uv + vec2(-t3.x, -t3.y)) * 0.72);

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
            flame_base * 0.23
            + flame_expanded * 0.77
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
