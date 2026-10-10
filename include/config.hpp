#pragma once

namespace Config {
    constexpr double DENSITY = 4430.0;
    constexpr double SPECIFIC_HEAT = 553.0;
    constexpr double THERMAL_CONDUCTIVITY = 6.7;
    constexpr double ALPHA = THERMAL_CONDUCTIVITY / (DENSITY * SPECIFIC_HEAT);

    constexpr double DX = 10e-6;
    constexpr double DY = 10e-6;
    constexpr double DZ = 10e-6;

    constexpr int NX_GLOBAL = 200;
    constexpr int NY_GLOBAL = 200;
    constexpr int NZ_GLOBAL = 50;

    constexpr double DT = 1e-6;

    constexpr double DIFFUSION_SUM =
        ALPHA * DT *
        (1.0 / (DX * DX) +
        1.0 / (DY * DY) +
        1.0 / (DZ * DZ));

    static_assert(
        DIFFUSION_SUM <= 0.5,
        "Unstable explicit diffusion timestep"
    );

    constexpr int TOTAL_STEPS = 1000;

    constexpr double LASER_POWER = 200.0;

    // Demonstration values; calibrate for the actual material and process.
    constexpr double ABSORPTIVITY = 0.35;
    constexpr double LASER_RADIUS = 50e-6; // Gaussian spot radius in metres
    constexpr double SCAN_SPEED = 0.8;    // Metres per second

    static_assert(
        LASER_RADIUS > 0.0 &&
        ABSORPTIVITY >= 0.0 &&
        ABSORPTIVITY <= 1.0,
        "Invalid laser parameters"
    );
    
    constexpr double AMBIENT_TEMP = 293.15;
}
