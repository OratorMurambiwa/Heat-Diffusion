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

    constexpr double DT = 1e-5;
    constexpr int TOTAL_STEPS = 1000;

    constexpr double LASER_POWER = 200.0;
    constexpr double AMBIENT_TEMP = 293.15;
}
