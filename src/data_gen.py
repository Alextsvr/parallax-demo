import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


RANDOM_SEED = 42

N_SHOOTERS = 10
SESSIONS_PER_SHOOTER = 20
SERIES_PER_SESSION = 50
SHOTS_PER_SERIES = 5

OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "shooting_synthetic.csv")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def sample_weapon() -> str:
    """Sample weapon type."""
    return random.choices(
        population=["rifle", "carbine", "pistol"],
        weights=[0.4, 0.35, 0.25],  # slightly more rifle/carbine
        k=1,
    )[0]


def sample_distance_m(weapon: str) -> float:
    """Sample shooting distance in meters depending on weapon."""
    if weapon == "rifle":
        choices = [100, 200, 300, 400, 600, 800]
        weights = [0.1, 0.25, 0.3, 0.2, 0.1, 0.05]
    elif weapon == "carbine":
        choices = [50, 100, 150, 200, 300, 400]
        weights = [0.15, 0.25, 0.25, 0.2, 0.1, 0.05]
    else:  # pistol
        choices = [5, 10, 15, 25, 35, 50]
        weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]

    return float(random.choices(choices, weights=weights, k=1)[0])


def get_weapon_base_spread(weapon: str) -> float:
    """Base spread parameter per weapon (relative units)."""
    if weapon == "pistol":
        return 3.0
    if weapon == "carbine":
        return 2.0
    if weapon == "rifle":
        return 1.5
    return 2.5


def target_radius(distance_m: float, base_radius: float = 10.0) -> float:
    """
    Target radius in the same units as x, y.

    Grows with sqrt(distance) to reflect increasing target size / difficulty compensation.
    """
    return base_radius * math.sqrt(distance_m / 100.0)


def generate_shooter_params(n_shooters: int):
    """
    Generate per-shooter parameters:
    - skill: higher is better (less spread),
    - bias_x, bias_y: systematic offset of aiming point.
    """
    shooters = {}
    for shooter_id in range(1, n_shooters + 1):
        # Skill: 0.5 (weak) .. 1.5 (very strong)
        skill = np.random.uniform(0.5, 1.5)

        # Bias: systematic offset around center (e.g. always slightly low-left)
        bias_x = np.random.normal(loc=0.0, scale=5.0)
        bias_y = np.random.normal(loc=0.0, scale=5.0)

        shooters[shooter_id] = {"skill": skill, "bias_x": bias_x, "bias_y": bias_y}
    return shooters


def generate_data() -> pd.DataFrame:
    """Generate full synthetic shooting dataset."""
    set_seed()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shooters = generate_shooter_params(N_SHOOTERS)

    rows = []

    # Base date for timestamps
    base_date = datetime(2024, 1, 1, 9, 0, 0)

    for shooter_id in range(1, N_SHOOTERS + 1):
        shooter_params = shooters[shooter_id]
        skill = shooter_params["skill"]
        bias_x = shooter_params["bias_x"]
        bias_y = shooter_params["bias_y"]

        for session_id in range(1, SESSIONS_PER_SHOOTER + 1):
            # Each session starts at some day/time offset
            session_start = base_date + timedelta(
                days=int(np.random.uniform(0, 60)),
                hours=int(np.random.uniform(0, 8)),
            )

            time_since_start = 0.0  # seconds from session start

            for series_id in range(1, SERIES_PER_SESSION + 1):
                weapon = sample_weapon()
                distance_m = sample_distance_m(weapon)

                base_spread = get_weapon_base_spread(weapon)
                dist_factor = distance_m / 100.0
                sigma = base_spread * dist_factor / skill

                series_hits = 0
                series_shots = 0

                for shot_idx in range(1, SHOTS_PER_SERIES + 1):
                    # Time between shots
                    delta_t = np.random.uniform(1.0, 4.0)
                    time_since_start += delta_t

                    # Shot coordinates around bias
                    x = np.random.normal(loc=bias_x, scale=sigma)
                    y = np.random.normal(loc=bias_y, scale=sigma)

                    r = math.sqrt(x**2 + y**2)
                    radius = target_radius(distance_m)

                    hit = int(r <= radius)

                    series_hits += hit
                    series_shots += 1

                    timestamp = session_start + timedelta(seconds=time_since_start)

                    rows.append(
                        {
                            "shooter_id": shooter_id,
                            "session_id": session_id,
                            "series_id": series_id,
                            "shot_idx": shot_idx,
                            "x": x,
                            "y": y,
                            "distance_m": distance_m,
                            "weapon": weapon,
                            "time_since_start": time_since_start,
                            "hit": hit,
                            "timestamp": timestamp.isoformat(),
                        }
                    )

    df = pd.DataFrame(rows)

    # Basic sanity checks
    expected_rows = (
        N_SHOOTERS * SESSIONS_PER_SHOOTER * SERIES_PER_SESSION * SHOTS_PER_SERIES
    )
    assert len(df) == expected_rows, f"Unexpected number of rows: {len(df)}"

    return df


def main():
    df = generate_data()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved synthetic data to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()