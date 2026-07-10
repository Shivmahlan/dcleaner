import os
import numpy as np
import pandas as pd

# Build a small, deterministic messy dataset once for the test run.
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "sample.csv")

os.makedirs(os.path.dirname(DATA), exist_ok=True)
if not os.path.exists(DATA):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "city": rng.choice(["NY", "LA", "SF"], 60),
        "age": rng.integers(15, 60, 60),
        "salary": rng.integers(30000, 120000, 60).astype(float),
        "score": rng.normal(50, 10, 60),
    })
    df.loc[0:3, "salary"] = np.nan
    df.loc[5, "age"] = np.nan
    df.to_csv(DATA, index=False)
