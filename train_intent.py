#!/usr/bin/env python3
"""Prototype intent classifier for two ROS 2 recorded trajectories.
Run in the ROS 2 Docker environment with pandas and scikit-learn installed.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path('/workspace/intent_csv')
OUTPUT = Path('/workspace/intent_model.pkl')
JOINTS = [f'joint{i}_deg' for i in range(1, 5)]
FEATURES = JOINTS + [f'delta_{j}' for j in JOINTS]


def make_features(path, label):
    df = pd.read_csv(path).sort_values('time_s').reset_index(drop=True)
    if not all(col in df.columns for col in ['time_s', *JOINTS]):
        raise ValueError(f'Missing columns in {path}')
    # Position at a given sample and displacement from ~0.3 sec earlier.
    prior_times = df['time_s'].to_numpy() - 0.3
    prior_idx = np.searchsorted(df['time_s'].to_numpy(), prior_times, side='right') - 1
    prior_idx = np.clip(prior_idx, 0, len(df) - 1)
    for joint in JOINTS:
        df[f'delta_{joint}'] = df[joint].to_numpy() - df[joint].to_numpy()[prior_idx]
    df['label'] = label
    # Drop first second because robot is initially idle; retain some stopped periods
    # for demonstration. This is NOT a validated classifier.
    df = df[df['time_s'] >= 1.0]
    return df


def main():
    pick = make_features(DATA_DIR / 'pick01.csv', 1)
    nonpick = make_features(DATA_DIR / 'nonpick01.csv', 0)
    dataset = pd.concat([pick, nonpick], ignore_index=True)
    x = dataset[FEATURES]
    y = dataset['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=5)
    model.fit(x, y)
    print(f'Rows: pick={len(pick)}, nonpick={len(nonpick)}')
    print('Training complete (demonstration only; no independent test runs).')
    for name, importance in sorted(zip(FEATURES, model.feature_importances_), key=lambda p: -p[1]):
        print(f'  {name}: {importance:.3f}')
    with OUTPUT.open('wb') as f:
        pickle.dump({'model': model, 'features': FEATURES}, f)
    print(f'Saved model: {OUTPUT}')
    print('Do not interpret training predictions as accuracy on new runs.')


if __name__ == '__main__':
    main()
