"""Tests for event_wraping module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from vix_xai.event_wraping import (
    dtw_from_cost_matrix, dtdw_1d, wdtdw_1d,
    dtdw_embedding, wdtdw_embedding,
    wasserstein_1d, energy_distance_1d, mmd_rbf,
    apply_event_weighting,
)

def test_wasserstein_same():
    a = np.array([1.0, 2.0, 3.0])
    assert wasserstein_1d(a, a) < 1e-10

def test_wasserstein_different():
    assert wasserstein_1d(np.zeros(3), np.ones(3)) > 0

def test_energy_distance():
    assert energy_distance_1d(np.zeros(2), np.array([10.,10.])) > 0
    assert energy_distance_1d(np.zeros(2), np.zeros(2)) < 1e-10

def test_mmd_rbf():
    assert mmd_rbf(np.array([[0.],[0.1]]), np.array([[10.],[10.1]])) > 0

def test_dtw_identity():
    a = np.array([1.,2.,3.,4.,5.])
    res = dtdw_1d(a, a, k=1, band=3)
    assert res.normalized_cost < 1e-6

def test_dtw_different():
    res = dtdw_1d(np.array([1.,2.,3.,4.,5.]), np.array([5.,4.,3.,2.,1.]), k=1, band=3)
    assert res.normalized_cost > 0

def test_wdtdw():
    a = np.array([1.,2.,3.,4.,5.])
    g = np.array([0.1,0.2,0.5,0.8,1.0])
    res = wdtdw_1d(a, a+0.5, g, g, k=1, band=3, alpha=2.0)
    assert res.normalized_cost > 0

def test_embedding_dtw():
    res = dtdw_embedding(np.random.randn(10,4), np.random.randn(10,4), method="l2", k=0, band=5)
    assert res.normalized_cost > 0

def test_event_weighting():
    C = np.ones((5,5))
    g = np.array([0.,0.,0.5,1.,1.])
    Cw = apply_event_weighting(C, g, g, alpha=2.0, mode="local")
    assert Cw[0,0] < Cw[4,4]

if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn(); print(f"  PASS {name}"); passed += 1
            except Exception as e: print(f"  FAIL {name}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
