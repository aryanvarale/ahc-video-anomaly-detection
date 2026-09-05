"""
CPU-only checks for the stage-2 tiled verification logic.

Runs without a GPU or a live model server: tile geometry is checked against real
frames from a test video, and the verification flow is exercised with a stub
classifier so the plumbing (selection, budget, timing accounting) can be tested
independently of the model.

  python test_tiling.py
"""
import numpy as np

import vad_pipeline as vp

VIDEO = "data/Train and Test/test/videos/T031.mp4"


def test_tile_geometry_covers_every_pixel():
    """Tiles must partition the frame exactly - no gaps, no overlaps, nothing
    dropped on the remainder row/column of an odd-sized frame."""
    for h, w in [(1080, 1920), (721, 1281), (100, 100), (3, 5)]:
        frame = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        for grid in (2, 3):
            tiles = vp.tile_frame(frame, grid)
            area = sum(t.shape[0] * t.shape[1] for _, t in tiles)
            assert area == h * w, f"{h}x{w} grid={grid}: covered {area} of {h*w}"
            canvas = np.zeros((h, w), dtype=np.int32)
            th, tw = h // grid, w // grid
            if th >= 1 and tw >= 1:
                for (r, c), t in tiles:
                    y0, x0 = r * th, c * tw
                    canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] += 1
                assert canvas.min() == 1 and canvas.max() == 1, \
                    f"{h}x{w} grid={grid}: overlapping or uncovered pixels"
    print("tile geometry: partitions exactly, no gaps/overlaps  OK")


def test_tiles_gain_pixel_density():
    """The whole point: a tile encoded at long_side carries more detail for its
    region than that region does inside the downscaled full frame."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    grid = 2
    full_h, full_w = 1080, 1920
    s = vp.CFG["long_side"] / max(full_h, full_w)
    region_px_in_full = (full_h * s / grid) * (full_w * s / grid)

    _, tile = vp.tile_frame(frame, grid)[0]
    th, tw = tile.shape[:2]
    ts = vp.CFG["long_side"] / max(th, tw)
    region_px_in_tile = (th * min(ts, 1.0)) * (tw * min(ts, 1.0))

    gain = region_px_in_tile / region_px_in_full
    assert gain > 3.0, f"expected ~grid^2 density gain, got {gain:.2f}x"
    print(f"pixel density for a corner region: {gain:.1f}x more in tile  OK")


def test_verify_selection_rules():
    lo, hi = vp.CFG["verify_band"]
    n = vp.CFG["verify_every_n"]
    mid = (lo + hi) / 2

    assert vp.wants_verify(mid, 1), "uncertain chunk should be verified"
    assert not vp.wants_verify(0.001, 1), "confident-normal off-cycle chunk should be skipped"
    assert not vp.wants_verify(0.999, 1), "confident-anomaly off-cycle chunk should be skipped"
    assert vp.wants_verify(0.001, 0), "periodic sweep must still sample confident-normal chunks"
    assert vp.wants_verify(0.001, n), "periodic sweep must fire every Nth chunk"
    print("verification selection: uncertain band + periodic sweep  OK")


def test_verification_flow_with_stub_model():
    """End-to-end plumbing with a fake classifier: a planted 'anomaly' in one
    tile position should be picked up, timings recorded, budget respected."""
    calls = {"n": 0}

    def stub_classify(imgs):
        calls["n"] += 1
        # every 4th call (one tile position) looks strongly anomalous
        if calls["n"] % 4 == 0:
            return {"normal": 0.02, "road_spill_or_debris": 0.98}
        return {"normal": 0.97, "road_spill_or_debris": 0.03}

    best, times = vp.verify_chunk(VIDEO, 10.0, 12.0, 4, classify_fn=stub_classify)
    assert best is not None, "verify_chunk returned nothing - could not read frames?"
    assert 1.0 - best.get("normal", 0.0) > 0.9, f"strongest tile not selected: {best}"
    assert len(times) == calls["n"] == vp.CFG["verify_grid"] ** 2, \
        f"expected one call per tile position, got {len(times)} calls"
    print(f"verify_chunk: {len(times)} tile calls, picked the anomalous tile  OK")


def test_budget_cap_is_enforced():
    chunks = [(float(i), float(i) + 2.0, None) for i in range(200)]
    probs = [{"normal": 0.5} for _ in chunks]        # all uncertain -> all want verifying
    old_enabled, old_cap = vp.CFG["verify_enabled"], vp.CFG["verify_max_calls"]
    vp.CFG["verify_max_calls"] = 8
    try:
        _, times = vp.apply_verification(VIDEO, chunks, probs, level=2)
        assert len(times) <= 8 + vp.CFG["verify_grid"] ** 2, \
            f"budget blown: {len(times)} calls with cap {vp.CFG['verify_max_calls']}"
        print(f"cost cap: {len(times)} calls under cap {vp.CFG['verify_max_calls']}  OK")
    finally:
        vp.CFG["verify_enabled"], vp.CFG["verify_max_calls"] = old_enabled, old_cap


def test_disabled_by_default():
    assert vp.CFG["verify_enabled"] is False, \
        "verification must stay off by default so the submitted path is unchanged"
    print("default config: verification off, existing behaviour untouched  OK")


if __name__ == "__main__":
    test_disabled_by_default()
    test_tile_geometry_covers_every_pixel()
    test_tiles_gain_pixel_density()
    test_verify_selection_rules()
    test_verification_flow_with_stub_model()
    test_budget_cap_is_enforced()
    print("\nall stage-2 checks passed (no GPU / no model server needed)")
