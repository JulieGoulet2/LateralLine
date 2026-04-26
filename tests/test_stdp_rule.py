import pathlib


def _ll_mon_stdp_block() -> str:
    """Extract LL->MON STDP source block from the simulation file."""
    src = (pathlib.Path(__file__).parent.parent / "ll_stdp_brian2.py").read_text()
    start = src.index("# LL->MON with STDP")
    end = src.index("# MON -> TS:", start)
    return src[start:end]


def test_ll_mon_on_pre_is_multiplicative_depression():
    """on_pre must depress proportional to current weight (apost*w), not additively."""
    block = _ll_mon_stdp_block()
    assert "apost*w" in block, (
        "REGRESSION: LL->MON on_pre uses additive LTD. "
        "Must be: w = clip(w + apost*w, 0*mV, wmax)"
    )


def test_ll_mon_on_post_is_multiplicative_potentiation():
    """on_post must potentiate proportional to distance from wmax (apre*(wmax-w))."""
    block = _ll_mon_stdp_block()
    assert "apre*(wmax - w)" in block, (
        "REGRESSION: LL->MON on_post uses additive LTP. "
        "Must be: w = clip(w + apre*(wmax - w), 0*mV, wmax)"
    )


def test_additive_pattern_not_in_ll_mon_block():
    """Guard: bare additive pattern must not appear in LL->MON block."""
    block = _ll_mon_stdp_block()
    assert "w + apost)" not in block and "w + apost," not in block, (
        "REGRESSION: Found additive LTD pattern in LL->MON STDP block."
    )
