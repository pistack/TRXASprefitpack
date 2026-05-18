import sys
import os
import warnings
import numpy as np
import pytest
from scipy.stats import f

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')
from TRXASprefitpack.driver.anal_fit import (
    CIResult,
    _clip_to_bound,
    _find_ci_bracket,
    _safe_parameter_step,
    ci_scan_opt_f,
    confidence_interval,
    is_better_fit,
    res_scan_opt,
)


def test_is_better_fit_returns_expected_p_value():
    result1 = {"chi2": 10.0, "n_param": 4, "num_pts": 100}
    result2 = {"chi2": 15.0, "n_param": 3, "num_pts": 100}

    p = is_better_fit(result1, result2)

    dfn = result1["n_param"] - result2["n_param"]
    dfd = result1["num_pts"] - result1["n_param"]
    f_stat = (result2["chi2"] - result1["chi2"]) / dfn / (result1["chi2"] / dfd)
    expected = 1.0 - f.cdf(f_stat, dfn, dfd)

    assert p == pytest.approx(expected)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize(
    "result1,result2",
    [
        (
            {"chi2": 10.0, "n_param": 3, "num_pts": 100},
            {"chi2": 12.0, "n_param": 3, "num_pts": 100},
        ),
        (
            {"chi2": 10.0, "n_param": 2, "num_pts": 100},
            {"chi2": 12.0, "n_param": 3, "num_pts": 100},
        ),
    ],
)
def test_is_better_fit_rejects_equal_or_fewer_parameters(result1, result2):
    with pytest.raises(ValueError, match="strictly greater"):
        is_better_fit(result1, result2)


def test_is_better_fit_rejects_different_num_pts():
    result1 = {"chi2": 10.0, "n_param": 4, "num_pts": 100}
    result2 = {"chi2": 12.0, "n_param": 3, "num_pts": 99}

    with pytest.raises(ValueError, match="number of data points"):
        is_better_fit(result1, result2)


def test_safe_parameter_step_uses_valid_eps():
    assert _safe_parameter_step(10.0, 0.2) == pytest.approx(0.2)


@pytest.mark.parametrize("bad_eps", [0.0, -1.0, np.nan, np.inf])
def test_safe_parameter_step_fallback_for_invalid_eps(bad_eps):
    step = _safe_parameter_step(10.0, bad_eps)

    assert np.isfinite(step)
    assert step > 0


def test_safe_parameter_step_fallback_scales_with_parameter_value():
    step_small = _safe_parameter_step(0.0, np.nan)
    step_large = _safe_parameter_step(100.0, np.nan)

    assert step_large > step_small


def test_clip_to_bound():
    assert _clip_to_bound(-1.0, 0.0, 2.0) == 0.0
    assert _clip_to_bound(3.0, 0.0, 2.0) == 2.0
    assert _clip_to_bound(1.0, 0.0, 2.0) == 1.0
    assert _clip_to_bound(1.0, None, None) == 1.0


def test_find_ci_bracket_returns_endpoint_for_upper_direction():
    def scan_func(x, *_):
        return x - 2.0

    endpoint = _find_ci_bracket(
        center=1.0,
        step=0.25,
        direction=1,
        scan_func=scan_func,
        fargs=(),
        lower_bound=None,
        upper_bound=None,
        max_expand=10,
    )

    assert endpoint is not None
    assert endpoint >= 2.0


def test_find_ci_bracket_returns_endpoint_for_lower_direction():
    def scan_func(x, *_):
        return -x - 2.0

    endpoint = _find_ci_bracket(
        center=-1.0,
        step=0.25,
        direction=-1,
        scan_func=scan_func,
        fargs=(),
        lower_bound=None,
        upper_bound=None,
        max_expand=10,
    )

    assert endpoint is not None
    assert endpoint <= -2.0


def test_find_ci_bracket_returns_none_when_upper_bound_reached():
    def scan_func(x, *_):
        return -1.0

    endpoint = _find_ci_bracket(
        center=1.0,
        step=0.25,
        direction=1,
        scan_func=scan_func,
        fargs=(),
        lower_bound=None,
        upper_bound=1.5,
        max_expand=10,
    )

    assert endpoint is None


def test_find_ci_bracket_returns_none_when_lower_bound_reached():
    def scan_func(x, *_):
        return -1.0

    endpoint = _find_ci_bracket(
        center=1.0,
        step=0.25,
        direction=-1,
        scan_func=scan_func,
        fargs=(),
        lower_bound=0.5,
        upper_bound=None,
        max_expand=10,
    )

    assert endpoint is None


def test_find_ci_bracket_returns_none_after_max_expand():
    def scan_func(x, *_):
        return -1.0

    endpoint = _find_ci_bracket(
        center=1.0,
        step=0.25,
        direction=1,
        scan_func=scan_func,
        fargs=(),
        lower_bound=None,
        upper_bound=None,
        max_expand=3,
    )

    assert endpoint is None


def test_find_ci_bracket_rejects_nonpositive_step():
    def scan_func(x, *_):
        return x

    with pytest.raises(ValueError, match="finite positive"):
        _find_ci_bracket(
            center=1.0,
            step=0.0,
            direction=1,
            scan_func=scan_func,
            fargs=(),
            lower_bound=None,
            upper_bound=None,
            max_expand=3,
        )


def test_find_ci_bracket_rejects_invalid_direction():
    def scan_func(x, *_):
        return x

    with pytest.raises(ValueError, match="direction"):
        _find_ci_bracket(
            center=1.0,
            step=0.1,
            direction=0,
            scan_func=scan_func,
            fargs=(),
            lower_bound=None,
            upper_bound=None,
            max_expand=3,
        )


def test_ci_result_attribute_access_and_dir():
    result = CIResult()
    result["method"] = "f"
    result["alpha"] = 0.05

    assert result.method == "f"
    assert result.alpha == 0.05
    assert "method" in dir(result)
    assert "alpha" in dir(result)


def test_ci_result_missing_attribute_raises_attribute_error():
    result = CIResult()

    with pytest.raises(AttributeError):
        _ = result.missing_key


def test_ci_result_str_skips_zero_intervals_and_reports_nan():
    result = CIResult()
    result["method"] = "f"
    result["alpha"] = 0.05
    result["param_name"] = np.array(["fixed", "failed", "tau_1"], dtype=object)
    result["x"] = np.array([1.0, 2.0, 3.0])
    result["ci"] = [(0, 0), (np.nan, np.nan), (-0.2, 0.3)]

    report = str(result)

    assert "[Report for Confidence Interval based on F-test]" in report
    assert "Method: f" in report
    assert "fixed" not in report
    assert "failed: confidence interval not found" in report
    assert "tau_1" in report


def test_ci_scan_opt_f_uses_half_chi2_scale(monkeypatch):
    import TRXASprefitpack.driver.anal_fit as anal_fit

    def fake_res_scan_opt(p, *args):
        return 7.0

    monkeypatch.setattr(anal_fit, "res_scan_opt", fake_res_scan_opt)

    # res_scan_opt = 7
    # chi2_opt / 2 = 5
    # denominator = chi2_opt / (2 * dfd) = 10 / 10 = 1
    # dfn = 1
    # F_alpha = 1.5
    # value = (7 - 5) / 1 / 1 - 1.5 = 0.5
    value = ci_scan_opt_f(
        0.0,
        1.5,   # F_alpha
        1,     # dfn
        5,     # dfd
        10.0,  # chi2_opt
        "remaining",
        "args",
    )

    assert value == pytest.approx(0.5)


def test_res_scan_opt_warns_when_minimize_fails(monkeypatch):
    import TRXASprefitpack.driver.anal_fit as anal_fit

    class FakeResult(dict):
        @property
        def message(self):
            return self["message"]

    def fake_minimize(*args, **kwargs):
        return FakeResult(success=False, message="failed", fun=123.0)

    def fake_func(x, *args):
        return 0.0, np.zeros_like(x)

    monkeypatch.setattr(anal_fit, "minimize", fake_minimize)

    # args layout expected by res_scan_opt:
    # args[0] = scan index
    # args[1] = parameter
    # args[2] = bounds
    # args[3] = objective function
    # args[-4] = fixed_param_idx
    with pytest.warns(RuntimeWarning, match="did not converge"):
        value = res_scan_opt(
            1.0,
            0,
            np.array([1.0]),
            [(0.0, 2.0)],
            fake_func,
            "extra1",
            "extra2",
            "extra3",
            np.array([False]),
        )

    assert value == pytest.approx(123.0)


def test_confidence_interval_rejects_unsupported_model():
    result = {
        "model": "unknown",
        "x": np.array([1.0]),
        "bounds": [(0.0, 2.0)],
        "n_param": 1,
        "num_pts": 10,
        "chi2": 1.0,
        "param_name": np.array(["p"]),
        "x_eps": np.array([0.1]),
    }

    with pytest.raises(ValueError, match="Unsupported model"):
        confidence_interval(result, alpha=0.05)