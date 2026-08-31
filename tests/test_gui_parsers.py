import numpy as np
import pytest

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.parsers import (
    parse_bounds,
    parse_float,
    parse_float_array,
    parse_fwhm_eta,
    parse_irf,
    parse_nonnegative_float,
    parse_positive_float,
    parse_positive_float_array,
)


def test_parse_float_accepts_scientific_notation():
    assert parse_float("1e-3", "value") == pytest.approx(1e-3)
    assert parse_float(" -2.5E+1 ", "value") == pytest.approx(-25.0)


@pytest.mark.parametrize("text", ["", "abc", "nan", "inf", "-inf"])
def test_parse_float_rejects_invalid_values(text):
    with pytest.raises(ValueError):
        parse_float(text, "value")


def test_parse_positive_float():
    assert parse_positive_float("1.5", "value") == pytest.approx(1.5)

    with pytest.raises(ValueError, match="positive"):
        parse_positive_float("0", "value")


def test_parse_nonnegative_float():
    assert parse_nonnegative_float("0", "value") == pytest.approx(0.0)

    with pytest.raises(ValueError, match="non-negative"):
        parse_nonnegative_float("-1", "value")


def test_parse_float_array():
    values = parse_float_array("1, 2.5, 1e-3", "values")

    np.testing.assert_allclose(values, np.array([1.0, 2.5, 1e-3]))


def test_parse_float_array_allow_empty():
    assert parse_float_array("", "values", allow_empty=True) is None


@pytest.mark.parametrize("text", ["1,,2", "1, abc", "nan, 1", "inf, 1"])
def test_parse_float_array_rejects_invalid_values(text):
    with pytest.raises(ValueError):
        parse_float_array(text, "values")


def test_parse_positive_float_array():
    values = parse_positive_float_array("1, 2, 3", "tau")

    np.testing.assert_allclose(values, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="positive"):
        parse_positive_float_array("1, 0, 3", "tau")


@pytest.mark.parametrize("text,expected", [("g", "g"), ("G", "g"), (" c ", "c"), ("PV", "pv")])
def test_parse_irf(text, expected):
    assert parse_irf(text) == expected


def test_parse_irf_rejects_invalid_value():
    with pytest.raises(ValueError, match="irf"):
        parse_irf("bad")


def test_parse_fwhm_eta_gaussian():
    fwhm, eta = parse_fwhm_eta("g", "0.12", "")

    assert fwhm == pytest.approx(0.12)
    assert eta is None


def test_parse_fwhm_eta_cauchy():
    fwhm, eta = parse_fwhm_eta("c", "", "0.34")

    assert fwhm == pytest.approx(0.34)
    assert eta is None


def test_parse_fwhm_eta_pseudo_voigt():
    fwhm, eta = parse_fwhm_eta("pv", "0.12", "0.34")

    assert np.isfinite(fwhm)
    assert np.isfinite(eta)


def test_parse_fwhm_eta_rejects_missing_pv_width():
    with pytest.raises(ValueError):
        parse_fwhm_eta("pv", "0.12", "")


def test_parse_bounds_broadcasts_scalar_bounds():
    bounds = parse_bounds("0", "10", np.array([1.0, 2.0, 3.0]), "tau")

    assert bounds == [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]


def test_parse_bounds_accepts_vector_bounds():
    bounds = parse_bounds(
        "0, 1, 2",
        "10, 11, 12",
        np.array([1.0, 2.0, 3.0]),
        "tau",
    )

    assert bounds == [(0.0, 10.0), (1.0, 11.0), (2.0, 12.0)]


def test_parse_bounds_rejects_wrong_length():
    with pytest.raises(ValueError, match="length"):
        parse_bounds("0, 1", "10, 11", np.array([1.0, 2.0, 3.0]), "tau")


def test_parse_bounds_rejects_lower_greater_than_upper():
    with pytest.raises(ValueError, match="<="):
        parse_bounds("2", "1", np.array([1.5]), "tau")