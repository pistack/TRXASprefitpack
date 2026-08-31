import numpy as np
import pytest

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.data_loader import (
    make_tscan_dataset_from_files,
    read_escan_dataset,
    read_tscan_trace,
)


def test_read_tscan_trace(tmp_path):
    path = tmp_path / "trace_a.dat"
    data = np.array(
        [
            [0.0, 1.0, 0.1],
            [1.0, 2.0, 0.1],
            [2.0, 3.0, 0.1],
        ]
    )
    np.savetxt(path, data)

    trace = read_tscan_trace(path)

    assert trace.name == "trace_a"
    assert trace.path == path
    np.testing.assert_allclose(trace.t, data[:, 0])
    np.testing.assert_allclose(trace.intensity, data[:, 1])
    np.testing.assert_allclose(trace.eps, data[:, 2])


def test_read_tscan_trace_with_custom_name(tmp_path):
    path = tmp_path / "trace_a.dat"
    data = np.array(
        [
            [0.0, 1.0, 0.1],
            [1.0, 2.0, 0.1],
        ]
    )
    np.savetxt(path, data)

    trace = read_tscan_trace(path, name="custom")

    assert trace.name == "custom"


def test_read_tscan_trace_ignores_extra_columns(tmp_path):
    path = tmp_path / "trace_extra.dat"
    data = np.array(
        [
            [0.0, 1.0, 0.1, 999.0],
            [1.0, 2.0, 0.1, 999.0],
        ]
    )
    np.savetxt(path, data)

    trace = read_tscan_trace(path)

    np.testing.assert_allclose(trace.t, data[:, 0])
    np.testing.assert_allclose(trace.intensity, data[:, 1])
    np.testing.assert_allclose(trace.eps, data[:, 2])


def test_read_tscan_trace_rejects_too_few_columns(tmp_path):
    path = tmp_path / "bad_trace.dat"
    data = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ]
    )
    np.savetxt(path, data)

    with pytest.raises(ValueError, match="at least three columns"):
        read_tscan_trace(path)


def test_make_tscan_dataset_from_files(tmp_path):
    path1 = tmp_path / "trace_1.dat"
    path2 = tmp_path / "trace_2.dat"

    data1 = np.array(
        [
            [0.0, 1.0, 0.1],
            [1.0, 2.0, 0.1],
            [2.0, 3.0, 0.1],
        ]
    )
    data2 = np.array(
        [
            [0.0, 4.0, 0.2],
            [1.0, 5.0, 0.2],
            [2.0, 6.0, 0.2],
        ]
    )

    np.savetxt(path1, data1)
    np.savetxt(path2, data2)

    dataset = make_tscan_dataset_from_files("dataset", [path1, path2])
    t, intensity, eps = dataset.to_arrays()

    assert dataset.name == "dataset"
    assert dataset.n_trace == 2
    np.testing.assert_allclose(t, data1[:, 0])
    np.testing.assert_allclose(intensity[:, 0], data1[:, 1])
    np.testing.assert_allclose(intensity[:, 1], data2[:, 1])
    np.testing.assert_allclose(eps[:, 0], data1[:, 2])
    np.testing.assert_allclose(eps[:, 1], data2[:, 2])


def test_make_tscan_dataset_from_files_rejects_empty_paths():
    with pytest.raises(ValueError, match="At least one"):
        make_tscan_dataset_from_files("dataset", [])


def test_make_tscan_dataset_from_files_rejects_different_time_axes(tmp_path):
    path1 = tmp_path / "trace_1.dat"
    path2 = tmp_path / "trace_2.dat"

    data1 = np.array(
        [
            [0.0, 1.0, 0.1],
            [1.0, 2.0, 0.1],
            [2.0, 3.0, 0.1],
        ]
    )
    data2 = np.array(
        [
            [0.0, 4.0, 0.2],
            [1.1, 5.0, 0.2],
            [2.0, 6.0, 0.2],
        ]
    )

    np.savetxt(path1, data1)
    np.savetxt(path2, data2)

    with pytest.raises(ValueError, match="same time axis"):
        make_tscan_dataset_from_files("dataset", [path1, path2])


def test_read_escan_dataset(tmp_path):
    intensity_path = tmp_path / "escan.dat"
    time_path = tmp_path / "time.dat"
    eps_path = tmp_path / "eps.dat"

    energy = np.array([100.0, 101.0, 102.0])
    time = np.array([0.0, 1.0])
    intensity = np.array(
        [
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ]
    )
    raw = np.column_stack([energy, intensity])
    eps = np.full_like(intensity, 0.1)

    np.savetxt(intensity_path, raw)
    np.savetxt(time_path, time)
    np.savetxt(eps_path, eps)

    dataset = read_escan_dataset(intensity_path, time_path, eps_path)

    assert dataset.name == "escan"
    assert dataset.intensity_path == intensity_path
    assert dataset.time_path == time_path
    assert dataset.eps_path == eps_path

    np.testing.assert_allclose(dataset.energy, energy)
    np.testing.assert_allclose(dataset.time, time)
    np.testing.assert_allclose(dataset.intensity, intensity)
    np.testing.assert_allclose(dataset.eps, eps)


def test_read_escan_dataset_with_custom_name(tmp_path):
    intensity_path = tmp_path / "escan.dat"
    time_path = tmp_path / "time.dat"
    eps_path = tmp_path / "eps.dat"

    raw = np.array(
        [
            [100.0, 1.0, 1.1],
            [101.0, 2.0, 2.1],
        ]
    )
    time = np.array([0.0, 1.0])
    eps = np.full((2, 2), 0.1)

    np.savetxt(intensity_path, raw)
    np.savetxt(time_path, time)
    np.savetxt(eps_path, eps)

    dataset = read_escan_dataset(
        intensity_path,
        time_path,
        eps_path,
        name="custom_escan",
    )

    assert dataset.name == "custom_escan"


def test_read_escan_dataset_rejects_bad_intensity_matrix(tmp_path):
    intensity_path = tmp_path / "bad_escan.dat"
    time_path = tmp_path / "time.dat"
    eps_path = tmp_path / "eps.dat"

    raw = np.array([100.0, 101.0, 102.0])
    time = np.array([0.0, 1.0])
    eps = np.ones((3, 2))

    np.savetxt(intensity_path, raw)
    np.savetxt(time_path, time)
    np.savetxt(eps_path, eps)

    with pytest.raises(ValueError, match="energy in the first column"):
        read_escan_dataset(intensity_path, time_path, eps_path)


def test_read_escan_dataset_rejects_shape_mismatch(tmp_path):
    intensity_path = tmp_path / "escan.dat"
    time_path = tmp_path / "time.dat"
    eps_path = tmp_path / "eps.dat"

    raw = np.array(
        [
            [100.0, 1.0, 1.1],
            [101.0, 2.0, 2.1],
            [102.0, 3.0, 3.1],
        ]
    )
    time = np.array([0.0, 1.0])
    eps = np.ones((3, 3))

    np.savetxt(intensity_path, raw)
    np.savetxt(time_path, time)
    np.savetxt(eps_path, eps)

    with pytest.raises(ValueError, match="same shape"):
        read_escan_dataset(intensity_path, time_path, eps_path)


def test_read_escan_dataset_rejects_nonpositive_eps(tmp_path):
    intensity_path = tmp_path / "escan.dat"
    time_path = tmp_path / "time.dat"
    eps_path = tmp_path / "eps.dat"

    raw = np.array(
        [
            [100.0, 1.0, 1.1],
            [101.0, 2.0, 2.1],
        ]
    )
    time = np.array([0.0, 1.0])
    eps = np.array(
        [
            [0.1, 0.1],
            [0.1, 0.0],
        ]
    )

    np.savetxt(intensity_path, raw)
    np.savetxt(time_path, time)
    np.savetxt(eps_path, eps)

    with pytest.raises(ValueError, match="positive"):
        read_escan_dataset(intensity_path, time_path, eps_path)