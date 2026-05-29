import numpy as np
import pytest
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.models import (
    EScanDataset,
    TScanDataset,
    TScanTrace,
    tscan_datasets_to_driver_inputs,
)


def make_trace(name="trace", t=None, intensity=None, eps=None):
    if t is None:
        t = np.array([0.0, 1.0, 2.0])
    if intensity is None:
        intensity = np.array([1.0, 2.0, 3.0])
    if eps is None:
        eps = np.array([0.1, 0.1, 0.1])

    return TScanTrace(
        path=f"{name}.dat",
        name=name,
        t=t,
        intensity=intensity,
        eps=eps,
    )


def test_tscan_trace_accepts_valid_1d_arrays():
    trace = make_trace()

    assert trace.path.name == "trace.dat"
    assert trace.name == "trace"
    assert trace.n_time == 3
    assert trace.t.shape == (3,)
    assert trace.intensity.shape == (3,)
    assert trace.eps.shape == (3,)


def test_tscan_trace_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        TScanTrace(
            path="trace.dat",
            name="",
            t=np.array([0.0, 1.0]),
            intensity=np.array([1.0, 2.0]),
            eps=np.array([0.1, 0.1]),
        )


def test_tscan_trace_rejects_non_1d_time():
    with pytest.raises(ValueError, match="t must be a 1D"):
        make_trace(t=np.array([[0.0, 1.0, 2.0]]))


def test_tscan_trace_rejects_non_1d_intensity():
    with pytest.raises(ValueError, match="intensity must be a 1D"):
        make_trace(intensity=np.array([[1.0, 2.0, 3.0]]))


def test_tscan_trace_rejects_non_1d_eps():
    with pytest.raises(ValueError, match="eps must be a 1D"):
        make_trace(eps=np.array([[0.1, 0.1, 0.1]]))


def test_tscan_trace_rejects_mismatched_intensity_eps_shape():
    with pytest.raises(ValueError, match="same shape"):
        make_trace(
            intensity=np.array([1.0, 2.0, 3.0]),
            eps=np.array([0.1, 0.1]),
        )


def test_tscan_trace_rejects_mismatched_time_length():
    with pytest.raises(ValueError, match="match t length"):
        make_trace(
            t=np.array([0.0, 1.0]),
            intensity=np.array([1.0, 2.0, 3.0]),
            eps=np.array([0.1, 0.1, 0.1]),
        )


def test_tscan_trace_rejects_nonpositive_eps():
    with pytest.raises(ValueError, match="positive"):
        make_trace(eps=np.array([0.1, 0.0, 0.1]))


def test_tscan_dataset_accepts_traces_with_shared_time_axis():
    t = np.array([0.0, 1.0, 2.0])
    trace1 = make_trace("a", t=t, intensity=np.array([1.0, 2.0, 3.0]))
    trace2 = make_trace("b", t=t, intensity=np.array([4.0, 5.0, 6.0]))

    dataset = TScanDataset(name="dataset", traces=(trace1, trace2))

    assert dataset.name == "dataset"
    assert dataset.n_time == 3
    assert dataset.n_trace == 2
    assert dataset.trace_names == ("a", "b")
    assert dataset.time_range == (0.0, 2.0)


def test_tscan_dataset_rejects_empty_name():
    trace = make_trace()

    with pytest.raises(ValueError, match="name"):
        TScanDataset(name="", traces=(trace,))


def test_tscan_dataset_rejects_empty_trace_list():
    with pytest.raises(ValueError, match="at least one trace"):
        TScanDataset(name="dataset", traces=())


def test_tscan_dataset_rejects_different_time_axis_shape():
    trace1 = make_trace("a", t=np.array([0.0, 1.0, 2.0]))
    trace2 = make_trace("b", t=np.array([0.0, 1.0]),
                        intensity=np.array([1.0, 2.0]),
                        eps=np.array([0.1, 0.1]))

    with pytest.raises(ValueError, match="same time axis"):
        TScanDataset(name="dataset", traces=(trace1, trace2))


def test_tscan_dataset_rejects_different_time_axis_values():
    trace1 = make_trace("a", t=np.array([0.0, 1.0, 2.0]))
    trace2 = make_trace("b", t=np.array([0.0, 1.1, 2.0]))

    with pytest.raises(ValueError, match="same time axis"):
        TScanDataset(name="dataset", traces=(trace1, trace2))


def test_tscan_dataset_to_arrays():
    t = np.array([0.0, 1.0, 2.0])
    trace1 = make_trace(
        "a",
        t=t,
        intensity=np.array([1.0, 2.0, 3.0]),
        eps=np.array([0.1, 0.1, 0.1]),
    )
    trace2 = make_trace(
        "b",
        t=t,
        intensity=np.array([4.0, 5.0, 6.0]),
        eps=np.array([0.2, 0.2, 0.2]),
    )

    dataset = TScanDataset(name="dataset", traces=(trace1, trace2))
    t_out, intensity, eps = dataset.to_arrays()

    np.testing.assert_allclose(t_out, t)
    assert intensity.shape == (3, 2)
    assert eps.shape == (3, 2)
    np.testing.assert_allclose(intensity[:, 0], trace1.intensity)
    np.testing.assert_allclose(intensity[:, 1], trace2.intensity)
    np.testing.assert_allclose(eps[:, 0], trace1.eps)
    np.testing.assert_allclose(eps[:, 1], trace2.eps)


def test_tscan_datasets_to_driver_inputs():
    dataset1 = TScanDataset(
        name="dset1",
        traces=(make_trace("a"),),
    )

    dataset2 = TScanDataset(
        name="dset2",
        traces=(
            make_trace("b", intensity=np.array([4.0, 5.0, 6.0])),
            make_trace("c", intensity=np.array([7.0, 8.0, 9.0])),
        ),
    )

    t, intensity, eps, names = tscan_datasets_to_driver_inputs(
        [dataset1, dataset2]
    )

    assert len(t) == 2
    assert len(intensity) == 2
    assert len(eps) == 2
    assert list(names) == ["dset1", "dset2"]

    assert intensity[0].shape == (3, 1)
    assert intensity[1].shape == (3, 2)


def test_tscan_datasets_to_driver_inputs_rejects_empty_list():
    with pytest.raises(ValueError, match="At least one"):
        tscan_datasets_to_driver_inputs([])


def make_escan_dataset(
    name="escan",
    energy=None,
    time=None,
    intensity=None,
    eps=None,
):
    if energy is None:
        energy = np.array([100.0, 101.0, 102.0])
    if time is None:
        time = np.array([0.0, 1.0])
    if intensity is None:
        intensity = np.array(
            [
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
            ]
        )
    if eps is None:
        eps = np.full_like(intensity, 0.1)

    return EScanDataset(
        name=name,
        energy=energy,
        time=time,
        intensity=intensity,
        eps=eps,
        intensity_path="intensity.dat",
        eps_path="eps.dat",
        time_path="time.dat",
    )


def test_escan_dataset_accepts_valid_arrays():
    dataset = make_escan_dataset()

    assert dataset.name == "escan"
    assert dataset.n_energy == 3
    assert dataset.n_time == 2
    assert dataset.energy_range == (100.0, 102.0)
    assert dataset.time_range == (0.0, 1.0)
    assert dataset.intensity.shape == (3, 2)
    assert dataset.eps.shape == (3, 2)
    assert dataset.intensity_path.name == "intensity.dat"
    assert dataset.eps_path.name == "eps.dat"
    assert dataset.time_path.name == "time.dat"


def test_escan_dataset_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        make_escan_dataset(name="")


def test_escan_dataset_rejects_non_1d_energy():
    with pytest.raises(ValueError, match="energy must be a 1D"):
        make_escan_dataset(energy=np.array([[100.0, 101.0, 102.0]]))


def test_escan_dataset_rejects_non_1d_time():
    with pytest.raises(ValueError, match="time must be a 1D"):
        make_escan_dataset(time=np.array([[0.0, 1.0]]))


def test_escan_dataset_rejects_non_2d_intensity():
    with pytest.raises(ValueError, match="intensity must be a 2D"):
        make_escan_dataset(intensity=np.array([1.0, 2.0, 3.0]))


def test_escan_dataset_rejects_non_2d_eps():
    with pytest.raises(ValueError, match="eps must be a 2D"):
        make_escan_dataset(eps=np.array([0.1, 0.1, 0.1]))


def test_escan_dataset_rejects_mismatched_intensity_eps_shape():
    with pytest.raises(ValueError, match="same shape"):
        make_escan_dataset(eps=np.ones((3, 3)))


def test_escan_dataset_rejects_wrong_matrix_shape():
    with pytest.raises(ValueError, match=r"\(n_energy, n_time\)"):
        make_escan_dataset(intensity=np.ones((2, 3)), eps=np.ones((2, 3)))


def test_escan_dataset_rejects_nonpositive_eps():
    eps = np.array(
        [
            [0.1, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
        ]
    )

    with pytest.raises(ValueError, match="positive"):
        make_escan_dataset(eps=eps)