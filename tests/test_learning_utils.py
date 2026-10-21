import numpy as np

from lunar_lander.learning.plot_q_learning import moving_average


def test_moving_average_returns_windowed_means() -> None:
    result = moving_average([1, 2, 3, 4], window=2)

    np.testing.assert_allclose(result, [1.5, 2.5, 3.5])


def test_moving_average_returns_values_when_series_is_shorter_than_window() -> None:
    result = moving_average([1, 2], window=3)

    np.testing.assert_allclose(result, [1, 2])
