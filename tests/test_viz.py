import numpy as np
import pytest
from geom_nmf import GeoNMF
from geom_nmf import viz


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(42)
    X = rng.random((100, 10))
    y = rng.random(100)
    return GeoNMF().fit(X, y)


@pytest.fixture
def coords():
    rng = np.random.default_rng(42)
    return rng.uniform(low=[-180, -90], high=[180, 90], size=(100, 2))


class TestPlotComponents:
    def test_not_implemented(self, fitted_model):
        with pytest.raises(NotImplementedError):
            viz.plot_components(fitted_model)


class TestPlotReconstructionError:
    def test_not_implemented(self, fitted_model):
        with pytest.raises(NotImplementedError):
            viz.plot_reconstruction_error(fitted_model)


class TestPlotSpatialMap:
    def test_not_implemented(self, fitted_model, coords):
        with pytest.raises(NotImplementedError):
            viz.plot_spatial_map(fitted_model, coords)


class TestPlotPredictedVsActual:
    def test_not_implemented(self):
        y_true = np.random.default_rng(42).random(100)
        y_pred = np.random.default_rng(0).random(100)
        with pytest.raises(NotImplementedError):
            viz.plot_predicted_vs_actual(y_true, y_pred)
