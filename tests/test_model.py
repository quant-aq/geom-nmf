import numpy as np
import pytest
from geom_nmf import GeoNMF


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = rng.random((100, 10))
    y = rng.random(100)
    return X, y


class TestGeoNMFInit:
    def test_default_params(self):
        model = GeoNMF()
        assert model.n_components == 10
        assert model.max_iter == 200
        assert model.tol == 1e-4
        assert model.random_state is None

    def test_custom_params(self):
        model = GeoNMF(n_components=5, max_iter=100, tol=1e-3, random_state=0)
        assert model.n_components == 5
        assert model.max_iter == 100
        assert model.tol == 1e-3
        assert model.random_state == 0

    def test_get_params(self):
        model = GeoNMF(n_components=5)
        params = model.get_params()
        assert params["n_components"] == 5

    def test_set_params(self):
        model = GeoNMF()
        model.set_params(n_components=3)
        assert model.n_components == 3


class TestGeoNMFFit:
    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        model = GeoNMF()
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_n_features_in(self, sample_data):
        X, y = sample_data
        model = GeoNMF().fit(X, y)
        assert model.n_features_in_ == X.shape[1]

    def test_fit_sets_is_fitted(self, sample_data):
        X, y = sample_data
        model = GeoNMF().fit(X, y)
        assert model.is_fitted_

    def test_fit_invalid_input(self):
        model = GeoNMF()
        with pytest.raises(ValueError):
            model.fit([[1, 2], [3, 4]], [1])  # mismatched samples


class TestGeoNMFPredict:
    def test_predict_raises_before_fit(self, sample_data):
        X, _ = sample_data
        model = GeoNMF()
        with pytest.raises(Exception):
            model.predict(X)

    def test_predict_not_implemented(self, sample_data):
        X, y = sample_data
        model = GeoNMF().fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict(X)
