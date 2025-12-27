import pytest
import deepchem as dc
import numpy as np
import tempfile

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_construction():
    from deepchem.models.torch_models import DeepONet
    model = DeepONet(branch_input_dim=10, trunk_input_dim=1)
    assert model is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_forward():
    from deepchem.models.torch_models import DeepONet
    model = DeepONet(branch_input_dim=10, trunk_input_dim=1)
    branch_input = torch.randn(5, 10)
    trunk_input = torch.randn(5, 1)
    output = model([branch_input, trunk_input])
    assert output.shape == (5, 1)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_model_construction():
    from deepchem.models.torch_models import DeepONetModel
    model = DeepONetModel(branch_input_dim=10, trunk_input_dim=1, batch_size=32)
    assert model is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_model_fit():
    from deepchem.models.torch_models import DeepONetModel
    X = np.random.randn(100, 11).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    model = DeepONetModel(branch_input_dim=10, trunk_input_dim=1, batch_size=32)
    model.fit(dataset, nb_epoch=1)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_model_predict():
    from deepchem.models.torch_models import DeepONetModel
    X = np.random.randn(100, 11).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    model = DeepONetModel(branch_input_dim=10, trunk_input_dim=1, batch_size=32)
    model.fit(dataset, nb_epoch=1)
    predictions = model.predict_on_batch(X)
    assert predictions.shape == (100, 1)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_model_overfit():
    from deepchem.models.torch_models import DeepONetModel
    np.random.seed(42)
    X = np.random.randn(10, 11).astype(np.float32)
    y = np.random.randn(10, 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    model = DeepONetModel(branch_input_dim=10,
                          trunk_input_dim=1,
                          branch_hidden=(64, 64),
                          trunk_hidden=(64, 64),
                          batch_size=10,
                          learning_rate=1e-3)
    model.fit(dataset, nb_epoch=500)
    predictions = model.predict_on_batch(X)
    mse = np.mean((predictions - y)**2)
    assert mse < 0.1


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_model_restore():
    from deepchem.models.torch_models import DeepONetModel
    X = np.random.randn(50, 11).astype(np.float32)
    y = np.random.randn(50, 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    model_dir = tempfile.mkdtemp()
    model = DeepONetModel(branch_input_dim=10,
                          trunk_input_dim=1,
                          batch_size=32,
                          model_dir=model_dir)
    model.fit(dataset, nb_epoch=10)
    pred_before = model.predict_on_batch(X)
    restored_model = DeepONetModel(branch_input_dim=10,
                                   trunk_input_dim=1,
                                   batch_size=32,
                                   model_dir=model_dir)
    restored_model.restore()
    pred_after = restored_model.predict_on_batch(X)
    assert np.allclose(pred_before, pred_after, atol=1e-5)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_different_hidden_dims():
    from deepchem.models.torch_models import DeepONetModel
    X = np.random.randn(50, 11).astype(np.float32)
    y = np.random.randn(50, 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    for hidden in [(32,), (64, 64), (128, 128, 128)]:
        model = DeepONetModel(branch_input_dim=10,
                              trunk_input_dim=1,
                              branch_hidden=hidden,
                              trunk_hidden=hidden,
                              batch_size=32)
        model.fit(dataset, nb_epoch=1)
        predictions = model.predict_on_batch(X)
        assert predictions.shape == (50, 1)
