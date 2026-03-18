import builtins
import importlib
import pathlib

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path


def test_download_falls_back_to_plain_tqdm_when_tqdm_loggable_missing(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm_loggable.auto" or name == "tqdm_loggable" or name.startswith("tqdm_loggable."):
            raise ModuleNotFoundError("No module named 'tqdm_loggable'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    reloaded = importlib.reload(download)
    assert reloaded.tqdm.__name__ == "tqdm.auto"
    importlib.reload(reloaded)


def test_download_does_not_mask_other_missing_dependency(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm_loggable.auto":
            exc = ModuleNotFoundError("No module named 'other_dep'")
            exc.name = "other_dep"
            raise exc
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ModuleNotFoundError, match="other_dep"):
        importlib.reload(download)
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(download)
