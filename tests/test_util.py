"""Tests for halvoice.util: the CA bundle helper."""

import ssl

from halvoice import util


def missing_verify_paths(tmp_path):
    return ssl.DefaultVerifyPaths(
        cafile=str(tmp_path / "none.pem"),
        capath=str(tmp_path / "none"),
        openssl_cafile_env="SSL_CERT_FILE",
        openssl_cafile=str(tmp_path / "none.pem"),
        openssl_capath_env="SSL_CERT_DIR",
        openssl_capath=str(tmp_path / "none"),
    )


def test_ca_bundle_picks_the_first_existing_system_bundle(monkeypatch, tmp_path):
    first, second = tmp_path / "a.crt", tmp_path / "b.crt"
    second.write_text("x")
    first.write_text("x")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: missing_verify_paths(tmp_path))
    monkeypatch.setattr(util, "CA_BUNDLES", (str(tmp_path / "gone.crt"), str(first), str(second)))
    assert util.ca_bundle() == str(first)
    assert util.os.environ["SSL_CERT_FILE"] == str(first)
    util.os.environ.pop("SSL_CERT_FILE")


def test_ca_bundle_leaves_the_environment_when_python_has_default_paths(monkeypatch, tmp_path):
    present = tmp_path / "ca.pem"
    present.write_text("x")
    paths = missing_verify_paths(tmp_path)._replace(cafile=str(present))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)
    assert util.ca_bundle() is None
    assert "SSL_CERT_FILE" not in util.os.environ


def test_ca_bundle_never_replaces_a_value_the_user_set(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", "/mine.pem")
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: missing_verify_paths(tmp_path))
    assert util.ca_bundle() is None
    assert util.os.environ["SSL_CERT_FILE"] == "/mine.pem"


def test_ca_bundle_returns_none_when_no_bundle_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: missing_verify_paths(tmp_path))
    monkeypatch.setattr(util, "CA_BUNDLES", (str(tmp_path / "gone.crt"),))
    assert util.ca_bundle() is None
