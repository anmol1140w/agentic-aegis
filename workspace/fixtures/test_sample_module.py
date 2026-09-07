from sample_module import health_check


def test_health_check_is_local_and_healthy() -> None:
    assert health_check() == {"status": "ok", "scope": "local"}
