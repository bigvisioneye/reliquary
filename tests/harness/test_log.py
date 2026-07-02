from harness.log import configure_harness_logging, get_logger


def test_configure_harness_logging_and_get_logger(capsys) -> None:
    configure_harness_logging("INFO")
    logger = get_logger("test")
    logger.info("progress ping")
    out = capsys.readouterr().out
    assert "harness.test" in out
    assert "progress ping" in out
