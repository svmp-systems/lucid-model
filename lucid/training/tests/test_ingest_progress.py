from lucid.training.ingest_progress import should_log_progress


def test_should_log_progress_every_article_when_interval_one() -> None:
    assert should_log_progress(1, 10, 1) is True
    assert should_log_progress(5, 10, 1) is True
    assert should_log_progress(10, 10, 1) is True


def test_should_log_progress_every_n_and_boundaries() -> None:
    assert should_log_progress(1, 100, 5) is True
    assert should_log_progress(2, 100, 5) is False
    assert should_log_progress(5, 100, 5) is True
    assert should_log_progress(10, 100, 5) is True
    assert should_log_progress(99, 100, 5) is False
    assert should_log_progress(100, 100, 5) is True
