import sys

import pytest

from parcel2d_modflow import logging as logging_module

CUSTOM_SINK = object()


@pytest.mark.parametrize(
    ("sink", "kwargs", "expected_sink", "expected_kwargs"),
    [
        (None, {}, sys.stdout, {}),
        (
            CUSTOM_SINK,
            {"level": "DEBUG", "format": "{message}"},
            CUSTOM_SINK,
            {"level": "DEBUG", "format": "{message}"},
        ),
    ],
    ids=["default-stdout", "custom-sink-with-kwargs"],
)
def test_init_logger_behaviour(
    monkeypatch, sink, kwargs, expected_sink, expected_kwargs
):
    calls = []

    monkeypatch.setattr(
        logging_module.logger, "remove", lambda: calls.append(("remove",))
    )
    monkeypatch.setattr(
        logging_module.logger,
        "add",
        lambda sink, **kwargs: calls.append(("add", sink, kwargs)),
    )

    if sink is None:
        logging_module.init_logger(**kwargs)
    else:
        logging_module.init_logger(sink=sink, **kwargs)

    assert calls == [
        ("remove",),
        ("add", expected_sink, expected_kwargs),
    ]
