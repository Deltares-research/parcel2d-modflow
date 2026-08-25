from sqlalchemy import create_engine

from parcel2d_modflow.io import database


def test_load_from_connection_uses_sqlalchemy_connection(monkeypatch):
    engine = create_engine("sqlite://")
    monkeypatch.setattr(database, "make_connection", lambda: engine)

    dataframe = database.load_from_connection(select_query="1 AS value")

    assert dataframe.to_dict("list") == {"value": [1]}
