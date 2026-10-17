from marshmallow import utils


def parse_when(value):
    return utils.from_iso_datetime(value)
