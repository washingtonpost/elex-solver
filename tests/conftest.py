import json
import logging
import os
import sys

import pandas as pd
import pytest

_TEST_FOLDER = os.path.dirname(__file__)
FIXTURE_DIR = os.path.join(_TEST_FOLDER, "fixtures")


@pytest.fixture(autouse=True, scope="session")
def setup_logging():
    LOG = logging.getLogger("elexsolver")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(handler)


@pytest.fixture(scope="session")
def get_fixture():
    def _get_fixture(filename, load=False, csv=True):
        fileobj = open(os.path.join(FIXTURE_DIR, filename))
        if load:
            return json.load(fileobj)
        elif csv:
            return pd.read_csv(os.path.join(FIXTURE_DIR, filename))
        return fileobj

    return _get_fixture


@pytest.fixture(scope="session")
def random_data_no_weights(get_fixture):
    return get_fixture("random_data_n100_p5_17554.csv")


@pytest.fixture(scope="session")
def random_data_weights(get_fixture):
    return get_fixture("random_data_n100_p5_12549_weights.csv")
