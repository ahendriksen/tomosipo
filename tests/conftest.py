# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--interactive",
        action="store_true",
        default=False,
        help="Display interactive visualizations",
    )


@pytest.fixture
def interactive(request):
    return request.config.getoption("--interactive")
