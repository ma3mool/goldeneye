import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--num-sys-name",
        action="store",
        default="fp32",
        help="provide a num sys arg to run the test with",
    )


@pytest.fixture
def params(request):
    return request.config.getoption("--num-sys-name")
