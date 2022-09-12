import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--num-sys-name",
        action="store",
        default="fp32",
        help="provide a num sys arg to run the test with",
    )
    parser.addoption(
        "--model",
        action="store",
        default="alexnet",
        help="provide a num sys arg to run the test with",
    )


@pytest.fixture
def params(request):
    params = {}
    params["num_sys_name"] = request.config.getoption("--num-sys-name")
    params["model"] = request.config.getoption("--model")
    # if params["num_sys_name"] is None or params["password"] is None:
    #     pytest.skip()
    return params
