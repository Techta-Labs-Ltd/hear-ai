from hear.orchestrator import Orchestrator


class DeploymentUnavailableError(Exception):
    pass


def test_deployment_unavailable_error_is_actionable_for_backend():
    sanitize = Orchestrator.func_or_class._sanitize_error

    assert sanitize(DeploymentUnavailableError("internal deployment id")) == (
        "Required AI processing service is unavailable. Please contact operations."
    )
