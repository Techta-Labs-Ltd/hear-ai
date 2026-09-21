from hear.services.jobs.workflows import normalize_job_type, workflow_for


def test_workflow_registry_normalizes_legacy_aliases_without_fallback():
    assert normalize_job_type("tagging") == "categorization"
    assert workflow_for("tagging").operation == "categorization"
    assert workflow_for("not-a-job") is None


def test_rebuild_retains_legacy_pipeline_lane():
    workflow = workflow_for("rebuild")
    assert workflow is not None
    assert workflow.operation == "pipeline"
    assert workflow.execution_lane == "pipeline"
