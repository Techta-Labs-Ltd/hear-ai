from hear.deployments.gateway import http_app


EXPECTED_HTTP_ENDPOINTS = {
    ("/", "get"),
    ("/health", "get"),
    ("/ready", "get"),
    ("/process", "post"),
    ("/discovery", "post"),
}


def test_every_http_endpoint_has_the_expected_method():
    schema = http_app.openapi()
    actual = {
        (path, method)
        for path, operations in schema["paths"].items()
        for method in operations
        if method in {"get", "post", "put", "patch", "delete"}
    }
    assert actual == EXPECTED_HTTP_ENDPOINTS


def test_every_http_endpoint_has_an_operation_id_and_success_response():
    schema = http_app.openapi()
    for path, method in EXPECTED_HTTP_ENDPOINTS:
        operation = schema["paths"][path][method]
        assert operation["operationId"]
        assert any(code.startswith("2") for code in operation["responses"])


def test_fastapi_system_routes_are_registered():
    paths = set(http_app.openapi()["paths"])

    assert {"/", "/health", "/ready", "/process", "/discovery"} <= paths


def test_process_is_the_primary_http_job_submission_route():
    schema = http_app.openapi()
    paths = set(schema["paths"])

    assert "/process" in paths
    assert "/discovery" in paths
    assert "/api/v1/process" not in paths
    operation = schema["paths"]["/process"]["post"]
    assert "requestBody" in operation
    assert {parameter["name"] for parameter in operation["parameters"]} == {
        "X-Service-Key"
    }


def test_process_contract_exposes_magic_clean_stem_percentages():
    schema = http_app.openapi()
    request = schema["components"]["schemas"]["PipelineRequest"]

    for field in ("speech", "music", "background"):
        definition = request["properties"][field]
        percentage = next(
            item for item in definition["anyOf"] if item.get("type") == "integer"
        )
        assert percentage["minimum"] == 0
        assert percentage["maximum"] == 100

    assert request["properties"]["cut_silence"]["type"] == "boolean"
    assert request["properties"]["cut_silence"]["default"] is False
