"""Fleet health findings and repairs against a scripted RunPod."""

import json

import httpx

from hear.tools.runpod_api import RunPodApi
from scripts.fleet_health import FleetHealth

MIG24 = "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"
PLANS = {
    "hear-ai-pipeline-llm": {
        "name": "hear-ai-pipeline-llm",
        "workersMax": 5,
        "gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA L4"],
        "gpuExcludes": [MIG24],
    },
    "hear-ai-cleaner": {
        "name": "hear-ai-cleaner",
        "workersMax": 3,
        "gpuTypeIds": ["NVIDIA RTX A5000"],
        "gpuExcludes": [MIG24],
    },
}


def endpoint(name, workers_max, gpu_ids, pods=(), health=None):
    return {
        "id": name[-4:],
        "name": name,
        "templateId": "tpl",
        "gpuIds": gpu_ids,
        "idleTimeout": 180,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": workers_max,
        "workersStandby": workers_max,
        "networkVolumeId": "",
        "locations": None,
        "template": {"id": "tpl", "imageName": "img@sha256:x"},
        "pods": [{"id": f"p{i}", "machine": {"gpuTypeId": gpu}} for i, gpu in enumerate(pods)],
        "_health": health or {"jobs": {"inQueue": 0}, "workers": {"ready": 1}},
    }


def harness(endpoints, balance=100.0, pod_status=200):
    calls = []
    saved = []

    def handle(request):
        path = request.url.path
        calls.append((request.method, path))
        if path == "/graphql":
            body = json.loads(request.content)
            if "saveEndpoint" in body["query"]:
                saved.append(body["variables"]["input"])
                return httpx.Response(
                    200, json={"data": {"saveEndpoint": body["variables"]["input"]}}
                )
            visible = [{k: v for k, v in e.items() if k != "_health"} for e in endpoints]
            return httpx.Response(
                200, json={"data": {"myself": {"clientBalance": balance, "endpoints": visible}}}
            )
        for e in endpoints:
            if path == f"/v2/{e['id']}/health":
                return httpx.Response(200, json=e["_health"])
        if path == "/readyz":
            return httpx.Response(pod_status, json={})
        raise AssertionError(path)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    api = RunPodApi(
        "rpa_test",
        rest_base="https://rest.test/v1",
        run_base="https://run.test/v2",
        graphql_base="https://gql.test/graphql",
        client=client,
    )
    return FleetHealth(api, PLANS, min_balance=25, pod_url="https://pod.test", client=client), saved


def test_healthy_fleet_has_no_findings():
    checker, _ = harness(
        [
            endpoint("hear-ai-pipeline-llm", 5, f"AMPERE_24,-{MIG24}", pods=["NVIDIA RTX A5000"]),
            endpoint("hear-ai-cleaner", 3, f"AMPERE_24,-NVIDIA L4,-{MIG24}"),
        ]
    )
    assert checker.findings(checker.snapshot()) == []


def test_every_outage_condition_is_named():
    checker, _ = harness(
        [
            endpoint(
                "hear-ai-pipeline-llm",
                5,
                "AMPERE_24",
                pods=[MIG24, "NVIDIA RTX A5000"],
                health={"jobs": {"inQueue": 1}, "workers": {"ready": 0, "unhealthy": 5}},
            ),
            endpoint(
                "hear-ai-cleaner",
                0,
                "AMPERE_24,-NVIDIA L4",
                health={"jobs": {"inQueue": 1}, "workers": {}},
            ),
        ],
        balance=18.09,
        pod_status=404,
    )
    assert checker.findings(checker.snapshot()) == [
        "unhealthy_workers:hear-ai-pipeline-llm:5",
        "queue_stuck:hear-ai-pipeline-llm:1",
        f"worker_on_unplanned_gpu:hear-ai-pipeline-llm:{MIG24}",
        f"gpu_exclusion_missing:hear-ai-pipeline-llm:{MIG24}",
        "workers_max_below_plan:hear-ai-cleaner:0<3",
        "queue_stuck:hear-ai-cleaner:1",
        f"gpu_exclusion_missing:hear-ai-cleaner:{MIG24}",
        "balance_low:18.09<25.00",
        "pod_not_ready:404",
    ]


def test_repair_restores_caps_and_excludes_every_unplanned_gpu_seen():
    checker, saved = harness(
        [
            endpoint(
                "hear-ai-pipeline-llm",
                5,
                "AMPERE_24",
                pods=[MIG24, "NVIDIA RTX PRO 4000 Blackwell"],
            ),
            endpoint("hear-ai-cleaner", 0, f"AMPERE_24,-NVIDIA L4,-{MIG24}"),
        ]
    )
    actions = checker.repair(checker.snapshot())
    assert len(actions) == 2 and len(saved) == 2
    pipeline, cleaner = saved
    assert (
        pipeline["id"] == "-llm"
        and pipeline["gpuIds"] == f"AMPERE_24,-{MIG24},-NVIDIA RTX PRO 4000 Blackwell"
    )
    assert pipeline["workersMax"] == 5 and pipeline["templateId"] == "tpl"
    assert (
        cleaner["workersMax"] == 3
        and cleaner["workersStandby"] == 3
        and cleaner["gpuIds"] == f"AMPERE_24,-NVIDIA L4,-{MIG24}"
    )
    assert "networkVolumeId" not in cleaner and "locations" not in cleaner


def test_repair_leaves_a_compliant_endpoint_alone():
    checker, saved = harness([endpoint("hear-ai-cleaner", 3, f"AMPERE_24,-NVIDIA L4,-{MIG24}")])
    assert checker.repair(checker.snapshot()) == [] and saved == []


def test_committed_plans_exclude_blackwell_mig_and_load_by_name(tmp_path):
    from pathlib import Path

    plans = FleetHealth.load_plans(Path("deploy/runpod"))
    assert set(plans) == {"hear-ai-pipeline-llm", "hear-ai-cleaner", "hear-ai-reconstruction"}
    assert all(any("MIG" in item for item in plan["gpuExcludes"]) for plan in plans.values())
    assert (
        plans["hear-ai-reconstruction"]["workersMax"] == 2
        and plans["hear-ai-cleaner"]["workersMax"] == 3
    )
