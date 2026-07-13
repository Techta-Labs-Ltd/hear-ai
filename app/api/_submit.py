from ray import serve as _ray_serve

def _get_orchestrator():
    return _ray_serve.get_deployment_handle("orchestrator", "default")

def _submit_job(job_id: str, run_id: str, user_id: str | None = None):
    try:
        import asyncio
        h = _get_orchestrator()
        ref = h.process.remote(job_id, run_id)
        # Verify submission works synchronously
        print(f"[API] Job {job_id[:20]} submitted to orchestrator")
    except Exception as exc:
        print(f"[API] Job {job_id[:20]} FAILED to submit: {exc}")
        raise
