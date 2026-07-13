import torch

_cuda_inference_lock = None

def _get_lock():
    global _cuda_inference_lock
    if _cuda_inference_lock is None:
        import threading
        _cuda_inference_lock = threading.Lock()
    return _cuda_inference_lock

cuda_inference_lock = None

def __getattr__(name):
    if name == "cuda_inference_lock":
        return _get_lock()
    raise AttributeError(name)

class _GpuStub:
    def log_memory(self, tag: str = ""):
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            gb = 1024**3
            print(f"[GPU:{tag}] used={used/gb:.1f}GB free={free/gb:.1f}GB total={total/gb:.1f}GB")

    def acquire(self):
        pass

    def release(self):
        pass

    def exclusive(self):
        return _get_lock()

    def idle_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


gpu = _GpuStub()
