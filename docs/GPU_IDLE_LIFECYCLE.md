# GPU lazy loading and idle eviction

The RunPod worker processes, RabbitMQ consumers, and HTTP API stay alive even when
GPU engines are cold. Heavy models are loaded on the first job that needs them,
kept warm for a bounded idle period, and then evicted from GPU memory if no job is
using them.

Production defaults:

| Engine | Idle TTL |
| --- | ---: |
| Pipeline Qwen ASR / classifiers | 600 s |
| Magic Clean DeepFilterNet3 | 300 s |
| Fish Speech reconstruction | 1200 s |
| AudioSep overlap specialist | 90 s |

`HEAR_GPU_IDLE_EVICTION_ENABLED=true` enables the policy.

The worker remains ready while cold. Readiness means the queue connection, runtime
assets, configuration, and CUDA availability are valid; it no longer means every
model is already resident in VRAM.
## Lifecycle

A cold job acquires a single-flight loader. If several requests arrive while the
engine is cold, one load is started and the other requests await the same load.
They do not create duplicate model instances.

When the last active borrower releases an engine, its idle timer starts. A new job
cancels that timer. When the timer expires, the engine is closed and its model
references are released. PyTorch cleanup includes garbage collection, CUDA cache
release, and IPC cache collection where applicable.

Fish uses a stronger boundary: its inference model lives in a supervised child
process. Idle eviction closes that child process. The reconstruction queue consumer
continues running and the next reconstruction job starts a new Fish child.

DeepFilterNet3 already loaded on first use; this release adds idle eviction to its
cached runtime. AudioSep remains separate from ordinary cleaning and is loaded only
for selected overlap-repair work.
## A40 verification

The real simulation runtime was started with all queue consumers ready and no model
jobs submitted. GPU memory was 3 MiB and there were no CUDA compute applications.

A real Pipeline + Magic Clean + Fish reconstruction cycle then ran through the job
API. The sampled device peak was 10,165 MiB. With proof TTLs of 12 s, 10 s, and
15 s respectively, GPU memory fell to 798 MiB 22 seconds after completion while
the API and all RabbitMQ consumers remained ready.

A second Pipeline job proved reload after eviction:

- before job: 798 MiB
- reload peak: 9,394 MiB
- after idle eviction: 798 MiB
- job exit: success

The Sound Cleanup canary also passed under lazy loading. It rose from 798 MiB to a
4,014 MiB sampled peak and settled to about 1,848 MiB after idle. The residual is
CUDA-context/runtime overhead in the four long-lived cleaner processes, not retained
DeepFilterNet/AudioSep/PANNs weights.
## CPU RAM

This is not GPU-to-RAM model offloading. Qwen and Fish are configured for CUDA
execution. Fish NF4 has a temporary CPU staging phase during cold load; earlier
measurement observed about 18.5 GiB process RSS during that phase and roughly
3.1 GiB settled RSS after residency. Idle eviction terminates the Fish inference
child entirely.

For zero-ish VRAM after a model has been used, the worker process itself would also
need to terminate because a CUDA context can retain hundreds of MiB. This design
intentionally keeps queue consumers alive, so a small context footprint can remain.

## Serverless

The same lazy wrappers are compatible with Serverless. The platform can terminate a
whole instance independently. For a future Serverless deployment, local idle
eviction can be disabled or TTLs shortened without changing job contracts.
