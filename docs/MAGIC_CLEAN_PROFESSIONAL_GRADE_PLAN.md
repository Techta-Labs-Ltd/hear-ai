# Magic Clean production quality plan

## Certification identity

Each profile certificate binds the profile version, engine and runtime digest, checkpoint digest when the engine has a model, precision policy, long-form policy, and supported input/resource limits. Runtime validates the certificate before accepting profile work. Changes to any bound item require new certification evidence.

## Profile-specific validation

| Profile | Required behavior to certify |
| --- | --- |
| Natural | Preserve channels and duration; enforce the selected attenuation limit; compare against approved natural-cleaning references. |
| SAM Audio | Require one text-prompted sound source and an explicit remove or isolate action; accept stereo only under the acknowledged dual-mono policy; reject absent isolated targets; validate residual integrity and artifact limits. |

All certified profiles must preserve source identity, produce deterministic manifests for the same inputs and configuration, enforce deadlines and resource limits, and publish outputs atomically with checksummed artifact metadata.

## Evidence required for release

1. Maintain a versioned golden set covering speech, music, noise, silence, clipping, channel layouts, sample rates, and long recordings.
2. Record subjective review and objective measurements for intelligibility, loudness, clipping, duration, channel behavior, and profile-specific separation or noise reduction.
3. Compare Pod and Serverless outputs and confirm that backend attempt fencing prevents stale artifacts from becoming current.
4. Measure peak GPU memory, CPU memory, scratch use, runtime, and cancellation latency at supported input limits.
5. Exercise missing/corrupt assets, worker restart, interrupted upload, duplicate publication, and deadline expiry.
6. Approve the profile certificate and pin it with the deployed image and assets.

Runtime tests establish contract and failure behavior; they do not replace audio quality review or production certification. Until the evidence is approved, report a profile as implemented but not production certified.
