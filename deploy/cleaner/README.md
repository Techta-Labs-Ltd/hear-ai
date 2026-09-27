# Magic Clean runtime

Magic Clean now uses DeepFilterNet3 with Natural, Studio Voice, Outdoor & Mobile,
and Clean & Raw presets. All use `magic_clean_natural`, built as
`magic-clean-natural-pod` or `magic-clean-natural-serverless`.

Use `HEAR_OPTIONAL_ENGINE_MODE=available` for these presets. No SAM engine, model
provisioning or SAM image target remains. The older Natural-only certificate mode
remains available; certificates containing a SAM section must be regenerated.

See [the current profile and deployment guide](../../docs/DEEPFILTER_CLEANING_PROFILES.md)
for exact options, defaults, local full-file testing, export validation, certificate
migration and API/UI integration. Historical evidence in this directory concerns
previous releases and does not certify this new processing chain.
