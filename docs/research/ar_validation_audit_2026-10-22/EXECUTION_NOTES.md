# Execution notes

22 October 2026. The first preflight passed strict model loading, tensor equality against `best.pt`, vocabulary checks and tokenisation equality across validation. It performed the two selection-subset evaluations but failed while serialising Transformers 5.5.0's loading-info sets to JSON. No preflight metrics were committed. The reporting code was corrected to serialise sets as sorted lists; the protocol, model, data and inference settings were unchanged. The first attempt remains in `preflight.log`; its retry uses `preflight-retry.log`.
