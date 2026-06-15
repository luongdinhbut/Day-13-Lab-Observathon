# Diagnosis scratchpad

Implemented mitigations:

| symptom (from telemetry) | which requests | suspected cause | config fix? | wrapper fix? |
|---|---|---|---|---|
| error spikes / tool failures | intermittent tool calls | `tool_error_rate` enabled, retry disabled | set `tool_error_rate: 0.0`, enable bounded retry | log `trace_errors`, retry once |
| latency and cost spikes | repeated or long agent runs | high `max_steps`, no loop guard/tool budget, verbose context | cap steps/tokens/context, enable loop guard/cache | cache repeated sanitized questions, log latency/tokens/cost |
| quality drift | later session turns | `session_drift_rate` enabled, no reset | set drift to 0, reset every 6 turns, low temperature | route strong prompt per request |
| tool failure on names/cities | MacBook / diacritic destinations | bad `catalog_override`, Unicode normalization disabled | clear override, enable normalization | strip accents for cache/sanitize matching |
| PII leak | requests containing email/phone | redaction disabled and prompt had no PII rule | enable `redact_pii` | redact question before call and answer after call |
| fabrication / arithmetic errors | unknown/out-of-stock or discounted orders | weak prompt, high temperature, no formula | low temperature, verify, self-consistency 2 | prompt requires grounding and exact formula |
| tool overuse / loops | multi-tool orders | no budget or per-tool limit | `tool_budget: 4`, loop guard | detect repeated tools in telemetry |
| prompt injection | private note/GHI CHU attacks | notes treated as instructions | injection-safe prompt | strip note/instruction tails before model call |

Validation:

- `python -m py_compile solution/wrapper.py` passed.
- `python harness/selfcheck.py` passed all checks.
- `observathon-sim.exe --help` failed in this environment while loading PyInstaller's temporary `python312.dll`, so practice traffic still needs to be run in a normal local shell with a configured LLM key/endpoint.
