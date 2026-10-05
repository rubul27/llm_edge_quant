# Phase 3 — Qualcomm AI Hub (on-device) notes

Target: Samsung Galaxy S24 (SM-S921U1), SoC SM8650 (Snapdragon 8 Gen 3), Hexagon v75 NPU,
QNN 2.50, fp16-capable. Profiled free via AI Hub's cloud device farm (no physical device owned).

Pipeline validated with MobileNet-V2: 0.37 ms inference, ~195 MB peak.

Qwen-0.5B custom export journey (all via `qai-hub`, target `qnn_dlc`):
1. FP32 ONNX -> TFLite default hit the 2 GB flatbuffer limit.
2. int64 intermediates -> fixed with `--truncate_64bit_tensors true`.
3. Target `qnn_dlc` (not LiteRT): FP16 compiled; profile FAILED -> exceeds device memory.
4. INT8 quantize-job (fp32 source; fp16 source breaks QuantizeLinear) -> compiled; profile
   FAILED -> QnnModel_composeGraphsFromDlc MODEL_GRAPH_ERROR.
5. Device runtime log named the culprit: node_IsNaN_206 (qti.aisw:IsNan), bool input (0x408),
   HTP validator error 3110 -> a single unsupported op from SDPA attention-mask handling.
6. Re-export with attn_implementation="eager" removed the IsNaN op (verified locally: 0 IsNaN
   nodes). Profiled the backbone (no lm_head) to isolate the transformer.

RESULT (INT8 backbone, 128-token prefill, real S24 NPU):
  - inference (prefill) latency: ~13.4 ms  (~9,500 tokens/sec)
  - peak inference memory: ~501 MB  (fits; FP16 did not)
  - cold load ~11.8 s, warm load ~0.35 s

Key lesson: a naive monolithic ONNX export of an autoregressive LLM needs op-level surgery
(remove IsNaN) and quantization to run on the Hexagon HTP. FP16 exceeds NPU memory; INT8 fits.
