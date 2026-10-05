"""Phase 3 — compile + profile Qwen2.5-0.5B on a real Snapdragon via Qualcomm AI Hub.

The exact chain that produced the on-device numbers (INT8 backbone, 128-token prefill,
Samsung Galaxy S24 / Snapdragon 8 Gen 3 NPU):
  export (EAGER attention -> no IsNaN op) -> package external weights ->
  INT8 quantize-job -> compile to qnn_dlc -> profile.

Prereqs (run once in the session, before this script):
  pip install -q qai-hub onnx onnxscript
  qai-hub configure --api_token <YOUR_TOKEN>      # from workbench.aihub.qualcomm.com

Key lessons baked in (each was a real failure we fixed):
  - TFLite default -> 2 GB flatbuffer limit; target qnn_dlc instead.
  - int64 intermediates -> --truncate_64bit_tensors true.
  - FP16 initializers unsupported by LiteRT, and FP16 model exceeds NPU memory -> quantize to INT8.
  - Quantize from FP32 (FP16 breaks QuantizeLinear).
  - HTP rejects IsNaN (from SDPA mask) -> attn_implementation="eager" removes it.
  - Large ONNX needs the external-weights directory format: <dir>.onnx/{model.onnx, model.data}.

Run:  python src/deploy/aihub_profile.py
"""
import os, json, glob, shutil, numpy as np, torch, onnx, qai_hub as hub
from transformers import AutoModelForCausalLM

MODEL_DIR = "/kaggle/working/qwen2.5-0.5b-instruct"
SEQ       = 128
DEVICE    = "Samsung Galaxy S24"


def ensure_model():
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return MODEL_DIR
    hits = [h for h in glob.glob("/kaggle/input/**/config.json", recursive=True) if "qwen" in h.lower()]
    if hits:
        shutil.copytree(os.path.dirname(hits[0]), MODEL_DIR, dirs_exist_ok=True)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir=MODEL_DIR)
    return MODEL_DIR


class Backbone(torch.nn.Module):
    """Transformer backbone only (no lm_head) — isolates the stack, and drops the
    151,936-wide output that bloats the graph."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids):
        return self.m.model(input_ids=input_ids.long(), use_cache=False).last_hidden_state


def export_and_package(model_dir):
    # eager attention => no IsNaN op in the mask path => composes on HTP
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    torch.onnx.export(Backbone(model).eval(), (torch.zeros(1, SEQ, dtype=torch.int32),),
                      "qwen_eager.onnx", input_names=["input_ids"], output_names=["hidden"],
                      opset_version=18, dynamo=True)
    assert not [n for n in onnx.load("qwen_eager.onnx", load_external_data=False).graph.node
                if n.op_type == "IsNaN"], "IsNaN still present — HTP will reject it"
    m = onnx.load("qwen_eager.onnx", load_external_data=True)
    shutil.rmtree("qwen_eager_pkg.onnx", ignore_errors=True); os.makedirs("qwen_eager_pkg.onnx")
    onnx.save(m, "qwen_eager_pkg.onnx/model.onnx", save_as_external_data=True,
              location="model.data", all_tensors_to_one_file=True)
    return "qwen_eager_pkg.onnx"


def main():
    model_dir = ensure_model()
    pkg = export_and_package(model_dir)
    dev = hub.Device(DEVICE)

    # INT8 calibration: a small token set is enough for the latency/memory COST we measure.
    rng = np.random.default_rng(0)
    calib = dict(input_ids=[rng.integers(0, 151936, size=(1, SEQ)).astype(np.int32) for _ in range(32)])

    qj = hub.submit_quantize_job(model=pkg, calibration_data=calib,
                                 weights_dtype=hub.QuantizeDtype.INT8,
                                 activations_dtype=hub.QuantizeDtype.INT8)
    qj.wait(); print("quantize:", qj.get_status().code, qj.url)

    cj = hub.submit_compile_job(model=qj.get_target_model(), device=dev,
                                input_specs=dict(input_ids=((1, SEQ), "int32")),
                                options="--target_runtime qnn_dlc --quantize_io --truncate_64bit_tensors true")
    cj.wait(); cs = cj.get_status(); print("compile:", cs.code, getattr(cs, "message", None), cj.url)
    if str(cs.code) != "SUCCESS":
        return

    ij = hub.submit_inference_job(model=cj.get_target_model(), device=dev, profile=True)
    ij.wait()
    prof = ij.download_profile()
    print(json.dumps(prof.get("execution_summary", {"error": ij.get_status().message}), indent=2, default=str))


if __name__ == "__main__":
    main()
