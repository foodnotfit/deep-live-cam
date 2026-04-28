"""
PyTorch + MPS backed face swapper for Apple Silicon.

Drop-in replacement for `insightface.model_zoo.get_model(<inswapper.onnx>)`.
Uses the same insightface INSwapper alignment/paste-back logic, but runs
the actual neural network forward pass on Apple's GPU via PyTorch's MPS
backend instead of onnxruntime's CPU/CoreML EP.

Why this exists:
  - onnxruntime's CoreML EP can only dispatch 34/273 ops in inswapper
    (verified by GetCapability) so it falls back to CPU for 87% of work.
  - onnxruntime's CPU path is broken on this machine (~654 ms/call).
  - PyTorch's MPS backend gives ~79 ms/call on the same model (8x faster).

Validated end-to-end in mps/phase3_swap_face.py: produces a correct
Trump-on-Obama swap at ~6.9 FPS for the full live pipeline.
"""

from typing import Any, List
import warnings
import onnx
import torch
from onnx import numpy_helper

# Lazy import — onnx2torch is only required when MPS swapper is actually used
_onnx2torch_convert = None


class _MockIO:
    """Minimal stand-in for onnxruntime's NodeArg objects."""
    def __init__(self, name: str):
        self.name = name


class _MPSSession:
    """Mimics onnxruntime.InferenceSession but runs on Apple GPU via MPS.

    insightface's INSwapper only uses three things from a session:
      - session.run(output_names, input_dict) -> list[np.ndarray]
      - session.get_inputs() -> list[NodeArg(name=...)]
      - session.get_outputs() -> list[NodeArg(name=...)]
    """

    def __init__(self, onnx_path: str):
        global _onnx2torch_convert
        if _onnx2torch_convert is None:
            from onnx2torch import convert as _conv
            _onnx2torch_convert = _conv

        self.device = torch.device("mps")
        onnx_model = onnx.load(onnx_path)
        self.torch_model = _onnx2torch_convert(onnx_model).to(self.device).eval()
        self._input_names = [i.name for i in onnx_model.graph.input]
        self._output_names = [o.name for o in onnx_model.graph.output]
        self._onnx_model = onnx_model

        # Warmup: first MPS call compiles kernels (~1-2 sec). Better to pay
        # this cost at session construction than on the first user-visible swap.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dummy = []
            for inp in onnx_model.graph.input:
                dims = [d.dim_value if d.dim_value > 0 else 1
                        for d in inp.type.tensor_type.shape.dim]
                dummy.append(torch.randn(*dims, device=self.device,
                                         dtype=torch.float32))
            with torch.inference_mode():
                _ = self.torch_model(*dummy)
                torch.mps.synchronize()

    def run(self, output_names, input_dict):
        torch_inputs = []
        for name in self._input_names:
            arr = input_dict[name]
            torch_inputs.append(torch.from_numpy(arr).to(self.device))
        with torch.inference_mode():
            out = self.torch_model(*torch_inputs)
            torch.mps.synchronize()
        if isinstance(out, (tuple, list)):
            return [t.detach().cpu().numpy() for t in out]
        return [out.detach().cpu().numpy()]

    def get_inputs(self) -> List[Any]:
        return [_MockIO(n) for n in self._input_names]

    def get_outputs(self) -> List[Any]:
        return [_MockIO(n) for n in self._output_names]


def get_mps_inswapper(model_file: str):
    """Construct an insightface INSwapper that uses MPS for inference.

    Mirrors what insightface.model_zoo.get_model does for inswapper, but
    swaps the onnxruntime session for an MPS-backed one.
    """
    from insightface.model_zoo.inswapper import INSwapper

    session = _MPSSession(model_file)
    swapper = INSwapper.__new__(INSwapper)
    swapper.model_file = model_file
    swapper.session = session
    swapper.taskname = "inswapper"

    # The emap is the last initializer in the inswapper ONNX graph
    swapper.emap = numpy_helper.to_array(session._onnx_model.graph.initializer[-1])

    swapper.input_mean = 0.0
    swapper.input_std = 255.0

    inputs = session.get_inputs()
    swapper.input_names = [inputs[0].name, inputs[1].name]
    swapper.output_names = [o.name for o in session.get_outputs()]

    # input_size from the model's first input (target image): (W, H)
    dims = session._onnx_model.graph.input[0].type.tensor_type.shape.dim
    H, W = dims[2].dim_value, dims[3].dim_value
    swapper.input_size = (W, H)

    return swapper
