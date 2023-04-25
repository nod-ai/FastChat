import torch
import torch_mlir
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from io import BytesIO

def compile_vicuna(model, model_inputs, model_vmfb_name):
    fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward,
                torch.ops.aten.slice_backward,
                torch.ops.aten.select_backward,
                torch.ops.aten.norm.ScalarOpt_dim,
                torch.ops.aten.native_group_norm,
                torch.ops.aten.upsample_bilinear2d.vec,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
            ]
        ),
    )(*model_inputs)
    print("Got FX_G")

    def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
        removed_indexes = []
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, (list, tuple)):
                    node_arg = list(node_arg)
                    node_args_len = len(node_arg)
                    for i in range(node_args_len):
                        curr_index = node_args_len - (i + 1)
                        if node_arg[curr_index] is None:
                            removed_indexes.append(curr_index)
                            node_arg.pop(curr_index)
                    node.args = (tuple(node_arg),)
                    break

        if len(removed_indexes) > 0:
            fx_g.graph.lint()
            fx_g.graph.eliminate_dead_code()
            fx_g.recompile()
        removed_indexes.sort()
        return removed_indexes


    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
        """
        Replace tuple with tuple element in functions that return one-element tuples.
        Returns true if an unwrapping took place, and false otherwise.
        """
        unwrapped_tuple = False
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    if len(node_arg) == 1:
                        node.args = (node_arg[0],)
                        unwrapped_tuple = True
                        break

        if unwrapped_tuple:
            fx_g.graph.lint()
            fx_g.recompile()
        return unwrapped_tuple


    def transform_fx(fx_g):
        for node in fx_g.graph.nodes:
            if node.op == "call_function":
                if node.target in [
                    torch.ops.aten.empty,
                ]:
                    # aten.empty should be filled with zeros.
                    if node.target in [torch.ops.aten.empty]:
                        with fx_g.graph.inserting_after(node):
                            new_node = fx_g.graph.call_function(
                                torch.ops.aten.zero_,
                                args=(node,),
                            )
                            node.append(new_node)
                            node.replace_all_uses_with(new_node)
                            new_node.args = (node,)

        fx_g.graph.lint()


    transform_fx(fx_g)
    fx_g.recompile()
    removed_none_indexes = _remove_nones(fx_g)
    was_unwrapped = _unwrap_single_tuple_return(fx_g)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    print("FX_G recompile")

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()


    strip_overloads(fx_g)

    ts_g = torch.jit.script(fx_g)

    print("Got TS_G")

    module = torch_mlir.compile(
        ts_g,
        [*model_inputs],
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    bytecode_stream = BytesIO()
    module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    from shark.shark_inference import SharkInference
    shark_module = SharkInference(
        mlir_module=bytecode, device="cuda", mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    import os
    path = shark_module.save_module(
        os.getcwd(), model_vmfb_name, []
    )
    print("Saved vmfb at ", str(path))

    return shark_module


kwargs = {"torch_dtype": torch.float32}
model_path = "TheBloke/vicuna-7B-1.1-HF"

# Requires input_ids as tensor(1x40)
class FirstVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    def forward(self, input_ids):
        op = self.model(input_ids=input_ids, use_cache=True)
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)
firstVicuna = FirstVicuna(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
prompt = "User said hello"
input_ids = tokenizer(prompt).input_ids
print("Got input_ids from the tokenizer")
firstVicunaInput = tuple([torch.as_tensor([input_ids], device=torch.device("cpu"))])

shark_first_vicuna = compile_vicuna(firstVicuna, firstVicunaInput, "first_vicuna")
# output_first_vicuna = shark_first_vicuna("forward", (input_ids,))

# Uncomment this after verifying that SecondVicuna compiles as well.
# Might have to cast to_numpy.
# last_token_logits = output_first_vicuna[0][0][-1]
# temperature = 0.7
# probs = torch.softmax(last_token_logits / temperature, dim=-1)
# token = int(torch.multinomial(probs, num_samples=1))

# Requires input_ids as tensor(1x1),
#          past_key_values = 32 length tuple containing tuple of tensor pairs, which is same as output
#                            of firstVicuna[1:]
# class SecondVicuna(torch.nn.Module):
#     def __init__(self, model_path):
#         super().__init__()
#         self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#     def forward(self, input_tuple):
#         input_ids = input_tuple[0]
#         past_key_values = []
#         for e1, e2 in zip(input_tuple, input_tuple[1:]):
#             past_key_values.append(tuple(e1, e2))
#         past_key_values = tuple(past_key_values)
#         op = self.model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
#         return_vals = []
#         return_vals.append(op.logits)
#         temp_past_key_values = op.past_key_values
#         for item in temp_past_key_values:
#             return_vals.append(item[0])
#             return_vals.append(item[1])
#         return tuple(return_vals)
# secondVicuna = SecondVicuna(model_path)