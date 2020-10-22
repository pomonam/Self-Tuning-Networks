from torch.autograd.functional import _as_tuple, _grad_preprocess, _check_requires_grad, \
    _validate_v, _autograd_grad, _fill_in_zeros, _grad_postprocess, _tuple_postprocess

import torch


def rop(y, x, v):
    w = torch.ones_like(y, requires_grad=True)
    t = torch.autograd.grad(y, x, w,
                            retain_graph=True,
                            create_graph=True)
    return torch.autograd.grad(t, w, v,
                               retain_graph=True,
                               create_graph=True)[0]


def jvp(model, data, inputs, h_param, pert, create_graph=True, strict=False):
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    v = pert
    _, v = _as_tuple(v, "v", "jvp")
    v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
    _validate_v(v, inputs, is_inputs_tuple)

    outputs = model(data, inputs[0] - inputs[0].detach(), h_param)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
    _check_requires_grad(outputs, "outputs", strict=strict)
    # The backward is linear so the value of grad_outputs is not important as
    # it won't appear in the double backward graph. We only need to ensure that
    # it does not contain inf or nan.
    grad_outputs = tuple(torch.zeros_like(out, requires_grad=True) for out in outputs)

    grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
    _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)

    jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(jvp, is_outputs_tuple)
