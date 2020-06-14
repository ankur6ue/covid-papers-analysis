def export_to_onnx(model, onnx_file_name):
    model.eval()
    eval_batch_size = 1
    max_seq_length = 384
    device = torch.device("cpu")
    model.to(device)
    dummy_input = torch.ones((1, 384), dtype=torch.long)

    torch.onnx.export(model,
                      (dummy_input, dummy_input, dummy_input),
                      onnx_file_name,
                      input_names=["input_ids", "input_mask", "segment_ids"],
                      verbose=True,
                      output_names=["output_start_logits", "output_end_logits"],
                      do_constant_folding=True, opset_version=11,
                      dynamic_axes=({"input_ids": {0: "batch_size"}, "input_mask": {0: "batch_size"},
                                     "segment_ids": {0: "batch_size"},
                                     "output_start_logits": {0: "batch_size"}, "output_end_logits": {0: "batch_size"}}))
