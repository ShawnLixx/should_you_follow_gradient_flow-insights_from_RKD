import torch
import numpy as np

def get_gradient_norm(model, norm_type=2.0):
    total_norm = torch.norm(torch.stack(
        [torch.norm(
            p.grad.detach(), norm_type) \
                    for p in model.parameters()]), norm_type)
    return total_norm.item()

def chunk_forward(model, inputs, labels, criterion, args):
    # Chunk inputs and labels for very large batch
    batch_size = len(inputs)
    if args.chunk_size is None:
        chunk_size = batch_size
    else:
        chunk_size = args.chunk_size

    num_chunks = int(np.ceil(batch_size / chunk_size))
    sum_loss = torch.zeros(1, device=args.device)
    outputs = []
    for chunk_inputs, chunk_labels in zip(
            inputs.chunk(num_chunks, 0),
            labels.chunk(num_chunks, 0)):

        chunk_size = len(chunk_inputs)
        chunk_outputs = model(chunk_inputs)
        loss = criterion(chunk_outputs, chunk_labels) \
                * (chunk_size / batch_size)
        loss.backward()

        sum_loss += loss.detach()
        outputs.append(chunk_outputs.detach())

    return outputs, sum_loss
