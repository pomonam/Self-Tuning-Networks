def next_batch(data_iter, data_loader, curr_epoch, device):
    """ Return the next inputs, perturbed inputs, and targets given the DataLoader.
    :param data_iter: iterator
    :param data_loader: DataLoader
    :param curr_epoch: int
    :param device: Device
    :return: Tensor, Tensor, Tensor, iterator, int
    """
    try:
        data = data_iter.next()
        if len(data) == 2:
            inputs, targets = data
            perturbed_inputs = None
        elif len(data) == 3:
            inputs, targets, perturbed_inputs = data
        else:
            raise Exception("Data type not matched... Use STN dataset.")

    except StopIteration:
        # Epoch finished.
        curr_epoch += 1
        data_iter = iter(data_loader)
        data = data_iter.next()
        if len(data) == 2:
            inputs, targets = data
            perturbed_inputs = None
        elif len(data) == 3:
            inputs, targets, perturbed_inputs = data
        else:
            raise Exception("Data type not matched.")

    inputs, targets = inputs.to(device), targets.to(device)
    perturbed_inputs = perturbed_inputs if perturbed_inputs is None else perturbed_inputs.to(device)
    return inputs, perturbed_inputs, targets, data_iter, curr_epoch
