import torch
import torch.nn.utils.prune as prune

def prune_model(model, pruning_ratio, method, layer):
    parameters_to_prune = []
    if hasattr(model, 'module'):
        model = model.module
    if 'structured' not in method:
        if 'encoder' in layer:
            for i in range(12):
                parameters_to_prune.append((model.encoder.encoder.layer[i].attention.self.query, 'weight'))
                parameters_to_prune.append((model.encoder.encoder.layer[i].attention.self.key, 'weight'))
                parameters_to_prune.append((model.encoder.encoder.layer[i].attention.self.value, 'weight'))
                parameters_to_prune.append((model.encoder.encoder.layer[i].attention.output.dense, 'weight'))
                parameters_to_prune.append((model.encoder.encoder.layer[i].intermediate.dense, 'weight'))
                parameters_to_prune.append((model.encoder.encoder.layer[i].output.dense, 'weight'))
            parameters_to_prune.append((model.encoder.pooler.dense, 'weight'))
        if 'decoder' in layer:
            for i in range(6):
                parameters_to_prune.append((model.decoder.layers[i].self_attn.out_proj, 'weight'))
                parameters_to_prune.append((model.decoder.layers[i].multihead_attn.out_proj, 'weight'))
                parameters_to_prune.append((model.decoder.layers[i].linear1, 'weight'))
                parameters_to_prune.append((model.decoder.layers[i].linear2, 'weight'))
        parameters_to_prune = tuple(parameters_to_prune)
        # random pruning

        if method == "l1":
            pruning_method = prune.L1Unstructured
        elif method == "random":
            pruning_method = prune.RandomUnstructured
        else:
            raise NotImplementedError 

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=pruning_ratio
        )
    elif 'structured' in method:
        if method == "l1_structured":
            pruning_method = prune.L1Unstructured
        elif method == "random_structured":
            pruning_method = prune.RandomUnstructured
        else:
            raise NotImplementedError

        if 'encoder' in layer:
            for i in range(12):
                prune.global_unstructured([(model.encoder.encoder.layer[i].attention.self.query, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.encoder.encoder.layer[i].attention.self.key, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.encoder.encoder.layer[i].attention.self.value, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.encoder.encoder.layer[i].attention.output.dense, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.encoder.encoder.layer[i].intermediate.dense, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.encoder.encoder.layer[i].output.dense, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
            prune.global_unstructured([(model.encoder.pooler.dense, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
        if 'decoder' in layer:
            for i in range(6):
                prune.global_unstructured([(model.decoder.layers[i].self_attn.out_proj, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.decoder.layers[i].multihead_attn.out_proj, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.decoder.layers[i].linear1, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)
                prune.global_unstructured([(model.decoder.layers[i].linear2, 'weight')], pruning_method=pruning_method, amount=pruning_ratio)

def see_weight_rate(model, layer):
    if hasattr(model, 'module'):
        model = model.module
    sum_list = 0
    zero_sum = 0
    if 'encoder' in layer:
        for ii in range(12):
            sum_list = sum_list+float(model.encoder.encoder.layer[ii].attention.self.query.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(model.encoder.encoder.layer[ii].attention.self.query.weight == 0))

            sum_list = sum_list+float(model.encoder.encoder.layer[ii].attention.self.key.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(model.encoder.encoder.layer[ii].attention.self.key.weight == 0))

            sum_list = sum_list+float(model.encoder.encoder.layer[ii].attention.self.value.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(model.encoder.encoder.layer[ii].attention.self.value.weight == 0))

            sum_list = sum_list+float(model.encoder.encoder.layer[ii].attention.output.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(model.encoder.encoder.layer[ii].attention.output.dense.weight == 0))

            sum_list = sum_list+float(model.encoder.encoder.layer[ii].intermediate.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(model.encoder.encoder.layer[ii].intermediate.dense.weight == 0))

            sum_list = sum_list+float(model.encoder.encoder.layer[ii].output.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(model.encoder.encoder.layer[ii].output.dense.weight == 0))

        sum_list = sum_list+float(model.encoder.pooler.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.pooler.dense.weight == 0))

    if 'decoder' in layer:
        for ii in range(6):
            sum_list = sum_list + float(model.decoder.layers[ii].self_attn.out_proj.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(model.decoder.layers[ii].self_attn.out_proj.weight == 0))

            sum_list = sum_list + float(model.decoder.layers[ii].multihead_attn.out_proj.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(model.decoder.layers[ii].multihead_attn.out_proj.weight == 0))

            sum_list = sum_list + float(model.decoder.layers[ii].linear1.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(model.decoder.layers[ii].linear1.weight == 0))

            sum_list = sum_list + float(model.decoder.layers[ii].linear2.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(model.decoder.layers[ii].linear2.weight == 0))

    return 100*zero_sum/sum_list

def rewind_model(model, orig_dict):
    if hasattr(model, 'module'):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    model_dict.update(orig_dict)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)

def capture_orig_state_dict(orig_weights, layer):
    recover_dict = {}
    name_list = []
    if 'encoder' in layer:
        for ii in range(12):
            name_list.append('encoder.encoder.layer.'+str(ii)+'.attention.self.query.weight')
            name_list.append('encoder.encoder.layer.'+str(ii)+'.attention.self.key.weight')
            name_list.append('encoder.encoder.layer.'+str(ii)+'.attention.self.value.weight')
            name_list.append('encoder.encoder.layer.'+str(ii)+'.attention.output.dense.weight')
            name_list.append('encoder.encoder.layer.'+str(ii)+'.intermediate.dense.weight')
            name_list.append('encoder.encoder.layer.'+str(ii)+'.output.dense.weight')
        name_list.append('encoder.pooler.dense.weight')
    if 'decoder' in layer:
        for ii in range(6):
            name_list.append('decoder.layers.'+str(ii)+'.self_attn.out_proj.weight')
            name_list.append('decoder.layers.'+str(ii)+'.multihead_attn.out_proj.weight')
            name_list.append('decoder.layers.'+str(ii)+'.linear1.weight')
            name_list.append('decoder.layers.'+str(ii)+'.linear2.weight')

    for key in orig_weights.keys():
        if key in name_list:
            new_key = key+'_orig'
        else:
            new_key = key
        recover_dict[new_key] = orig_weights[key]

    return recover_dict

def modify_state_dict(state_dict):
    """
    Takes a state dict of a pruned model and returns
    a state dict of the corresponding unpruned model.
    Pruned elements are 0.
    """
    new_state_dict = {}
    # multiply the _orig and the _mask
    for key in state_dict.keys():
        if "_orig" in key:
            orig_key = key
            mask_key = key.replace("_orig", "_mask")
            new_key = key.replace("_orig", "")
            new_state_dict[new_key] = state_dict[orig_key] * state_dict[mask_key]
        elif "_mask" in key:
            continue
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict