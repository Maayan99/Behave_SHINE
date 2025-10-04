def freeze(model, metamodel):
    for param in model.parameters():
        param.requires_grad = False  # freeze the base model
    for param in metamodel.parameters():
        param.requires_grad = False  # freeze the meta model except mem_tokens
    metamodel.mem_tokens.requires_grad = True