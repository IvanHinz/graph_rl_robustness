import torch

# here observation and state is the same
def fgsm_attack(state, model, epsilon=0.5):
    state.requires_grad = True
    # q-values
    q_values = model(state.unsqueeze(0))
    # the action that gives best q(s, a)
    action = torch.argmax(q_values, dim=-1)
    # define the loss
    loss = q_values[0, action]
    # calculate the gradient
    loss.backward()
    return (state + epsilon * state.grad.data.sign()).detach()

# # fgsm on critic of actor-critic
# def fgsm_critic(state_tensor, model, epsilon=0.05):
#     state_tensor.requires_grad = True
#     # get the critic estimate
#     _, value = model(state_tensor.unsqueeze(0))
#     # define the loss
#     loss = -value
#     # calculate the gradient
#     loss.backward()
#     grad = state_tensor.grad.data
#     return (state_tensor + epsilon * grad.sign()).detach()

# # fgsm on actor of actor-critic
# def fgsm_actor(state_tensor, model, epsilon=0.05):
#     state_tensor.requires_grad = True
#     # get the actor logits
#     logits, _ = model(state_tensor.unsqueeze(0))
#     action = torch.argmax(logits)
#     # define the loss
#     loss = -logits[0, action]
#     # calculate the gradient
#     loss.backward()
#     grad = state_tensor.grad.data
#     return (state_tensor + epsilon * grad.sign()).detach()

# eacn
def eacn_attack(state, model, epsilon=0.5):
    # state.requires_grad_(True)
    # # get critic value
    # _, value = model(state.unsqueeze(0))
    # # define the loss
    # loss = value
    # # calculate the gradient
    # loss.backward()
    # grad = state.grad
    # grad_norm = grad.norm(p=2)
    # # if grad_norm < 1e-8:
    # #     return state.detach()
    # return (state - epsilon * grad / grad_norm + 1e-8).detach()
    state = state.clone().detach().requires_grad_(True)
    _, v = model(state.unsqueeze(0))
    v.backward()
    grad = state.grad
    grad_norm = grad.norm(p=2)
    return (state - epsilon * grad / (grad_norm + 1e-8)).detach()

# eaan
def eaan_attack(state, model, epsilon=0.5):
    # state = state.clone().detach()
    # state.requires_grad_(True)
    # # get the actor logits
    # logits, _ = model(state.unsqueeze(0))
    # probs = torch.softmax(logits, dim=-1)
    # # the action with the biggest probability
    # a_d = torch.argmax(probs, dim=-1).item()
    # model.zero_grad()
    # # compute the sum of gradients except the gradient of highest probability with respect to observation/state
    # sum_except_ad = 1.0 - probs[0, a_d]
    # sum_except_ad.backward(retain_graph=True)
    # grad_except_ad = state.grad.clone().detach()
    # model.zero_grad()
    # # action with the highest probability and gradient of it with respect to observation/state
    # p_ad = probs[0, a_d]
    # p_ad.backward()
    # grad_ad = state.grad.clone().detach()
    # # compute H
    # H = grad_except_ad - grad_ad
    # H_norm = torch.norm(H, p=2)
    # return (state + epsilon * H / H_norm + 1e-8).detach()
    state = state.clone().detach().requires_grad_(True)
    logits, _ = model(state.unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)
    a_d = torch.argmax(probs, dim=-1)

    model.zero_grad()
    (1.0 - probs[0, a_d]).backward(retain_graph=True)
    grad1 = state.grad.clone().detach()
    state.grad.zero_()
    probs[0, a_d].backward()
    grad2 = state.grad.clone().detach()

    H = grad1 - grad2
    H_norm = torch.norm(H, p=2)
    return (state + epsilon * H / (H_norm + 1e-8)).detach()