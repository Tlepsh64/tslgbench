# Compute loss
# Prediction loss - should be per batch item, not aggregated
pred_loss = self.loss_fn(y_hat_loss, y_loss, mask)  # f(x) 
if pred_loss.dim() == 0:  # If it's a scalar, we need per-batch losses
    pred_loss_per_batch = self.loss_fn_per_batch(y_hat_loss, y_loss, mask)  # Shape: (batch,)
else:
    pred_loss_per_batch = pred_loss  # Already per-batch

# MuProp Loss  
pred_loss_meanfield = self.loss_fn(y_hat_loss_meanfield, y_loss, mask)  # f(x_bar) - scalar
print('pred_loss_meanfield shape:', pred_loss_meanfield.shape)

# Compute f'(x_bar) - gradient of scalar loss w.r.t. edge logits
f_prime = torch.autograd.grad(pred_loss_meanfield, edge_logits, retain_graph=True)[0]  
print('f_prime shape:', f_prime.shape)  # Should be (128, 30, 30)

# Taylor expansion term: f'(x_bar) * (x - x_bar)
# This should be summed over the graph dimensions to get one value per batch item
taylor_exp_term = (f_prime.detach() * difference.detach()).sum(dim=(1, 2))  # Sum over 30x30, keep batch dim
print('taylor_exp_term shape:', taylor_exp_term.shape)  # Should be (128,)

# Control variate: f(x_bar) + f'(x_bar) * (x - x_bar)  
control_variate = pred_loss_meanfield.detach() + taylor_exp_term  # Broadcast scalar to (128,)
print('control_variate shape:', control_variate.shape)  # Should be (128,)

# Score function: should be sum of log-probabilities per batch item
# If your current score is (128, 30), you probably need to sum it
score = connectivity['ll']  # Currently (128, 30)
if score.dim() > 1:
    score = score.sum(dim=1)  # Sum to get (128,) - total log-prob per batch item
print('score shape:', score.shape)  # Should be (128,)

# Residual: f(x) - control_variate  
residual = pred_loss_per_batch.detach() - control_variate  # Both should be (128,)
print('residual shape:', residual.shape)  # Should be (128,)

# Likelihood-ratio component: score * residual
residual_component = (score * residual).mean()  # Average over batch to get scalar
print('residual_component shape:', residual_component.shape)  # Should be scalar

# Mean-field component: let PyTorch handle this automatically
meanfield_component = pred_loss_meanfield  # This will give gradients via chain rule
print('meanfield_component shape:', meanfield_component.shape)  # Should be scalar

# Final MuProp loss
graph_loss = meanfield_component + residual_component  # Both scalars
print('graph_loss shape:', graph_loss.shape)  # Should be scalar
