import torch
from torchmetrics.utilities.checks import _check_same_shape
from lib.nn.metrics import MaskedScoreFuctionLoss
from lib.gradient_estimators import DoubleCVEstimator
from tsl.metrics.torch import mae
from tsl.engines import Predictor
from tsl.ops.connectivity import adj_to_edge_index
from einops import rearrange
from torch_geometric.utils import to_torch_sparse_tensor
from torch_geometric.utils import to_dense_adj
import torch.nn as nn
import numpy as np

class LatentGraphPredictor(Predictor):
    def __init__(self,
                 graph_module_class,
                 graph_module_kwargs,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scale_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 mc_samples=1,
                 eval_mode='mode'):
        super().__init__(model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs)

        self.graph_module_class = graph_module_class
        self.graph_module_kwargs = graph_module_kwargs
        self.graph_module = self.graph_module_class(**self.graph_module_kwargs)
        self.mc_samples = mc_samples
        self.eval_mode = eval_mode

        # Storage for test graphs (populated during evaluation with eval_mode='sampling')
        self.test_graphs = None  # List of adjacency matrices from MC samples

    def forward(self, *args, mode='forward', **kwargs):
        if mode == 'pred_only':
            connectivity = kwargs.pop('connectivity', None)
            if connectivity is not None:
                kwargs.update(**connectivity)
            return self.model(*args, **kwargs)
        if mode == 'graph_only':
            return self.graph_module(*args, **kwargs)
        if self.training or mode == 'sampling':
            connectivity = self.graph_module(*args, **kwargs)
            kwargs.update(**connectivity)
            return self.model(*args, **kwargs)

        if self.eval_mode == 'sampling':
            # if in inference mode, take the average of M MC samples
            outs = []
            adjs = []  # Collect adjacency matrices for mean computation
            for _ in range(self.mc_samples):
                connectivity = self.graph_module(*args, **kwargs)
                adjs.append(connectivity['adj'])
                kwargs_copy = kwargs.copy()
                kwargs_copy.update(**connectivity)
                out = self.model(*args, **kwargs_copy)
                outs.append(out)
            # Store individual adjacency matrices
            self.test_graphs = [adj.detach().cpu() for adj in adjs]
            return torch.stack(outs).mean(0)
        if self.eval_mode == 'mode':
            connectivity = self.graph_module(*args, **kwargs)['mean_graph']
            kwargs.update(**connectivity, disjoint=True, adj=None)
            return self.model(*args, **kwargs)

class SFGraphPredictor(LatentGraphPredictor):
    def __init__(self,
                 graph_module_class,
                 graph_module_kwargs,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scale_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 use_baseline=False,
                 sf_weight=1.,
                 mc_samples=1,
                 eval_mode='mode',
                 variance_reduced=True,
                 surrogate_lam=None,
                 gradient_level = 'graph_level',
                 doublecv_mode = 'full',
                 use_taylor = False):
        super().__init__(graph_module_class=graph_module_class,
                         graph_module_kwargs=graph_module_kwargs,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs,
                         mc_samples=mc_samples,
                         eval_mode=eval_mode,)

        self.sf_loss = MaskedScoreFuctionLoss(
            cost_fn=self.loss_fn.metric_fn,
            variance_reduced=variance_reduced,
            lam=surrogate_lam
        )

        self.use_baseline = use_baseline
        self.sf_weight = sf_weight
        if use_baseline == 'doublecv':
            # Create Double-CV gradient estimator
            self.doublecv_estimator = DoubleCVEstimator(
                mode=doublecv_mode,
                gradient_level=gradient_level,
                alpha_lr=1e-3
            )
            self.gradient_level = gradient_level
            self.use_taylor = use_taylor

    def training_step(self, batch, batch_idx):

        y = batch.y
        mask = batch.mask

        k = self.mc_samples

        if self.use_baseline == 'rloo':
            y_hat_loss_samples = []
            score_samples = []
            pred_loss_samples = []

            for i in range(k):
                connectivity_i = self.predict_batch(batch,
                                                    preprocess=False,
                                                    postprocess=False,
                                                    mode='graph_only')
                
                mean_graph = connectivity_i.pop('mean_graph')
                score_samples.append(connectivity_i['ll'])
                
                y_hat_scaled_i = self.predict_batch(batch,
                                                    preprocess=False,
                                                    postprocess=False,
                                                    mode='pred_only',
                                                    connectivity=connectivity_i)

                y_hat_i = batch.transform['y'].inverse_transform(y_hat_scaled_i)
                y_scaled = batch.transform['y'].transform(y)

                y_hat_loss_i = y_hat_scaled_i if self.scale_target else y_hat_i
                y_loss_i = y_scaled if self.scale_target else y
                y_hat_loss_samples.append(y_hat_loss_i)
                    
                # CRITICAL: Keep gradients in the cost computation
                pred_loss_i = mae(y_hat_loss_i, y_loss_i, mask, reduction='none')
                #pred_loss_i = self.loss_fn(y_hat_loss_i, y_loss_i, mask, reduction='none')
                mask = self.sf_loss._check_mask(mask, pred_loss_i)
                pred_loss_i = torch.where(mask, pred_loss_i, torch.zeros_like(pred_loss_i))
                pred_loss_samples.append(pred_loss_i)

            y_hat_loss_samples_tensor = torch.stack(y_hat_loss_samples, dim=0)
            score_samples_tensor = torch.stack(score_samples, dim=0)
            pred_loss_samples_tensor = torch.stack(pred_loss_samples, dim=0)

            # Compute LOO baselines for costs (keep this detached)
            if k > 1:
                with torch.no_grad():
                    sum_of_all_pred_loss = torch.sum(pred_loss_samples_tensor.detach(), dim=0, keepdim=True)
                    loo_baselines_values = (sum_of_all_pred_loss - pred_loss_samples_tensor.detach()) / (k - 1)
            else:
                loo_baselines_values = None

            # Compute graph loss maintaining gradients
            valid_element_count = mask.sum()
            graph_loss = 0
            for i in range(k):
                # Reset metric state before each call to prevent accumulation across samples
                self.sf_loss.reset()
                # We need the cost to compute the graph loss, but its gradient is not tracked.
                baseline_i = None if loo_baselines_values is None else loo_baselines_values[i]
                graph_loss+= self.sf_loss(score=score_samples_tensor[i],
                                    y_hat=y_hat_loss_samples[i].detach(),
                                    y=y_loss_i,
                                    baseline=baseline_i,
                                    mask=mask)

            graph_loss = graph_loss / k

            # Compute pred loss by taking the mean of k samples
            pred_loss = 0
            for i in range(k):
                mean_pred_loss_sample_i = pred_loss_samples_tensor[i].sum() / valid_element_count
                pred_loss += mean_pred_loss_sample_i
            pred_loss = pred_loss / k

            # Alternative to latest code block above: Weighted Average of Pred Loss by using baseline losses as weights
            """print('Shape of Pred Loss Samples Tensor:', pred_loss_samples_tensor.shape)
            pred_loss = 0
            tot_weight = 0
            for i in range(k):
                weighted_loss = pred_loss_samples_tensor[i] * loo_baselines_values[i].detach()
                weighted_loss = weighted_loss.sum()
                tot_weight += loo_baselines_values[i].detach().sum()
                #mean_pred_loss_sample_i = pred_loss_samples_tensor[i].sum() / valid_element_count
                #mean_pred_loss_sample_i = torch.mean(pred_loss_samples_tensor[i], dim=0)
                print('Shape of weighted_loss:', weighted_loss.shape)
                pred_loss += weighted_loss
            pred_loss = pred_loss / tot_weight"""

            # After computing the LOO baselines, add variance calculation
            if k > 1:
                with torch.no_grad():
                    # Calculate RLOO estimator variance
                    # For each sample i, compute (cost_i - baseline_i)
                    advantages = pred_loss_samples_tensor.detach() - loo_baselines_values # k,128,1,30,1
                    
                    # Variance across samples for each element
                    rloo_variance_per_element = torch.var(advantages, dim=0, unbiased=True)
                    
                    # Mean variance across valid elements (masked)
                    rloo_variance = (rloo_variance_per_element * mask).sum() / valid_element_count
                    
                    # Optionally: variance of the advantages weighted by log-probs
                    # This captures the actual variance of the gradient estimator
                    score_weights = score_samples_tensor.detach()  # shape: (k, batch_size, ...) # k,128,30
                    score_weights = rearrange(score_weights, 'k b n -> k b 1 n 1')
                    weighted_advantages = advantages * score_weights.sum(-2, keepdims=True)  # adjust dimensions as needed
                    weighted_variance = torch.var(weighted_advantages, dim=0, unbiased=True)
                    weighted_rloo_variance = (weighted_variance * mask).sum() / valid_element_count
            else:
                loo_baselines_values = None
                rloo_variance = torch.tensor(0.0)
                weighted_rloo_variance = torch.tensor(0.0)

            # Logging
            log_y_hat = y_hat_loss_samples_tensor.detach().mean(dim=0)
            self.train_metrics.update(log_y_hat, y, mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', pred_loss, batch_size=batch.batch_size)
            #if y_b is not None:
            #    self.log_loss('train_baseline', b_loss, batch_size=batch.batch_size)
            self.log_loss('graph', graph_loss, batch_size=batch.batch_size)

            # Log RLOO variance metrics
            if k > 1:
                #self.log('advantage_variance', rloo_variance, batch_size=batch.batch_size)
                self.log('estimator_variance', weighted_rloo_variance, batch_size=batch.batch_size)
                
                # Optional: log additional statistics
                #mean_advantage = (advantages * mask).sum() / valid_element_count
                #self.log('mean_advantage', mean_advantage.abs(), batch_size=batch.batch_size)

            # For checking if the model is being trained.
            print('Prediction network gradients:')
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f'{name}: grad_norm = {param.grad.norm().item()}')
                else:
                    print(f'{name}: NO GRADIENT')

            return pred_loss + self.sf_weight * graph_loss
            
        if self.use_baseline == 'doublecv':
            # ===== SAMPLING LOOP =====
            # This stays in the predictor since it needs access to self.predict_batch()
            y_hat_loss_mc_samples = []
            score_mc_samples = []
            pred_loss_mc_samples = []
            difference_mc_samples = []
            grads_mc_samples = []
            edge_logits = None

            for i in range(k):
                connectivity_i = self.predict_batch(batch,
                                                    preprocess=False,
                                                    postprocess=False,
                                                    mode='graph_only')

                sampled_adj_i = connectivity_i['adj'] # (128,30,30)

                if edge_logits is None:
                    edge_logits = connectivity_i['logits'] # (128,30,30)

                difference_i = sampled_adj_i - edge_logits # (128,30,30)
                difference_mc_samples.append(difference_i.detach())

                mean_graph = connectivity_i.pop('mean_graph')
                score_mc_samples.append(connectivity_i['ll'])

                connectivity_i['edge_weight'].requires_grad_()

                y_hat_scaled_i = self.predict_batch(batch,
                                                    preprocess=False,
                                                    postprocess=False,
                                                    mode='pred_only',
                                                    connectivity=connectivity_i)

                y_hat_i = batch.transform['y'].inverse_transform(y_hat_scaled_i)
                y_scaled = batch.transform['y'].transform(y)

                # Scale target and output
                if self.scale_target:
                    y_hat_loss_i = y_hat_scaled_i
                    y_loss_i = y_scaled
                else:
                    y_hat_loss_i = y_hat_i
                    y_loss_i = y

                y_hat_loss_mc_samples.append(y_hat_loss_i)

                # Compute elementwise MAE loss
                pred_loss_i = mae(y_hat_loss_i, y_loss_i, mask, reduction='none') # (128,1,30,1)

                # Compute gradient of loss w.r.t. edge weights for control variates
                if self.gradient_level == 'graph_level':
                    output = mae(y_hat_loss_i, y_loss_i, mask, reduction='none').sum(dim=2, keepdim=True) # (128,1,1,1)
                elif self.gradient_level == 'node_level':
                    output = mae(y_hat_loss_i, y_loss_i, mask, reduction='none') # (128,1,30,1)
                elif self.gradient_level == 'batch_level':
                    output = mae(y_hat_loss_i, y_loss_i, mask, reduction='mean') # Scalar

                f_prime_x = torch.autograd.grad(outputs=output,
                                                inputs=connectivity_i['edge_weight'],
                                                grad_outputs=torch.ones_like(output),
                                                retain_graph=True)[0]

                batch_v = torch.arange(128).repeat_interleave(y_loss_i.shape[2])
                f_prime_x_dense = to_dense_adj(connectivity_i['edge_index'].cpu(), batch=batch_v,
                                               edge_attr=f_prime_x.cpu(), max_num_nodes=y_loss_i.shape[2])
                f_prime_x_dense = f_prime_x_dense.to('cuda:0')
                grads_mc_samples.append(f_prime_x_dense.detach())

                pred_loss_mc_samples.append(pred_loss_i)

            # Stack tensors
            difference_mc_samples_tensor = torch.stack(difference_mc_samples, dim=0) # k, 128, 30, 30
            y_hat_loss_samples_tensor = torch.stack(y_hat_loss_mc_samples, dim=0)
            score_samples_tensor = torch.stack(score_mc_samples, dim=0)
            pred_loss_samples_tensor = torch.stack(pred_loss_mc_samples, dim=0) # (k, 128, 1, 30, 1)
            grads_mc_samples_tensor = torch.stack(grads_mc_samples, dim=0) # k, 128, 30, 30

            # ===== DELEGATE TO DoubleCVEstimator =====
            graph_loss, pred_loss, metrics = self.doublecv_estimator(
                costs=pred_loss_samples_tensor,
                grads=grads_mc_samples_tensor,
                scores=score_samples_tensor,
                edge_logits=edge_logits,
                differences=difference_mc_samples_tensor,
                mask=mask,
                batch_size=128
            )

            # ===== LOGGING =====
            self.train_metrics.update(y_hat_loss_samples_tensor.detach().mean(dim=0), y, mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', pred_loss, batch_size=batch.batch_size)
            self.log_loss('graph', graph_loss, batch_size=batch.batch_size)
            self.log('alpha', self.doublecv_estimator.alpha, batch_size=batch.batch_size)

            # Log correlations if available
            if 'corr_fxk_globalcv' in metrics:
                self.log('corr_fxk_globalcv', metrics['corr_fxk_globalcv'], batch_size=batch.batch_size)
            if 'corr_objective_baseline' in metrics:
                self.log('corr_objective_baseline', metrics['corr_objective_baseline'], batch_size=batch.batch_size)

            # Log alpha gradient if available
            if self.doublecv_estimator.alpha.grad is not None:
                self.log('alpha_grad_norm', self.doublecv_estimator.alpha.grad.norm(), batch_size=batch.batch_size)

            # Log variance metrics
            if k > 1:
                self.log('advantage_variance', metrics['advantage_variance'], batch_size=batch.batch_size)
                self.log('estimator_variance', metrics['estimator_variance'], batch_size=batch.batch_size)
                self.log('mean_advantage', metrics['mean_advantage'].abs(), batch_size=batch.batch_size)
                self.log('advantage_snr', metrics['snr'], batch_size=batch.batch_size)

            return pred_loss + self.sf_weight * graph_loss

        if self.use_baseline == 'muprop':
            connectivity = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='graph_only')
            
            sampled_adj = connectivity['adj'] # x in Eq (3)
            edge_logits = connectivity['logits'] # xbar
            logits_edge_index, logits_edge_weights = adj_to_edge_index(edge_logits) # x_bar as pyg edge index
            difference = sampled_adj - edge_logits # x-xbar
            difference_edge_index, difference_edge_index_weights = adj_to_edge_index(difference) # x-x_bar as pyg edge index

            mean_graph = connectivity.pop('mean_graph')
            fc_edge_index, fc_edge_weight = connectivity.pop('fc_edge_index'), connectivity.pop('fc_edge_weight')

            y_hat_scaled = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='pred_only',
                                            connectivity=connectivity)

            connectivity_meanfield = dict(edge_index=logits_edge_index, edge_weight=logits_edge_weights, adj=None, disjoint=True)

            y_hat_scaled_meanfield = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='pred_only',
                                            connectivity=connectivity_meanfield)
             
            y_hat = batch.transform['y'].inverse_transform(y_hat_scaled)
            y_hat_meanfield = batch.transform['y'].inverse_transform(y_hat_scaled_meanfield)
            y_scaled = batch.transform['y'].transform(y)
                
            # Scale target and output, eventually
            if self.scale_target:
                y_hat_loss = y_hat_scaled
                y_hat_loss_meanfield = y_hat_scaled_meanfield
                y_loss = y_scaled
            else:
                y_hat_loss = y_hat
                y_hat_loss_meanfield = y_hat_meanfield                
                y_loss = y

            # Compute loss

            # Prediction loss
            #pred_loss = self.loss_fn(y_hat_loss, y_loss, mask) # f(x) in Eq (3)
            pred_loss_elementwise = mae(y_hat_loss, y_loss, mask, reduction='none') # Shape(128,1,30,1)/(Batch size, Horizon, Node count, Feature per node)
            pred_loss = pred_loss_elementwise.mean() # Mean of f(x) over all batches and nodes. This is the pure forecasting loss. Scalar.
            pred_loss_perbatch = pred_loss_elementwise.mean(dim=(1,2,3)) # Mean of f(x) per every batch. Shape: (128,)
           
            # MuProp Loss
            # Meanfield Prediction Loss
            #pred_loss_meanfield = self.loss_fn(y_hat_loss_meanfield, y_loss, mask) # f(xbar) in Eq (3)
            pred_loss_meanfield_elementwise = mae(y_hat_loss_meanfield, y_loss, mask, reduction='none') # Shape(128,1,30,1)/(Batch size, Horizon, Node count, Feature per node)
            pred_loss_meanfield = pred_loss_meanfield_elementwise.mean() # Mean of f(xbar) over all batches and nodes. Scalar.
            pred_loss_meanfield_perbatch = pred_loss_meanfield_elementwise.mean(dim=(1,2,3)) # Mean of f(xbar) per every batch. Shape: (128,)
           
            # Taylor expansion term / Baseline
            f_prime_xbar = torch.autograd.grad(pred_loss_meanfield, edge_logits, retain_graph=True)[0] # f'(x_bar) in Eq (3). Shape: (128,30,30)
            #taylor_exp_term = pred_loss_meanfield_perbatch.detach() + (f_prime_xbar.detach() * difference.detach()).mean(dim=(1,2)) # f(x_bar) + f'(x_bar) * (x - x_bar) in Eq3. Shape: (128,)

            # Taylor expansion term / Baseline (DEEPSEEK)
            taylor_component = (f_prime_xbar.detach() * difference.detach()).sum(dim=(1,2))  # Sum over graph edges Shape: (128)
            print('shape of taylor component:', taylor_component.shape)
            taylor_component_expanded = taylor_component.view(-1, 1, 1, 1).expand_as(pred_loss_elementwise)  # Shape:(128, 1, 30, 1)
            print('shape of taylor component expanded:', taylor_component_expanded.shape)
            taylor_exp_term = pred_loss_meanfield_elementwise.detach() + taylor_component_expanded  # (128, 1, 30, 1)
            print('shape of taylor_exp_term:', taylor_exp_term.shape)

            # Residual Component
            #score = connectivity['ll'].mean(dim=1) # Mean log probs over batches. Shape: (128,30) > (128,)
            #residual_component_unscaled = pred_loss_perbatch.detach() - taylor_exp_term # f(x) - taylorexp(f(x)). Shape: (128,) - (128,) = (128,)
            #residual_component = (score * residual_component_unscaled).mean() # Elementwise multiplication followed by mean (128,) * (128,) > Scalar. Residual component is detached from the computation graph.
            
            # Residual Component (DEEPSEEK)
            score = connectivity['ll'] # (128,30)
            print('shape of score before expansion:', score.shape)
            residual_component_unscaled = pred_loss_elementwise.detach() - taylor_exp_term  # (128, 1, 30, 1)
            print('shape of residual_component_unscaled:', residual_component_unscaled.shape)
            #score_expanded = score.view(-1, 1, 1, 1).expand_as(residual_component_unscaled)  
            score_expanded = rearrange(score, 'b n -> b 1 n 1') # (128, 1, 30, 1)
            print('shape of score_expanded:', score_expanded.shape)
            total_elements = mask.sum() # 128*30 = 3840
            print('mask.sum():', total_elements)
            #residual_component = (score_expanded * residual_component_unscaled).sum() / total_elements
            residual_component = (score_expanded.sum(-2, keepdims=True) * residual_component_unscaled).sum() / total_elements

            # I think score_expanded.(-2, keepdims=True) should be added. Cini uses a global aggregated score, 
            # summed over all the nodes in the graph, then multiplies this score elementwise with the cost. Ie the original shapes are
            # (128,1,1,1) * (128,1,30,1), not (128,1,30,1) * (128,1,30,1). The nominator corresponds to self.value variable in the original
            # paper, and denominator corresponds to self.numel variable in the original paper. So the rest is correct. But inside the 
            # paranthesis I currently multiply local scores with locals costs, instead of global score with local costs. This corresponds to
            # the first term in the surrogate loss: "val = cost * score + lam * cost * score.sum(-2, keepdims=True)", while in reality 
            # it should have been equal to the second term in the surrogate loss, without the lam weight.

            # Mean Field Component / Correction Term
            #meanfield_component = (f_prime.detach() * edge_logits.detach() * (1 - edge_logits.detach())).mean() # OR: pred_loss_meanfield only, since loss.backward will implicitly calculate this term.
            #meanfield_component = (f_prime_xbar.detach() * edge_logits).mean() # Only edge logits are connected to the computation graph. Autodiff should handle their grads. Scalar.
            
            # Mean Field Component / Correction Term (DEEPSEEK)
            #meanfield_component = (f_prime_xbar.detach() * edge_logits).sum() / (edge_logits.numel()) # Divides by 128*30*30. Total num of elements in edge logits tensor
            meanfield_component = (f_prime_xbar.detach() * edge_logits).sum() / total_elements # Divides by 128*30. Total num of elements in mask.
            print('shape of edge_logits:', edge_logits.shape) # (128,30,30)
            print('shape of meanfield_component:', meanfield_component.shape)

            # Final Graph Loss
            graph_loss = residual_component + meanfield_component

            self.train_metrics.update(y_hat.detach(), y, mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', pred_loss, batch_size=batch.batch_size)
            self.log_loss('graph', graph_loss, batch_size=batch.batch_size)

            self.log_loss('residual_component_unscaled', residual_component_unscaled.mean(), batch_size=batch.batch_size)
            self.log_loss('residual_component', residual_component, batch_size=batch.batch_size)
            self.log_loss('score', score.mean(), batch_size=batch.batch_size)
            #self.log_loss('difference', difference.mean(), batch_size=batch.batch_size)
            self.log_loss('meanfield_component', meanfield_component, batch_size=batch.batch_size)
           
            return pred_loss + self.sf_weight * graph_loss

        if self.use_baseline == 'frechet':

            connectivity = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='graph_only')

            mean_graph = connectivity.pop('mean_graph')

            y_hat_scaled = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='pred_only',
                                            connectivity=connectivity,
                                            create_graph=True)
            
            y_hat = batch.transform['y'].inverse_transform(y_hat_scaled)
            y_scaled = batch.transform['y'].transform(y)

            with torch.no_grad():

                edge_index = mean_graph['edge_index']
                edge_weight = mean_graph['edge_weight']

                conn = dict(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    adj=None,
                    disjoint=True
                )
                y_b = self.predict_batch(batch,
                                        preprocess=False,
                                        postprocess=not self.scale_target,
                                        forward_kwargs=dict(
                                            mode='pred_only',
                                            connectivity=conn
                                        ))

            # Scale target and output, eventually
            if self.scale_target:
                y_hat_loss = y_hat_scaled
                y_loss = y_scaled
            else:
                y_hat_loss = y_hat                
                y_loss = y

            # Compute loss
            # Prediction loss
            pred_loss = self.loss_fn(y_hat_loss, y_loss, mask)

            if y_b is not None:
                b_loss = self.loss_fn(y_b, y_loss, mask)
            
            # Graph loss
            score = connectivity['ll']

            with torch.no_grad():
                b_cost, _ = self.sf_loss.compute_cost(y_b, y_loss, mask)

            # Reset metric state before call to prevent accumulation across batches
            self.sf_loss.reset()
            graph_loss = self.sf_loss(score=score,
                                    y_hat=y_hat_loss.detach(),
                                    y=y_loss,
                                    baseline=b_cost,
                                    mask=mask)

            # Logging
            self.train_metrics.update(y_hat.detach(), y, mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', pred_loss, batch_size=batch.batch_size)
            if y_b is not None:
                self.log_loss('train_baseline', b_loss, batch_size=batch.batch_size)
            self.log_loss('graph', graph_loss, batch_size=batch.batch_size)

            print('Prediction network gradients:')
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f'{name}: grad_norm = {param.grad.norm().item()}')
                else:
                    print(f'{name}: NO GRADIENT')

            if self.graph_module.edge_scorer.logits.grad is not None:
                grad_logits = self.graph_module.edge_scorer.logits.grad
                print('SHAPE OF Gradient for Logits:', grad_logits.shape)
                print('Graph module edge_scorer.logits grad_norm:', grad_logits.norm().item())
            else:
                print('Graph module edge_scorer.logits: NO GRADIENT')

            return pred_loss + self.sf_weight * graph_loss
            
        if self.use_baseline == 'reinforce':

            connectivity = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='graph_only')

            mean_graph = connectivity.pop('mean_graph')

            y_hat_scaled = self.predict_batch(batch,
                                            preprocess=False,
                                            postprocess=False,
                                            mode='pred_only',
                                            connectivity=connectivity)
            
            y_hat = batch.transform['y'].inverse_transform(y_hat_scaled)
            y_scaled = batch.transform['y'].transform(y)

            # Scale target and output, eventually
            if self.scale_target:
                y_hat_loss = y_hat_scaled
                y_loss = y_scaled
            else:
                y_hat_loss = y_hat                
                y_loss = y

            # Compute loss
            # Prediction loss
            pred_loss = self.loss_fn(y_hat_loss, y_loss, mask)

            # Graph loss
            score = connectivity['ll']
            y_b = None
            # Reset metric state before call to prevent accumulation across batches
            self.sf_loss.reset()
            graph_loss = self.sf_loss(score=score,
                                    y_hat=y_hat_loss.detach(),
                                    y=y_loss,
                                    baseline=y_b,
                                    mask=mask)

            #print('pred_loss requires grad:', pred_loss.requires_grad)
            #print('graph_loss requires grad:', graph_loss.requires_grad)

            # Logging
            self.train_metrics.update(y_hat.detach(), y, mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', pred_loss, batch_size=batch.batch_size)
            self.log_loss('graph', graph_loss, batch_size=batch.batch_size)

            if self.model_has_trainable_params():
                return pred_loss + self.sf_weight * graph_loss
            else:
                return self.sf_weight * graph_loss

    def model_has_trainable_params(self):
        return any(p.requires_grad for p in self.model.parameters())

class IMLEGraphPredictor(LatentGraphPredictor):
    def __init__(self,
                 graph_module_class,
                 graph_module_kwargs,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scale_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 lambda_val=1.0,
                 mc_samples=1,
                 eval_mode='mode',
                 clip_grad=False,
                 clip_grad_val=0.5):

        # Ensure we use IMLE components
        #graph_module_kwargs.update({
        #    'lambda_val': lambda_val,
        #    'nb_samples': mc_samples
        #})

        super().__init__(graph_module_class=graph_module_class,
                         graph_module_kwargs=graph_module_kwargs,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs,
                         mc_samples=mc_samples,
                         eval_mode=eval_mode)

        self.training_step_outputs = []
        self.imle_stats_history = []
        self.automatic_optimization = False
        self.clip_grad = clip_grad
        self.clip_grad_val = clip_grad_val

    def training_step(self, batch, batch_idx):  
        optimizer = self.optimizers() # IF MANUAL OPTIMIZATION

        y = batch.y
        mask = batch.mask

        # 1. Sample graph using IMLE 
        # This creates: edge_scorer.logits -> IMLEFunction -> sampled graph
        connectivity = self.predict_batch(batch,
                                          preprocess=False,
                                          postprocess=False,
                                          mode='graph_only')
        logits = connectivity['logits']
        print(f"Graph logits requires_grad: {connectivity['logits'].requires_grad}")

        # 2. Forward pass through forecasting model with sampled graph
        y_hat_scaled = self.predict_batch(batch,
                                          preprocess=False,
                                          postprocess=False,
                                          mode='pred_only',
                                          connectivity=connectivity)

        y_hat = batch.transform['y'].inverse_transform(y_hat_scaled)
        y_scaled = batch.transform['y'].transform(y)

        # 3. Compute loss - standard PyTorch loss
        if self.scale_target:
            y_hat_loss = y_hat_scaled
            y_loss = y_scaled
        else:
            y_hat_loss = y_hat
            y_loss = y

        pred_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Manual Backprop for Logits Goes Here
        # ...
        optimizer.zero_grad()
        self.manual_backward(pred_loss)

        # Apply gradient clipping if enabled (manual optimization requires explicit clipping)
        if self.clip_grad:
            self.clip_gradients(optimizer, gradient_clip_val=self.clip_grad_val, gradient_clip_algorithm='value')

        optimizer.step()


        imle_stats = self.graph_module.sampler.sampler.get_stats()
        print('PREDICTOR IMLE STATS:', imle_stats)

        with torch.no_grad():
            # Graph statistics
            adj = connectivity['adj']
            graph_density = adj.mean().item() # Mean Value of MAP(Logits+Noise)
            graph_std = adj.std().item() # Std of MAP(Logits+Noise)
            
            # Parameter statistics
            logits_mean = logits.mean().item() # Mean Value of Pure Logits
            logits_std = logits.std().item() # Std of Pure Logits
            
            # Gradient monitoring (will be available after backward)
            if self.graph_module.edge_scorer.logits.grad is not None:
                logits_grad_norm = self.graph_module.edge_scorer.logits.grad.norm().item() # Norm of gradient returned to Logits
                grad_logits = self.graph_module.edge_scorer.logits.grad
                print('SHAPE OF Gradient for Logits:', grad_logits.shape)
                print('Gradient for Logits:', logits_grad_norm)
                
                # Also check model gradients
                model_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        model_grad_norm += param.grad.norm().item()
                print(f"Model gradient norm: {model_grad_norm:.6f}")

                # Compute what the sum would be
                #gradient_sum = imle_stats['grad_sum']  # Sum across batch dimension
                #print(f"Gradient sum across batch shape: {gradient_sum.shape}")  # [30, 30]
                
                # Check if this matches the actual gradient
                #diff = (gradient_sum - self.graph_module.edge_scorer.logits.grad).abs().max()
                #print(f"Max difference between sum and actual gradient: {diff.item()}")
                
                # Check gradient magnitude
                #print(f"Batch gradient norm per sample: {gradient.norm(dim=(1,2)).mean().item():.6f}")
                #print(f"Total gradient norm (summed): {self.graph_module.edge_scorer.logits.grad.norm().item():.6f}")
            else:
                logits_grad_norm = 0.0
                print("Warning: No edge scorer gradients!")

        step_metrics = {
            'pred_loss': pred_loss,
            'graph_density': graph_density,
            'graph_std': graph_std,
            'logits_mean': logits_mean,
            'logits_std': logits_std,
            'logits_grad_norm': logits_grad_norm,
            'imle_stats': imle_stats or {}
        }
        self.training_step_outputs.append(step_metrics)
        self._log_training_metrics(step_metrics, batch.batch_size)
        
        # Logging
        #self.train_metrics.update(y_hat.detach(), y, mask)
        #self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        #self.log_loss('train', pred_loss, batch_size=batch.batch_size)

        # Check if gradients were computed
        if self.graph_module.edge_scorer.logits.grad is not None:
            print("SUCCESS: I-MLE gradients computed!")
            print(f"IMLE LOGITS Gradient norm: {self.graph_module.edge_scorer.logits.grad.norm()}")
        else:
            print("ERROR: No gradients computed!")
        
        return pred_loss

    def configure_optimizers(self):
        #Standard optimizer configuration - works seamlessly!
        # All parameters (both model and graph) will get proper gradients
        model_params = list(self.model.parameters())
        graph_params = list(self.graph_module.parameters())
        
        all_params = model_params + graph_params
        
        optimizer = self.optim_class(all_params, **self.optim_kwargs)
        
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            return [optimizer], [scheduler]
        
        return optimizer
    
    def _log_training_metrics(self, metrics, batch_size):
        # Log standard metrics
        self.log_loss('train', metrics['pred_loss'], batch_size=batch_size)
        
        # Log graph statistics
        self.log('train/graph_density', metrics['graph_density'], batch_size=batch_size)
        self.log('train/graph_std', metrics['graph_std'], batch_size=batch_size)
        self.log('train/logits_mean', metrics['logits_mean'], batch_size=batch_size)
        self.log('train/logits_std', metrics['logits_std'], batch_size=batch_size)
        self.log('train/logits_grad_norm', metrics['logits_grad_norm'], batch_size=batch_size)
        
        # Log I-MLE specific metrics
        imle_stats = metrics.get('imle_stats', {})
        #print(imle_stats)
        for key, value in imle_stats.items():
            self.log(f'train/imle_{key}', value, batch_size=batch_size)

    def on_train_epoch_end(self):
        # Compute epoch averages
        if not self.training_step_outputs:
            return
            
        epoch_metrics = {}
        for key in self.training_step_outputs[0].keys():
            if key != 'imle_stats':
                values = [step[key] for step in self.training_step_outputs]
                epoch_metrics[key] = sum(values) / len(values)
        
        # Log epoch averages
        for key, value in epoch_metrics.items():
            self.log(f'epoch/{key}', value, prog_bar=True)
        
        # Clear for next epoch
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # Regular validation
        result = super().validation_step(batch, batch_idx)
        
        # Additional I-MLE monitoring on validation
        with torch.no_grad():
            connectivity = self.predict_batch(batch, mode='graph_only')
            adj = connectivity['adj']
            val_density = adj.mean().item()
            self.log('val/graph_density', val_density, batch_size=batch.batch_size)
        
        return result
    
class AIMLEGraphPredictor(LatentGraphPredictor):
    def __init__(self,
                 graph_module_class,
                 graph_module_kwargs,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scale_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 mc_samples=1,
                 eval_mode='mode',
                 clip_grad=False,
                 clip_grad_val=0.5):

        super().__init__(graph_module_class=graph_module_class,
                         graph_module_kwargs=graph_module_kwargs,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs,
                         mc_samples=mc_samples,
                         eval_mode=eval_mode)

        self.training_step_outputs = []
        self.aimle_stats_history = []
        self.automatic_optimization = False
        self.clip_grad = clip_grad
        self.clip_grad_val = clip_grad_val

    def training_step(self, batch, batch_idx):  
        optimizer = self.optimizers() # IF MANUAL OPTIMIZATION

        y = batch.y
        mask = batch.mask

        # 1. Sample graph using IMLE 
        # This creates: edge_scorer.logits -> IMLEFunction -> sampled graph
        connectivity = self.predict_batch(batch,
                                          preprocess=False,
                                          postprocess=False,
                                          mode='graph_only')
        logits = connectivity['logits']
        print(f"Graph logits requires_grad: {connectivity['logits'].requires_grad}")

        # 2. Forward pass through forecasting model with sampled graph
        y_hat_scaled = self.predict_batch(batch,
                                          preprocess=False,
                                          postprocess=False,
                                          mode='pred_only',
                                          connectivity=connectivity)

        y_hat = batch.transform['y'].inverse_transform(y_hat_scaled)
        y_scaled = batch.transform['y'].transform(y)

        # 3. Compute loss - standard PyTorch loss
        if self.scale_target:
            y_hat_loss = y_hat_scaled
            y_loss = y_scaled
        else:
            y_hat_loss = y_hat
            y_loss = y

        pred_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Manual Backprop for Logits
        optimizer.zero_grad()
        self.manual_backward(pred_loss)

        # Apply gradient clipping if enabled
        if self.clip_grad:
            self.clip_gradients(optimizer, gradient_clip_val=self.clip_grad_val, gradient_clip_algorithm='value')

        optimizer.step()


        aimle_stats = self.graph_module.sampler.sampler.get_stats()
        print('PREDICTOR AIMLE STATS:', aimle_stats)

        with torch.no_grad():
            # Graph statistics
            adj = connectivity['adj']
            graph_density = adj.mean().item() # Mean Value of MAP(Logits+Noise)
            graph_std = adj.std().item() # Std of MAP(Logits+Noise)
            
            # Parameter statistics
            logits_mean = logits.mean().item() # Mean Value of Pure Logits
            logits_std = logits.std().item() # Std of Pure Logits
            
            # Gradient monitoring (will be available after backward)
            if self.graph_module.edge_scorer.logits.grad is not None:
                logits_grad_norm = self.graph_module.edge_scorer.logits.grad.norm().item() # Norm of gradient returned to Logits
                grad_logits = self.graph_module.edge_scorer.logits.grad
                print('SHAPE OF Gradient for Logits:', grad_logits.shape)
                print('Gradient for Logits:', logits_grad_norm)
                
                # Also check model gradients
                model_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        model_grad_norm += param.grad.norm().item()
                print(f"Model gradient norm: {model_grad_norm:.6f}")

                # Compute what the sum would be
                #gradient_sum = imle_stats['grad_sum']  # Sum across batch dimension
                #print(f"Gradient sum across batch shape: {gradient_sum.shape}")  # [30, 30]
                
                # Check if this matches the actual gradient
                #diff = (gradient_sum - self.graph_module.edge_scorer.logits.grad).abs().max()
                #print(f"Max difference between sum and actual gradient: {diff.item()}")
                
                # Check gradient magnitude
                #print(f"Batch gradient norm per sample: {gradient.norm(dim=(1,2)).mean().item():.6f}")
                #print(f"Total gradient norm (summed): {self.graph_module.edge_scorer.logits.grad.norm().item():.6f}")
            else:
                logits_grad_norm = 0.0
                print("Warning: No edge scorer gradients!")

        step_metrics = {
            'pred_loss': pred_loss,
            'graph_density': graph_density,
            'graph_std': graph_std,
            'logits_mean': logits_mean,
            'logits_std': logits_std,
            'logits_grad_norm': logits_grad_norm,
            'aimle_stats': aimle_stats or {}
        }
        self.training_step_outputs.append(step_metrics)
        self._log_training_metrics(step_metrics, batch.batch_size)
        
        # Logging
        #self.train_metrics.update(y_hat.detach(), y, mask)
        #self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        #self.log_loss('train', pred_loss, batch_size=batch.batch_size)

        # Check if gradients were computed
        if self.graph_module.edge_scorer.logits.grad is not None:
            print("SUCCESS: AIMLE gradients computed!")
            print(f"AIMLE LOGITS Gradient norm: {self.graph_module.edge_scorer.logits.grad.norm()}")
        else:
            print("ERROR: No gradients computed!")
        
        return pred_loss

    def configure_optimizers(self):
        #Standard optimizer configuration - works seamlessly!
        # All parameters (both model and graph) will get proper gradients
        model_params = list(self.model.parameters())
        graph_params = list(self.graph_module.parameters())
        
        all_params = model_params + graph_params
        
        optimizer = self.optim_class(all_params, **self.optim_kwargs)
        
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            return [optimizer], [scheduler]
        
        return optimizer
    
    def _log_training_metrics(self, metrics, batch_size):
        # Log standard metrics
        self.log_loss('train', metrics['pred_loss'], batch_size=batch_size)
        
        # Log graph statistics
        self.log('train/graph_density', metrics['graph_density'], batch_size=batch_size)
        self.log('train/graph_std', metrics['graph_std'], batch_size=batch_size)
        self.log('train/logits_mean', metrics['logits_mean'], batch_size=batch_size)
        self.log('train/logits_std', metrics['logits_std'], batch_size=batch_size)
        self.log('train/logits_grad_norm', metrics['logits_grad_norm'], batch_size=batch_size)
        
        # Log I-MLE specific metrics (only numeric values)
        imle_stats = metrics.get('aimle_stats', {})
        for key, value in imle_stats.items():
            if isinstance(value, (int, float)) or (hasattr(value, 'item') and value.numel() == 1):
                self.log(f'train/aimle_{key}', value, batch_size=batch_size)

    def on_train_epoch_end(self):
        # Compute epoch averages
        if not self.training_step_outputs:
            return
            
        epoch_metrics = {}
        for key in self.training_step_outputs[0].keys():
            if key != 'aimle_stats':
                values = [step[key] for step in self.training_step_outputs]
                epoch_metrics[key] = sum(values) / len(values)
        
        # Log epoch averages
        for key, value in epoch_metrics.items():
            self.log(f'epoch/{key}', value, prog_bar=True)
        
        # Clear for next epoch
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # Regular validation
        result = super().validation_step(batch, batch_idx)
        
        # Additional I-MLE monitoring on validation
        with torch.no_grad():
            connectivity = self.predict_batch(batch, mode='graph_only')
            adj = connectivity['adj']
            val_density = adj.mean().item()
            self.log('val/graph_density', val_density, batch_size=batch.batch_size)
        
        return result