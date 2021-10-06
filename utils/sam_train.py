import torch
import torch.nn.functional as F
import random

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,dropout=0., **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.dropout = dropout

    @torch.no_grad()
    def rand_int_w(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) /(1-self.dropout)

            for p in group["params"]:
                init = torch.normal(0.0,1e-8,p.size())
                e_w = init * scale.to(p)
                p.add_(e_w)  
                self.state[p]["e_w"] = e_w


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        taylor_appro = 0
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) / (1-self.dropout)
            for p in group["params"]:
                p.requires_grad = True
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

                taylor_appro += (p.grad**2).sum()


        if zero_grad: self.zero_grad()
        return taylor_appro * scale.to(p)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > (1-self.dropout):
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def sstep(self,args,inputs,targets,loss_fct,model,defined_backward=None):
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        cutoff = int(len(targets) * args["nograd_cutoff"])
        if cutoff != 0:
            with torch.no_grad():
                logits_1 = model(inputs[:cutoff])
                l_before_1 = loss_fct(logits_1,targets[:cutoff])

        logits_2 = model(inputs[cutoff:])
        l_before_2 = loss_fct(logits_2,targets[cutoff:])
        loss = l_before_2
        logits = torch.cat((logits_1,logits_2),0)
        l_before = torch.cat((l_before_1,l_before_2.clone().detach()),0).detach()
        predictions = logits
        loss = loss.mean()
        defined_backward(loss)
        self.first_step(True)


        with torch.no_grad():
            l_after = loss_fct(model(inputs),targets)
            phase2_coeff = (l_after-l_before)/args["temperature"]
            coeffs = F.softmax(phase2_coeff).detach()

            #codes for sorting 
            prob = 1 - args["opt_dropout"] 
            if prob >=0.99:
                indices = range(len(targets))
            else:
                pp = int(len(coeffs) * prob)
                cutoff,_ = torch.topk(phase2_coeff,pp)
                cutoff = cutoff[-1]
                # cutoff = 0
                #select top k% 
                indices = [phase2_coeff > cutoff] 


        # second forward-backward step
        loss = loss_fct(model(inputs[indices]), targets[indices])
        return_loss = loss.clone().detach()
        loss = (loss * coeffs[indices]).sum()
        defined_backward(loss)
        self.second_step(True)
        return predictions,return_loss
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
