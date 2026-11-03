import torch
from torch import nn
from exp_3.networks import Transform as BaseTransform, cost
from exp_6.safety import barrier


class Transform(BaseTransform):
    def __init__(self,checkpoint):
        super().__init__(checkpoint)
        self.mean=self.mean[:52].clone()
        self.scale=self.scale[:52].clone()

    def history(self,levels):
        return (levels[...,1:,:52]-levels[...,:-1,:52]-self.mean[:52])/self.scale[:52]


def adapt_state(state,actor,hidden):
    converted={k:v.clone() for k,v in state.items()}
    converted['gru.weight_ih_l0']=converted['gru.weight_ih_l0'][:,:52].clone()
    if actor:
        converted['head.weight']=converted['head.weight'][:11].clone()
        converted['head.bias']=converted['head.bias'][:11].clone()
    else:
        converted['head.weight']=converted['head.weight'][:,:hidden+11].clone()
    return converted


class Actor(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();hidden=checkpoint['config']['hidden_size']
        self.gru=nn.GRU(52,hidden,batch_first=True);self.head=nn.Linear(hidden,11)

    def forward(self,history,last,transform):
        _,h=self.gru(history)
        return (last[...,41:52]+self.head(h[-1])*transform.scale[41:52]+transform.mean[41:52]).clamp(0,100)


class Dynamics(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();hidden=checkpoint['config']['hidden_size']
        self.gru=nn.GRU(52,hidden,batch_first=True);self.head=nn.Linear(hidden+11,41)

    def forward(self,history,last,action,transform):
        _,h=self.gru(history)
        delta=(action-last[...,41:52]-transform.mean[41:52])/transform.scale[41:52]
        return self.head(torch.cat((h[-1],delta),dim=-1))


def tracking_loss(predicted_cost, target_cost):
    # Penalize each transition: errors above/below target must not cancel.
    return (predicted_cost-target_cost).square().mean()


def actor_update(actor,model,transform,optimizer,batch,config):
    history,last,_,_,reference,lower,upper=batch
    optimizer.zero_grad(set_to_none=True);model.requires_grad_(False)
    try:
        action=actor(history,last,transform)
        cv=transform.cv_levels(last,model(history,last,action,transform))
        predicted_cost=cost(cv,transform.frequency)
        economic=predicted_cost.mean()
        tracking=tracking_loss(predicted_cost,config['target_cost'])
        window=((action-reference.detach())/config['regularization_scale']).square().mean()
        cv_barrier=barrier(cv,lower[:,:41],upper[:,:41],config['barrier_epsilon'])
        mv_barrier=barrier(action,lower[:,41:],upper[:,41:],config['barrier_epsilon'])
        regularization=config['regularization_weight']*window
        weighted_barrier=config['barrier_weight']*(cv_barrier+mv_barrier)
        loss=tracking+regularization+weighted_barrier
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite actor loss')
        loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    finally:model.requires_grad_(True)
    return {k:float(v.detach()) for k,v in dict(actor_loss=loss,actor_cost=economic,actor_tracking_loss=tracking,actor_penalty=window,
            actor_regularization=regularization,cv_barrier=cv_barrier,mv_barrier=mv_barrier,actor_barrier=weighted_barrier).items()}
