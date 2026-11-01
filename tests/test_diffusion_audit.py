import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusion.text_denoising_diffusion import GaussianDiffusion, ModelPrediction


class AuditTests(unittest.TestCase):
    def model(self):
        net=nn.Linear(4,4)
        net.latent_dim=4
        net.self_condition=False
        net.class_conditional=False
        obj=GaussianDiffusion(net,max_seq_len=2,objective='pred_v',loss_type='l2')
        obj.using_latent_model=True
        return obj

    def test_velocity_roundtrip_and_schedule(self):
        obj=self.model()
        t=torch.tensor([0., .1, .5, .9, 1.])
        alpha=obj.train_schedule(t)
        self.assertTrue(torch.all(alpha[1:]<=alpha[:-1]))
        self.assertTrue(torch.equal(alpha,obj.sampling_schedule(t)))
        x=torch.randn(5,2,4)
        noise=torch.randn_like(x)
        a=alpha[:,None,None]
        z=a.sqrt()*x+(1-a).sqrt()*noise
        v=a.sqrt()*noise-(1-a).sqrt()*x
        self.assertTrue(torch.allclose(obj.predict_start_from_v(z,t,v),x,atol=1e-6))
        self.assertTrue(torch.allclose(obj.predict_noise_from_v(z,t,v),noise,atol=1e-6))

    def test_dpmpp_returns_current_terminal_prediction(self):
        obj=self.model()
        obj.sampling_timesteps=1
        obj.diffusion_model_predictions=lambda z,*args,**kwargs: ModelPrediction(z*0,z*0+7,z*0)
        result,_=obj.dpmpp_sample((1,2,4),None,None,None,None,z_t=torch.ones(1,2,4))
        self.assertTrue(torch.equal(result,torch.full((1,2,4),7.)))

    def test_dpmpp_uses_previous_prediction(self):
        obj=self.model()
        pairs=[(torch.tensor([.8]),torch.tensor([.6])),
               (torch.tensor([.6]),torch.tensor([.4]))]
        obj.get_sampling_timesteps=lambda *args,**kwargs:pairs
        seen=[]
        def predict(z,*args,**kwargs):
            seen.append(z.clone())
            x=torch.ones_like(z)*len(seen)
            return ModelPrediction(z*0,x,z*0)
        obj.diffusion_model_predictions=predict
        result,_=obj.dpmpp_sample((1,2,4),None,None,None,None,z_t=torch.ones(1,2,4))
        a0,a1,a2=[obj.sampling_schedule(torch.tensor([t])) for t in (.8,.6,.4)]
        logs=[torch.log(a/(1-a))/2 for a in (a0,a1,a2)]
        h0,h1=logs[1]-logs[0],logs[2]-logs[1]
        denoised=2+(2-1)/(2*(h0/h1))
        expected=((1-a2)/(1-a1)).sqrt()*seen[1]-a2.sqrt()*torch.expm1(-h1)*denoised
        self.assertTrue(torch.allclose(result,expected))


if __name__=='__main__':
    unittest.main()
