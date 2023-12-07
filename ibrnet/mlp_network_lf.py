# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import Mult
from cmath import isnan
from glob import glob
import pdb
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
# This script still learn sigma and color, but in an equivariant way
    
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiType_fc(nn.Module):
    def __init__(self, scalar_list, tensor_list, feature_type,n_freq,act='norm'):
        super().__init__()
        self.scalar_channel = feature_type[0]
        self.tensor_channel = feature_type[1]
        self. scalar_fc = nn.ModuleList()
        assert scalar_list[0] == self.scalar_channel
        assert tensor_list[0] == self.tensor_channel
        self.final_scalar_channel = scalar_list[-1]
        self.final_tensor_channel = tensor_list[-1]
        
        for i in range(len(scalar_list) - 1):
            self.scalar_fc.append(nn.Linear(scalar_list[i], scalar_list[i + 1]))
            self.scalar_fc.append(nn.ELU(inplace=True))
        
        self.tensor_fc = nn.ModuleList()
        for i in range(len(tensor_list) - 1):
            self.tensor_fc.append(MultiType_Linear(tensor_list[i], tensor_list[i + 1],n_freq-1))
            #ipdb.set_trace()
            if act =='norm':
                self.tensor_fc.append(MultiType_NonLinear_Norm(tensor_list[i+1],n_freq-1))
            else:
                self.tensor_fc.append(MultiType_NonLinear_Gate(tensor_list[i+1],n_freq-1))
        
    def forward(self, x):
        scalar_x = x[..., :self.scalar_channel]
        tensor_x = x[..., self.scalar_channel:]
        
        tensor_x = tensor_x.reshape(tensor_x.shape[0], tensor_x.shape[1], -1,self.tensor_channel,2) 
        #[n_rays, n_samples, n_freq, n_channels, 2]
        tensor_x = tensor_x.permute(0, 1,2,4,3) #[n_rays, n_samples, n_freq, 2, n_channels]
        
        for fc in self.scalar_fc:
            scalar_x = fc(scalar_x)
            
        
        for fc in self.tensor_fc:
            tensor_x = fc(tensor_x)
            
        tensor_x = tensor_x.permute(0, 1,2,4,3) #[n_rays, n_samples, n_freq, n_channels, 2]
        tensor_x = tensor_x.reshape(tensor_x.shape[0], tensor_x.shape[1], -1) 
        #[n_rays, n_samples, n_freq*n_channels*2]
        x = torch.cat([scalar_x,tensor_x],dim=-1)
        return x, [self.final_scalar_channel, self.final_tensor_channel]


class MultiType_fc_2(nn.Module):
    def __init__(self, tensor_list,n_freq,act='norm'):
        super().__init__()
        
        
        self.final_tensor_channel = tensor_list[-1]
    
        self.tensor_fc = nn.ModuleList()
        for i in range(len(tensor_list) - 1):
            self.tensor_fc.append(MultiType_Linear(tensor_list[i], tensor_list[i + 1],n_freq))
            if act =='norm':
                self.tensor_fc.append(MultiType_NonLinear_Norm(tensor_list[i+1],n_freq))
            else:
                self.tensor_fc.append(MultiType_NonLinear_Gate(tensor_list[i+1],n_freq))
        
    def forward(self, x):
        tensor_x = x.permute(0, 1,2,4,3) #[n_rays, n_samples, n_freq, 2, n_channels]
        
        for fc in self.tensor_fc:
            tensor_x = fc(tensor_x)
            
        x = tensor_x.permute(0, 1,2,4,3) #[n_rays, n_samples, n_freq, n_channels, 2]
        
        return x
    
class MultiType_Linear(nn.Module):   
    def __init__(self, in_channel,out_channel,n_freq):
        super().__init__( )
        self.weight = nn.Parameter(torch.zeros(n_freq,in_channel,out_channel))
        nn.init.kaiming_normal_(self.weight.data)
    def forward(self, x):
        x = torch.einsum('nrfvc,fco->nrfvo', x, self.weight)
        return x 

    
class MultiType_NonLinear_Gate(nn.Module):
    def __init__(self, out_channel,n_freq):
        super().__init__()
        self.linear = MultiType_Linear(out_channel, out_channel,n_freq)
        self.beta = nn.Parameter(torch.ones(1,1,n_freq,out_channel,2))
        
    def forward(self, x):
        x_2 = self.linear(x)
        norm = complex_inv(x_2.permute(0,1,2,4,3),x_2.permute(0,1,2,4,3))[...,0]
        x =self.beta[...,0].unsqueeze(-2)*x + x * F.elu(norm.unsqueeze(-2))*self.beta[...,1].unsqueeze(-2)

        
        return x
    
class MultiType_NonLinear_Norm(nn.Module):
    def __init__(self, out_channel,n_freq):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1,1,n_freq,out_channel))
        
    def forward(self, x):
        norm = torch.norm(x, dim=-2)
        norm_2 = F.elu(norm-self.beta)
        x = x/(norm.unsqueeze(-2)+1e-5) * norm_2.unsqueeze(-2)
        return x

class MultiHeadAttention_lf(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, inv_dmodel, inv_d_k, inv_d_v, t_dmodel, t_d_k, t_d_v, n_freq, dropout=0.1):
        super().__init__()
 
        self.inv_dmodel = inv_dmodel
        self.inv_d_k = inv_d_k
        self.inv_d_v = inv_d_v
        self.t_dmodel = t_dmodel
        self.t_d_k = t_d_k
        self.t_d_v = t_d_v
        self.n_freq = n_freq
        self.n_head = n_head
        

        self.d_k =  (inv_d_k + t_d_k*n_freq) // n_head
        
        self.w_qs_inv = nn.Linear(inv_dmodel, inv_d_k*n_head, bias=False)
        self.w_qs = MultiType_Linear(t_dmodel,t_d_k*n_head,n_freq)
        
        self.w_ks_inv = nn.Linear(inv_dmodel, inv_d_k*n_head, bias=False)
        self.w_ks = MultiType_Linear(t_dmodel,t_d_k*n_head,n_freq)
        
        self.w_vs_inv = nn.Linear(inv_dmodel, inv_d_v*n_head, bias=False)
      
        
        
        self.fc_1_inv = nn.Linear(inv_d_v*n_head, inv_dmodel, bias=False)
     
        
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        
        
        self.layer_norm = nn.LayerNorm(inv_dmodel, eps=1e-6)
        
    def forward(self, inv, tensor, mask=None):

        d_k, d_v, n_head = self.inv_d_k, self.inv_d_v, self.n_head
        
        sz_b, len_q, len_k, len_v = inv.size(0), inv.size(1), inv.size(1), inv.size(1)


        inv_q = self.w_qs_inv(inv).view(sz_b, len_q, n_head, d_k)
        inv_k = self.w_ks_inv(inv).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs_inv(inv).view(sz_b, len_v, n_head, d_v)
        inv_res = inv
        
        
        tensor_q = (self.w_qs(tensor.transpose(-1,-2)).permute(0,1,4,2,3)).reshape(sz_b, len_q, n_head,-1)
        tensor_k = (self.w_ks(tensor.transpose(-1,-2)).permute(0,1,4,2,3)).reshape(sz_b, len_k, n_head,-1)
      
        
        #  Pass through the pre-attention projection: b x lq x (n*dv)*2
        # Separate different heads: b x lq x n x dv x 2*2
        q = torch.cat([inv_q,tensor_q],dim=-1)
        k = torch.cat([inv_k,tensor_k],dim=-1)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc_1_inv(q)
        q += inv_res

        q = self.layer_norm(q)


        return q, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn

def complex_mul(x,y):
    """
    x: [...,2]
    y: [...,2]
    """
    real = x[...,0]*y[...,0] - x[...,1]*y[...,1]
    imag = x[...,0]*y[...,1] + x[...,1]*y[...,0]
    return torch.stack([real,imag],dim=-1)

def complex_inv(x,y):
    """
    x: [...,2]
    y: [...,2]
    """
    real = x[...,0]*y[...,0] + x[...,1]*y[...,1]
    imag = x[...,0]*y[...,1] - x[...,1]*y[...,0]
    return torch.stack([real,imag],dim=-1)

# class EncoderLayer(nn.Module):
#     ''' Compose with two layers '''

#     def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
#         super(EncoderLayer, self).__init__()
#         self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
#         self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

#     def forward(self, enc_input, slf_attn_mask=None):
#         enc_output, enc_slf_attn = self.slf_attn(
#             enc_input, enc_input, enc_input, mask=slf_attn_mask)
#         enc_output = self.pos_ffn(enc_output)
#         return enc_output, enc_slf_attn


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var

@torch.jit.script
def fused_mean_variance_lf(x, weight):
    """
    x: [n_rays, n_samples, n_views, c,2]
    weight: [n_rays, n_samples, n_views,1]
    """
    mean = torch.sum(x*weight[...,None], dim=2, keepdim=True)
    cov = complex_mul(x-mean, x-mean)
    var = torch.sum(weight[...,None] * cov, dim=2, keepdim=True) ###Actually not very sure about this
    return mean, var



class IBRNet(nn.Module):
    def __init__(self, args, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        self.args = args
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(25, 32),
                                        activation_func,
                                        nn.Linear(32, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)
        
        self.ray_dir_fc_2 = nn.Sequential(nn.Linear(32, 32),
                                        activation_func,
                                        nn.Linear(32, 24),
                                        activation_func)


        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        act = args.act
        
       

        self.geometry_fc_lf = MultiType_fc_2([2,16,4],12,act=act)
        
        
                                                
        self.ray_attention_lf = MultiHeadAttention_lf(4, 16,4, 4, 4,1,1,12)
        

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)


        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+25, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.out_geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        
        

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table
    
    def encode_lf_np(self, d_hid, theta):
        """
        theta: [n_rays, n_samples, n_views]
        return: [n_rays, n_samples, n_views, d_hid//2, 2]
        """
        def get_position_angle_vec(position):
            pos_list = [np.zeros_like(position),np.zeros_like(position)]
            pos_list_2 = [position *2 * (hid_j // 2)  for hid_j in range(d_hid)]
            pos_list  = pos_list + pos_list_2
            return  pos_list
        
        sinusoid_table = get_position_angle_vec(theta)  #[d_hid, n_rays, n_samples, n_views]
        sinusoid_table[0::2,...] = np.sin(sinusoid_table[0::2,...])  # dim 2i
        sinusoid_table[1::2,...] = np.cos(sinusoid_table[1::2,...])  # dim 2i+1
        sinusoid_table = np.transpose(sinusoid_table, (1,2,3,0)) #[n_rays, n_samples, n_views, d_hid]
        sinusoid_table = np.reshape(sinusoid_table, sinusoid_table.shape[0:3]+(d_hid//2,2)) #[n_rays, n_samples, n_views, d_hid//2, 2]
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table
        
    # def encode_lf(self, d_hid, theta):
    #     """
    #     theta: [n_rays, n_samples, n_views]
    #     return: [n_rays, n_samples, n_views, d_hid//2, 2]
    #     """
    #     def get_position_angle_vec(position):
    #         pos_list = [torch.zeros_like(position),torch.zeros_like(position)]
    #         #pos_list_2 = [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid-2)]
    #         pos_list_2 = [position *(2 * (hid_j // 2)+1)  for hid_j in range(d_hid-2)]
    #         pos_list  = pos_list + pos_list_2
    #         return  pos_list
        
    #     sinusoid_table = get_position_angle_vec(theta)  
    #     sinusoid_table = torch.stack(sinusoid_table, dim=0)  #[d_hid, n_rays, n_samples, n_views]
    #     #sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
    #     sinusoid_table[0::2,...] = torch.sin(sinusoid_table[0::2,...])  # dim 2i
    #     sinusoid_table[1::2,...] = torch.cos(sinusoid_table[1::2,...])  # dim 2i+1
    #     sinusoid_table = sinusoid_table.permute(1,2,3,0) #[n_rays, n_samples, n_views, d_hid]
    #     sinusoid_table = sinusoid_table.reshape(sinusoid_table.shape[0:3]+(d_hid//2,2)) #[n_rays, n_samples, n_views, d_hid//2, 2]
    #     #sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
    #     return sinusoid_table

    def encode_lf(self, d_hid, theta):
        """
        theta: [n_rays, n_samples, n_views]
        return: [n_rays, n_samples, n_views, d_hid//2, 2]
        """
        def get_position_angle_vec(position):
            pos_list = [position *((hid_j // 2)+1)  for hid_j in range(d_hid)]
            return  pos_list
        sinusoid_table = get_position_angle_vec(theta)  
        sinusoid_table = torch.stack(sinusoid_table, dim=0)  #[d_hid, n_rays, n_samples, n_views]
        sinusoid_table[0::2,...] = torch.sin(sinusoid_table[0::2,...])  # dim 2i
        sinusoid_table[1::2,...] = torch.cos(sinusoid_table[1::2,...])  # dim 2i+1
        sinusoid_table = sinusoid_table.permute(1,2,3,0) #[n_rays, n_samples, n_views, d_hid]
        sinusoid_table = sinusoid_table.reshape(sinusoid_table.shape[0:3]+(d_hid//2,2)) #[n_rays, n_samples, n_views, d_hid//2, 2]
        return sinusoid_table
    
        
    def forward(self, rgb_feat, ray_diff, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        
        n_rays = rgb_feat.shape[0]
        n_samples = rgb_feat.shape[1]
        num_views = rgb_feat.shape[2]
        dot = ray_diff[...,[0]]
       
        ###get invariant "attention" and add "attention weight"(direction feat) to each feature#############
        plane_angle  = ray_diff[...,1]
 
        dir_pose = self.encode_lf(24,plane_angle) ###[n_rays, n_samples, n_views, 24//2, 2]
        mean = torch.mean(dir_pose, dim=2,  keepdim=True)
        plane_info = complex_inv(mean, dir_pose)  
        plane_info = plane_info.reshape(n_rays, n_samples, num_views, -1)
        ray_diff = torch.cat([dot,plane_info],dim=-1) 

        num_views = rgb_feat.shape[2]
        ###invariant "attention" doesn't depend on what section map we choose (different section has the same "ray_diff")
        direction_feat = self.ray_dir_fc(ray_diff)
        #####################################################################################
        
        rgb_in = rgb_feat[..., :3]
        
        rgb_feat = rgb_feat + direction_feat

        if self.anti_alias_pooling:
            dot_prod,_ = torch.split(ray_diff, [1, 24], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
            
        
        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)
        
        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
    
      
    
        vis = self.vis_fc2(x * vis) * mask 
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)
        
        mean,var = fused_mean_variance(x, weight)
        
        globalfeat = torch.cat([weight.mean(dim=2),var.squeeze(2), mean.squeeze(2)], dim=-1)  # [n_rays, n_samples, 1+32+128]

        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        
        ######################################################################################################################
        x_dir = self.ray_dir_fc_2(x) #[n_rays, n_samples, n_views, 12*2]
        x_dir = x_dir.reshape(x_dir.shape[0], x_dir.shape[1], x_dir.shape[2], dir_pose.shape[-2], -1) #[n_rays, n_samples, n_views, 12, 2]
        x_dir = x_dir[...,None]*dir_pose.unsqueeze(-2) ###[n_rays, n_samples, n_views, 12,2,2]
        
        
    
        mean_dir = torch.mean(x_dir*weight[...,None,None], dim =2)

        mean_dir = self.geometry_fc_lf(mean_dir) #[n_rays, n_samples, 12,4,2]

        globalfeat, _= self.ray_attention_lf(globalfeat,mean_dir, mask=(num_valid_obs > 1).float())   #[n_rays, n_samples, 16]

        #######################################################################################################################
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

