

from cv2 import rotatedRectangleIntersection
import numpy as np
import torch


def get_section(x,d):
    """ 
    Args:
        x : points on the rays [...,3]
        d : direction of the rays [...,3]

    Returns:
        section : section of the rays (x,d) [...,3,4]
    """
    ####section_1#####
    theta, phi = get_angle(d)
    R = rotation_matrix(theta, phi)
    m = torch.cross(x,d, dim=-1)
    t = torch.cross(d,m, dim=-1)
    R[:,:,:,0] = -R[:,:,:,0]
    section = torch.cat([R,t.unsqueeze(-1)],dim=-1)
    return section

def get_section_2(x,d):
    """ 
    Args:
        x : upright camera_pose [...,3,4]
        d : direction of the rays [...,3]

    Returns:
        section : section of the rays (x,d) [...,3,4]
    """
    ####section_2#####
    point = x[...,:3,3]
    m = torch.cross(point,d)
    t = torch.cross(d,m)
    
    r_2 = x[...,:3,0]
    
    r_2 = torch.cross(d,r_2)
    r_1 = torch.cross(d,r_2)
    
    R = torch.stack([r_1,r_2,d],dim=-1)

    section = torch.cat([R,t.unsqueeze(-1)],dim=-1)
    return section

def get_angle(d):
    """
    Args:
        d: direction of ray [...,3]

    Returns:
        theta: [...]
        phi: [...]
    """
    mask = torch.bitwise_and((torch.abs(d[...,1])<1e-6) , (torch.abs(d[...,0])<1e-6))
    theta = torch.atan2(d[...,1],d[...,0]) #from zero to 2pi
    theta [mask] = 0
    phi = torch.atan2(torch.sqrt(d[...,0]**2+d[...,1]**2),d[...,2]) ### from zero to pi
    
    return theta, phi

def rotation_matrix(theta,phi):
    """_summary_

    Args:
        theta (_type_): [...]
        phi (_type_): [...]

    Returns:
        Rotation: [...,3,3]
    """
    

    R = torch.zeros(theta.shape+(3,3)).to(theta.device)
    R[...,0,0] = torch.cos(theta)
    R[...,0,1] = -torch.sin(theta)
    R[...,1,0] = torch.sin(theta)
    R[...,1,1] = torch.cos(theta)
    R[...,2,2] = 1
   
    R_2 = torch.zeros(phi.shape+(3,3)).to(phi.device)
    R_2[...,0,0] = torch.cos(phi)
    R_2[...,0,2] = torch.sin(phi)
    R_2[...,2,0] = -torch.sin(phi)
    R_2[...,2,2] = torch.cos(phi)
    R_2[...,1,1] = 1
    
    R = torch.matmul(R,R_2)
    
    return R

