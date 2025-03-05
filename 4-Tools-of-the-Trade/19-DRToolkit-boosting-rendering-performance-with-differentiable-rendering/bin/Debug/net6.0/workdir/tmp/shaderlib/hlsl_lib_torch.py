#pip install matplotlib numpy pyjion imageio PyOpenGL glfw
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

import sys
import os
import numpy as np
import torch
import pyjion #conflict with matplotlib
#pyjion.enable()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

s_gpu_tensor_cache = {}
vm_gpu_tensor_cache = {}

cpu_zeros_tensor_cache = {}
cpu_ones_tensor_cache = {}
gpu_zeros_tensor_cache = {}
gpu_ones_tensor_cache = {}

cpu_zero_tensor = torch.as_tensor(0.0, dtype=torch.float64)
gpu_zero_tensor = torch.as_tensor(0.0, dtype=torch.float64, device=device)
cpu_zero2_tensor = torch.as_tensor([0.0, 0.0], dtype=torch.float64)
gpu_zero2_tensor = torch.as_tensor([0.0, 0.0], dtype=torch.float64, device=device)

cpu_one_tensor = torch.as_tensor(1.0, dtype=torch.float64)
gpu_one_tensor = torch.as_tensor(1.0, dtype=torch.float64, device=device)

class TensorPools:
    def __init__(self) -> None:
        self.pools = {}
        self.allocs = {}
    def New(self, type, n, dim = None, dim2 = None):
        pools_by_type = self.pools.get(type)
        if pools_by_type is None:
            pools_by_type = {}
            self.pools[type] = pools_by_type
        if (dim is None or dim == 1) and dim2 is None:
            pool = pools_by_type.get(n)
            if pool is not None and len(pool) > 0:
                return pool.pop()
            t = torch.empty([n], dtype=type, device=device)
            self.Record(type, t, n, dim, dim2)
            return t
        elif dim2 is None:
            k = (n, dim)
            pool = pools_by_type.get(k)
            if pool is not None and len(pool) > 0:
                return pool.pop()
            t = torch.empty((n, dim), dtype=type, device=device)
            self.Record(type, t, n, dim, dim2)
            return t
        else:
            k = (n, dim, dim2)
            pool = pools_by_type.get(k)
            if pool is not None and len(pool) > 0:
                return pool.pop()
            t = torch.empty((n, dim, dim2), dtype=type, device=device)
            self.Record(type, t, n, dim, dim2)
            return t
    def Record(self, type, t, n, dim = None, dim2 = None):
        allocs_by_type = self.allocs.get(type)
        if allocs_by_type is None:
            allocs_by_type = {}
            self.allocs[type] = allocs_by_type
        if (dim is None or dim == 1) and dim2 is None:
            k = (n, 0, 0)
            allocs = allocs_by_type.get(k)
            if allocs is None:
                allocs = []
                allocs_by_type[k] = allocs
            allocs.append(t)
        elif dim2 is None:
            k = (n, dim, 0)
            allocs = allocs_by_type.get(k)
            if allocs is None:
                allocs = []
                allocs_by_type[k] = allocs
            allocs.append(t)
        else:
            k = (n, dim, dim2)
            allocs = allocs_by_type.get(k)
            if allocs is None:
                allocs = []
                allocs_by_type[k] = allocs
            allocs.append(t)
    def RecycleAll(self):
        for type, allocs in self.allocs.items():
            for ks, t in allocs.items():
                if ks[1] == 0 and ks[2] == 0:
                    k = ks[0]
                    pool = self.pools.get(k)
                    if pool is None:
                        pool = []
                        self.pools[k] = pool
                    for v in t:
                        pool.append(v)
                elif ks[2] == 0:
                    k = (ks[0], ks[1])
                    pool = self.pools.get(k)
                    if pool is None:
                        pool = []
                        self.pools[k] = pool
                    for v in t:
                        pool.append(v)
                else:
                    k = (ks[0], ks[1], ks[2])
                    pool = self.pools.get(k)
                    if pool is None:
                        pool = []
                        self.pools[k] = pool
                    for v in t:
                        pool.append(v)
        self.allocs.clear()

tensor_pools = TensorPools()

pool_N = {}

def poolGetN(num):
    arr = pool_N.get(num)
    if arr is None:
        arr = torch.arange(0, num, 1, dtype=torch.int64)
        pool_N[num] = arr
    return arr

def tuple_get_outparam(v, index):
    return v[index]
def tuple_get_retval(v):
    return v[0][0]
def tuple_get_value(v, index):
    return v[index]
def tuple_get_lastval(v):
    return v[len(v)-1]

def get_param_list(v, dim = None, dim2 = None):
    if (dim is None or dim == 1) and dim2 is None:
        if type(v) == torch.Tensor and len(v) > 4:
            return v
        else:
            return [v]
    elif dim2 is None:
        if type(v) == torch.Tensor and v.dim() == 2 and len(v) > 4 and len(v[0]) == dim:
            return v
        else:
            return [v]
    else:
        if type(v) == torch.Tensor and v.dim() == 3 and len(v) > 4 and len(v[0]) == dim and len(v[0, 0]) == dim2:
            return v
        else:
            return [v]
def new_list(n, dim = None, dim2 = None):
    vl = list()
    for i in range(n):
        vl.append(0.0)
    return vl
def svm_list_set(v, index, item):
    v[index] = item
def svm_list_get(v, index):
    if index < len(v):
        return v[index]
    else:
        return v[0]

def is_svm_array(v, dim = None, dim2 = None):
    if (dim is None or dim == 1) and dim2 is None:
        return type(v) == torch.Tensor and v.dim() == 1 and len(v) > 4
    elif dim2 is None:
        return type(v) == torch.Tensor and v.dim() == 2 and len(v) > 4 and len(v[0]) == dim
    else:
        return type(v) == torch.Tensor and v.dim() == 3 and len(v) > 4 and len(v[0]) == dim and len(v[0, 0]) == dim2
def maybe_svm_array(v):
    return type(v) == torch.Tensor and v.dim() >= 1 and len(v) > 4
def maybe_scalar_array(v):
    return type(v) == torch.Tensor and v.dim() == 1 and len(v) > 4
def maybe_vec_mat_array(v):
    return type(v) == torch.Tensor and v.dim() >= 2 and len(v) > 4

def new_tensor(type, n, dim = None, dim2 = None):
    return tensor_pools.New(type, n, dim, dim2)
def new_zero_tensor(n, dim = None, dim2 = None):
    t = tensor_pools.New(torch.float64, n, dim, dim2)
    t[...] = 0
    return t
def new_zero_int_tensor(n, dim = None, dim2 = None):
    t = tensor_pools.New(torch.int64, n, dim, dim2)
    t[...] = 0
    return t
def new_false_tensor(n, dim = None, dim2 = None):
    t = tensor_pools.New(torch.bool, n, dim, dim2)
    t[...] = False
    return t

def get_s_key(v):
    if type(v) == torch.Tensor:
        v = v.item()
    return type(v), v
def get_vm_key(v):
    if type(v) == torch.Tensor:
        dimn = v.dim()
        ty = None
        if dimn == 2:
            keys = []
            for elem in v:
                keys.append(tuple(elem))
            t = tuple(keys)
            return type(v[0][0]), t, 2
        else:
            return type(v[0]), tuple(v.tolist()), 1
    elif type(v[0]) == list:
        keys = []
        for elem in v:
            keys.append(tuple(elem))
        t = tuple(keys)
        return type(v[0][0]), t, 2
    else:
        return type(v[0]), tuple(v), 1
def get_s_gpu_tensor(v):
    '''
    if type(v) != torch.Tensor:
        return torch.as_tensor(v, device=device)
    elif not v.is_cuda:
        return v.cuda()
    return v
    '''
    if type(v) == torch.Tensor and v.is_cuda:
        return v
    caches = s_gpu_tensor_cache
    tkey, vkey = get_s_key(v)
    cache = caches.get(tkey)
    if cache is None:
        cache = {}
        caches[tkey] = cache
    nv = cache.get(vkey)
    if nv is not None:
        return nv
    if type(v) != torch.Tensor:
        nv = torch.as_tensor(v, device=device)
    else:
        nv = v.cuda()
    cache[vkey] = nv
    return nv

def get_vm_gpu_tensor(v):
    '''
    if type(v) != torch.Tensor:
        return torch.as_tensor(v, device=device)
    elif not v.is_cuda:
        return v.cuda()
    return v
    '''
    if type(v) == torch.Tensor and v.is_cuda:
        return v
    caches = vm_gpu_tensor_cache
    tkey, vkey, vdim = get_vm_key(v)
    cache = caches.get(tkey)
    if cache is None:
        cache = {}
        caches[tkey] = cache
    nv = cache.get(vkey)
    if nv is not None:
        return nv
    if type(v) != torch.Tensor:
        nv = torch.as_tensor(v, device=device)
        cache[vkey] = nv
    else:
        nv = v.cuda()
        if nv.dim() == vdim:
            cache[vkey] = nv
    return nv

def get_cpu_zeros(count):
    v = cpu_zeros_tensor_cache.get(count)
    if v is not None:
        return v
    zeros = torch.zeros(count)
    cpu_zeros_tensor_cache[count] = zeros
    return zeros
def get_cpu_ones(count):
    v = cpu_ones_tensor_cache.get(count)
    if v is not None:
        return v
    ones = torch.ones(count)
    cpu_ones_tensor_cache[count] = ones
    return ones
def get_gpu_zeros(count):
    v = gpu_zeros_tensor_cache.get(count)
    if v is not None:
        return v
    zeros = torch.zeros(count, device=device)
    gpu_zeros_tensor_cache[count] = zeros
    return zeros
def get_gpu_ones(count):
    v = gpu_ones_tensor_cache.get(count)
    if v is not None:
        return v
    ones = torch.ones(count, device=device)
    gpu_ones_tensor_cache[count] = ones
    return ones

def get_cpu_value(v):
    if type(v) == torch.Tensor and v.is_cuda:
        return v.cpu()
    return v
def get_gpu_value(v):
    if type(v) == torch.Tensor and not v.is_cuda:
        return v.cuda()
    return v

def change_to_same_f64(a, b):
    if a.dtype != torch.float64:
        a = a.double()
    if b.dtype != torch.float64:
        b = b.double()
    return a, b
def change_to_f64(v):
    if v.dtype != torch.float64:
        v = v.double()
    return v

def h_clamp_v_v_v(v, a, b):
    v = get_gpu_value(v)
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.clip(v, a, b)
def h_clamp_n_n_n(v, a, b):
    v = get_s_gpu_tensor(v)
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.clip(v, a, b)
def h_clamp_v_n_n(v, a, b):
    v = get_gpu_value(v)
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.clip(v, a, b)
def h_clamp_t_n_n_n(v, a, b):
    v = get_gpu_value(v)
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.clip(v, a, b)
def h_clamp_t_n_n_t_n(v, a, b):
    v = get_gpu_value(v)
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.clip(v, a, b)
def h_clamp_t_n_t_n_t_n(v, a, b):
    v = get_gpu_value(v)
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.clip(v, a, b)
def h_clamp_t_v_n_n(v, a, b):
    v = get_gpu_value(v)
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.clip(v, a, b)
def h_clamp_t_v_v_v(v, a, b):
    v = get_gpu_value(v)
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.clip(v, a, b)

def h_lerp_n_n_n(a, b, h):
    return (1 - h) * a + h * b
def h_lerp_v_v_n(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_v_v_v(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_v_v_t_n(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    a = torch.broadcast_to(a, (len(h), len(a)))
    b = torch.broadcast_to(b, (len(h), len(b)))
    return ((1 - h) * a.T + h * b.T).T
def h_lerp_v_t_v_t_n(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)    
    h = get_gpu_value(h)
    a = torch.broadcast_to(a, (len(b), len(a)))
    return ((1 - h) * a.T + h * b.T).T
def h_lerp_t_v_v_t_n(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)    
    h = get_gpu_value(h)
    b = torch.broadcast_to(b, (len(a), len(b)))
    return ((1 - h) * a.T + h * b.T).T
def h_lerp_v_v_t_v(a, b, h):
    r = ((1 - h) * a + h * b)
    return r
def h_lerp_n_n_t_n(a, b, h):
    return (1 - h) * a + h * b
def h_lerp_t_n_t_n_t_n(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_t_n_t_n_n(a, b, h):
    h = torch.broadcast_to(get_s_gpu_tensor(h), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_n_t_n_t_n(a, b, h):
    a = torch.broadcast_to(get_s_gpu_tensor(a), [len(b)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_t_n_n_t_n(a, b, h):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_t_n_n_n(a, b, h):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    h = torch.broadcast_to(get_s_gpu_tensor(h), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_n_t_n_n(a, b, h):
    a = torch.broadcast_to(get_s_gpu_tensor(a), [len(b)])
    h = torch.broadcast_to(get_s_gpu_tensor(h), [len(b)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_t_v_t_v_t_v(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_t_v_v_t_v(a, b, h):
    m = len(a)
    n = len(b)
    b = torch.broadcast_to(get_vm_gpu_tensor(b), (m, n))
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return (1 - h) * a + h * b
def h_lerp_t_v_t_v_t_n(a, b, h):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return ((1 - h) * a.T + h * b.T).T
def h_lerp_t_v_t_v_n(a, b, h):
    h = torch.broadcast_to(get_s_gpu_tensor(h), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return ((1 - h) * a.T + h * b.T).T
def h_lerp_t_v_v_n(a, b, h):
    h = torch.broadcast_to(get_s_gpu_tensor(h), [len(a)])
    b = torch.broadcast_to(b, (len(a), len(b)))
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    h = get_gpu_value(h)
    return ((1 - h) * a.T + h * b.T).T

def h_smoothstep_n_n_n(a, b, v):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    v = get_s_gpu_tensor(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_n_n_v(a, b, v):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    v = get_vm_gpu_tensor(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_n_n_t_n(a, b, v):
    a = torch.broadcast_to(get_s_gpu_tensor(a), [len(v)])
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(v)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    v = get_gpu_value(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_n_t_n_t_n(a, b, v):
    a = torch.broadcast_to(get_s_gpu_tensor(a), [len(v)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    v = get_gpu_value(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_t_n_n_t_n(a, b, v):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(v)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    v = get_gpu_value(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_n_n_t_v(a, b, v):
    m = len(v)
    n = len(v[0])
    a = torch.broadcast_to(get_s_gpu_tensor(a), (m, n))
    b = torch.broadcast_to(get_s_gpu_tensor(b), (m, n))
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    v = get_gpu_value(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_v_v_t_v(a, b, v):
    m = len(v)
    n = len(a)
    a = torch.broadcast_to(get_vm_gpu_tensor(a), (m, n))
    b = torch.broadcast_to(get_vm_gpu_tensor(b), (m, n))
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_t_n_t_n_t_n(a, b, v):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    v = get_gpu_value(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)
def h_smoothstep_t_n_t_n_n(a, b, v):
    v = torch.broadcast_to(get_s_gpu_tensor(v), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    v = get_gpu_value(v)
    t = (v - a) / (b - a)
    t = torch.clip(t, gpu_zero_tensor, gpu_one_tensor)
    return t * t * (3 - 2 * t)

def h_sin_n(v):
    v = get_s_gpu_tensor(v)
    return torch.sin(v)
def h_sin_v(v):
    v = get_gpu_value(v)
    return torch.sin(v)
def h_sin_t_n(v):
    return torch.sin(v)
def h_sin_t_v(v):
    return torch.sin(v)

def h_cos_n(v):
    v = get_s_gpu_tensor(v)
    return torch.cos(v)
def h_cos_v(v):
    v = get_gpu_value(v)
    return torch.cos(v)
def h_cos_t_n(v):
    return torch.cos(v)
def h_cos_t_v(v):
    return torch.cos(v)

def h_asin_n(v):
    return torch.arcsin(v)
def h_asin_v(v):
    return torch.arcsin(v)
def h_asin_t_n(v):
    return torch.arcsin(v)
def h_asin_t_v(v):
    return torch.arcsin(v)

def h_acos_n(v):
    return torch.arccos(v)
def h_acos_v(v):
    return torch.arccos(v)
def h_acos_t_n(v):
    return torch.arccos(v)
def h_acos_t_v(v):
    return torch.arccos(v)

def h_tan_n(v):
    v = get_s_gpu_tensor(v)
    return torch.tan(v)
def h_tan_t_n(v):
    return torch.tan(v)

def h_atan_n(v):
    v = get_s_gpu_tensor(v)
    return torch.arctan(v)
def h_atan_t_n(v):
    return torch.arctan(v)
def h_atan2_n_n(y, x):
    y = get_s_gpu_tensor(y)
    x = get_s_gpu_tensor(x)
    return torch.arctan2(y, x)
def h_atan2_t_n_t_n(y, x):
    return torch.arctan2(y, x)
def h_radians_n(v):
    v = get_s_gpu_tensor(v)
    return torch.deg2rad(v)
def h_radians_t_n(v):
    return torch.deg2rad(v)

def h_degrees_n(v):
    v = get_s_gpu_tensor(v)
    return torch.rad2deg(v)
def h_degrees_t_n(v):
    return torch.rad2deg(v)

def h_frac_n(v):
    v = get_s_gpu_tensor(v)
    return v - torch.floor(v)
def h_frac_v(v):
    return v - torch.floor(v)
def h_frac_t_n(v):
    return v - torch.floor(v)
def h_frac_t_v(v):
    return v - torch.floor(v)

def h_fmod_n_n(v1, v2):
    v1 = get_s_gpu_tensor(v1)
    v2 = get_s_gpu_tensor(v2)
    return torch.fmod(v1, v2)
def h_fmod_n_v(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_v_n(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    return torch.fmod(v1, v2)
def h_fmod_t_n_n(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_t_v_n(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_t_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    return torch.fmod(v1, v2)
def h_fmod_n_t_n(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_n_t_v(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_v_t_v(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_t_n_t_n(v1, v2):
    return torch.fmod(v1, v2)
def h_fmod_t_v_t_v(v1, v2):
    return torch.fmod(v1, v2)

def h_dot_n_n(v1, v2):
    v1 = get_s_gpu_tensor(v1)
    v2 = get_s_gpu_tensor(v2)
    return torch.dot(v1, v2)
def h_dot_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    return torch.dot(v1, v2)
def h_dot_t_n_n(v1, v2):
    return torch.dot(v1, v2)
def h_dot_t_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_vm_gpu_tensor(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    return torch.linalg.vecdot(v1, v2)
def h_dot_v_t_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    return torch.linalg.vecdot(v1, v2)
def h_dot_t_n_t_n(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    return torch.dot(v1, v2)
def h_dot_t_v_t_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    return torch.linalg.vecdot(v1, v2)

def h_reflect_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    dot = torch.dot(v1, v2)
    return v1 - 2 * dot * v2
def h_reflect_t_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    dot = torch.linalg.vecdot(v1, v2)
    dot = dot.reshape(len(dot), 1)
    _2_dot_v2 = torch.mul(dot, v2) * 2.0
    return v1 - _2_dot_v2
def h_reflect_v_t_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    dot = torch.linalg.vecdot(v1, v2)
    _2_dot_v2 = torch.mul(dot, v2.T).T * 2.0
    return v1 - _2_dot_v2
def h_reflect_t_v_t_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v1, v2 = change_to_same_f64(v1, v2)
    dot = torch.linalg.vecdot(v1, v2)
    _2_dot_v2 = torch.mul(dot, v2.T).T * 2.0
    return v1 - _2_dot_v2

def h_refract_v_v_n(I, N, eta):
    m = len(I)
    I = get_vm_gpu_value(I)
    N = get_vm_gpu_value(N)
    eta = get_s_gpu_tensor(eta)
    dotval = h_dot_v_v(N, I)
    k = gpu_one_tensor - eta * eta * (gpu_one_tensor - dotval * dotval)
    R0 = torch.broadcast_to(gpu_zero_tensor, [m])
    bvs = k >= gpu_zero_tensor
    bvs = torch.broadcast_to(get_s_gpu_tensor(bvs), [m])
    R = torch.where(bvs, eta * I - (eta * dotval + torch.sqrt(torch.abs(k))) * N, R0)
    return R
def h_refract_t_v_t_v_n(I, N, eta):
    m = len(I)
    n = len(I[0])
    I = get_gpu_value(I)
    N = get_gpu_value(N)
    eta = get_s_gpu_tensor(eta)
    dotval = h_dot_t_v_t_v(N, I)
    k = gpu_one_tensor - eta * eta * (gpu_one_tensor - dotval * dotval)
    R0 = torch.broadcast_to(gpu_zero_tensor, (m, n))
    bvs = k >= gpu_zero_tensor
    bvs = torch.broadcast_to(bvs, (n, m)).T
    R = torch.where(bvs, eta * I - ((eta * dotval + torch.sqrt(torch.abs(k))) * N.T).T, R0)
    return R
def h_refract_t_v_t_v_t_n(I, N, eta):
    m = len(I)
    n = len(I[0])
    I = get_gpu_value(I)
    N = get_gpu_value(N)
    eta = get_gpu_value(eta)
    dotval = h_dot_t_v_t_v(N, I)
    k = gpu_one_tensor - eta * eta * (gpu_one_tensor - dotval * dotval)
    R0 = torch.broadcast_to(gpu_zero_tensor, (m, n))
    bvs = k >= gpu_zero_tensor
    bvs = torch.broadcast_to(bvs, (n, m)).T
    R = torch.where(bvs, (eta * I.T).T - ((eta * dotval + torch.sqrt(torch.abs(k))) * N.T).T, R0)
    return R

def h_floor_n(v):
    v = get_s_gpu_tensor(v)
    return torch.floor(v)
def h_floor_v(v):
    return torch.floor(v)
def h_floor_t_n(v):
    return torch.floor(v)
def h_floor_t_v(v):
    return torch.floor(v)

def h_ceil_n(v):
    v = get_s_gpu_tensor(v)
    return torch.ceil(v)
def h_ceil_v(v):
    return torch.ceil(v)
def h_ceil_t_n(v):
    return torch.ceil(v)
def h_ceil_t_v(v):
    return torch.ceil(v)

def h_round_n(v):
    v = get_s_gpu_tensor(v)
    return torch.round(v)
def h_round_v(v):
    return torch.round(v)
def h_round_t_n(v):
    return torch.round(v)
def h_round_t_v(v):
    return torch.round(v)

def h_length_n(v):
    return torch.abs(v)
def h_length_t_n(v):
    return torch.abs(v)
def h_length_v(v):
    return torch.sqrt(torch.sum(torch.pow(v, 2)))
def h_length_t_v(v):
    return torch.sqrt(torch.linalg.vecdot(v, v))

def h_distance_v_v(v1, v2):
    v = v1 - v2
    v = get_vm_gpu_tensor(v)
    return torch.sqrt(torch.sum(torch.power(v, 2)))
def h_distance_t_v_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v = v1 - v2
    return torch.sqrt(torch.linalg.vecdot(v, v))
def h_distance_v_t_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v = v1 - v2
    return torch.sqrt(torch.linalg.vecdot(v, v))
def h_distance_t_v_t_v(v1, v2):
    v1 = get_gpu_value(v1)
    v2 = get_gpu_value(v2)
    v = v1 - v2
    return torch.sqrt(torch.linalg.vecdot(v, v))

def h_normalize_v(v):
    return v / (torch.linalg.norm(v) + sys.float_info.epsilon)
def h_normalize_t_v(v):
    return (v.T / (torch.linalg.norm(v, axis=1) + sys.float_info.epsilon)).T
def h_cross_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    a, b = change_to_same_f64(a, b)
    return torch.cross(a, b)
def h_cross_t_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    a, b = change_to_same_f64(a, b)
    m = len(a)
    n = len(b)
    b = torch.broadcast_to(b, (m, n))
    return torch.cross(a, b)
def h_cross_t_v_t_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    a, b = change_to_same_f64(a, b)
    return torch.cross(a, b)

def h_sqrt_n(v):
    v = get_s_gpu_tensor(v)
    return torch.sqrt(v)
def h_sqrt_v(v):
    v = get_gpu_value(v)
    return torch.sqrt(v)
def h_sqrt_t_n(v):
    v = get_gpu_value(v)
    return torch.sqrt(v)
def h_sqrt_t_v(v):
    return torch.sqrt(v)

def h_pow_n_n(v, n):
    v = get_s_gpu_tensor(v)
    n = get_s_gpu_tensor(n)
    return torch.pow(v, n)
def h_pow_v_v(v, n):
    v = get_gpu_value(v)
    n = get_gpu_value(n)
    return torch.pow(v, n)
def h_pow_v_n(v, n):
    v = get_gpu_value(v)
    n = get_gpu_value(n)
    return torch.pow(v, n)
def h_pow_t_n_n(v, n):
    return torch.pow(v, n)
def h_pow_n_t_n(v, n):
    return torch.pow(v, n)
def h_pow_t_n_t_n(v, n):
    return torch.pow(v, n)
def h_pow_t_v_v(v, n):
    v = get_gpu_value(v)
    n = get_gpu_value(n)
    return torch.pow(v, n)

def h_log_n(v):
    v = get_s_gpu_tensor(v)
    return torch.log(v)
def h_log_v(v):
    v = get_vm_gpu_tensor(v)
    return torch.log(v)
def h_log_t_n(v):
    return torch.log(v)
def h_log_t_v(v):
    return torch.log(v)

def h_log2_n(v):
    v = get_s_gpu_tensor(v)
    return torch.log2(v)
def h_log2_v(v):
    v = get_vm_gpu_tensor(v)
    return torch.log2(v)
def h_log2_t_n(v):
    return torch.log2(v)
def h_log2_t_v(v):
    return torch.log2(v)

def h_log10_n(v):
    v = get_s_gpu_tensor(v)
    return torch.log10(v)
def h_log10_v(v):
    v = get_vm_gpu_tensor(v)
    return torch.log10(v)
def h_log10_t_n(v):
    return torch.log10(v)
def h_log10_t_v(v):
    return torch.log10(v)

def h_exp_n(v):
    v = get_s_gpu_tensor(v)
    return torch.exp(v)
def h_exp_v(v):
    v = get_vm_gpu_tensor(v)
    return torch.exp(v)
def h_exp_t_n(v):
    return torch.exp(v)
def h_exp_t_v(v):
    return torch.exp(v)

def h_exp2_n(v):
    v = get_s_gpu_tensor(v)
    return torch.exp2(v)
def h_exp2_v(v):
    v = get_vm_gpu_tensor(v)
    return torch.exp2(v)
def h_exp2_t_n(v):
    return torch.exp2(v)
def h_exp2_t_v(v):
    return torch.exp2(v)
def h_sign_n(v):
    v = get_s_gpu_tensor(v)
    return torch.sign(v)
def h_sign_v(v):
    v = get_vm_gpu_tensor(v)
    return torch.sign(v)
def h_sign_t_n(v):
    return torch.sign(v)
def h_sign_t_v(v):
    return torch.sign(v)

def h_ddx_n(v):
    return get_s_gpu_tensor(0.001)
def h_ddy_n(v):
    return get_s_gpu_tensor(0.001)
def h_ddx_v(v):
    return gpu_zero2_tensor
def h_ddy_v(v):
    return gpu_zero2_tensor
def h_ddx_fine_n(v):
    return gpu_zero_tensor
def h_ddy_fine_n(v):
    return gpu_zero_tensor
def h_ddx_coarse_n(v):
    return gpu_zero_tensor
def h_ddy_coarse_n(v):
    return gpu_zero_tensor
def h_ddx_t_n(v):
    return torch.broadcast_to(get_s_gpu_tensor(0.001), [len(v)])
def h_ddy_t_n(v):
    return torch.broadcast_to(get_s_gpu_tensor(0.001), [len(v)])
def h_ddx_t_v(v):
    return torch.broadcast_to(gpu_zero2_tensor, (len(v), 2))
def h_ddy_t_v(v):
    return torch.broadcast_to(gpu_zero2_tensor, (len(v), 2))
def h_ddx_fine_t_n(v):
    return torch.broadcast_to(gpu_zero_tensor, [len(v)])
def h_ddy_fine_t_n(v):
    return torch.broadcast_to(gpu_zero_tensor, [len(v)])
def h_ddx_coarse_t_n(v):
    return torch.broadcast_to(gpu_zero_tensor, [len(v)])
def h_ddy_coarse_t_n(v):
    return torch.broadcast_to(gpu_zero_tensor, [len(v)])

def h_fwidth_n(v):
    return h_abs_n(h_ddx_n(v)) + h_abs_n(h_ddy_n(v))
def h_fwidth_t_n(v):
    return h_abs_t_n(h_ddx_t_n(v)) + h_abs_t_n(h_ddy_t_n(v))
def h_fwidth_v(v):
    return h_abs_v(h_ddx_v(v)) + h_abs_v(h_ddy_v(v))
def h_fwidth_t_v(v):
    return h_abs_t_v(h_ddx_t_v(v)) + h_abs_t_v(h_ddy_t_v(v))
def h_transpose_m(m):
    return torch.t(m)
def h_transpose_t_m(m):
    return m.transpose(1, 2)

def h_matmul_f2x2_f2(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(m, v)
def h_matmul_f2x2_t_f2(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(m, v.T).T
def h_matmul_t_f2x2_f2(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(m, v.T).T
def h_matmul_t_f2x2_t_f2(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    r = -torch.matmul(torch.broadcast_to(v, (2, len(v), 2)).transpose(0, 1), m).transpose(0, 1)[0, ...]
    return r
def h_matmul_f2_f2x2(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(v, m)
def h_matmul_t_f2_f2x2(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(v, m)
def h_matmul_t_f2_t_f2x2(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(torch.broadcast_to(v, (2, len(v), 2)).transpose(0, 1), m).transpose(0, 1)[0, ...]
def h_matmul_f3x3_f3(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(m, v)
def h_matmul_f3x3_f3x3(m1, m2):
    m1 = get_gpu_value(m1)
    m2 = get_gpu_value(m2)
    m1, m2 = change_to_same_f64(m1, m2)
    return torch.matmul(m1, m2)
def h_matmul_f3x3_t_f3(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    r = torch.matmul(m, v.T).T
    return r
def h_matmul_f3_f3x3(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(v, m)
def h_matmul_t_f3_f3x3(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(v, m)
def h_matmul_t_f3_t_f3x3(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(torch.broadcast_to(v, (3, len(v), 3)).transpose(0, 1), m).transpose(0, 1)[0, ...]
def h_matmul_f4x4_f4(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(m, v)
def h_matmul_f4x4_f4x4(m1, m2):
    m1 = get_gpu_value(m1)
    m2 = get_gpu_value(m2)
    m1, m2 = change_to_same_f64(m1, m2)
    return torch.matmul(m1, m2)
def h_matmul_f4x4_t_f4(m, v):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(m, v.T).T
def h_matmul_f4_f4x4(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(v, m)
def h_matmul_t_f4_f4x4(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(v, m)
def h_matmul_t_f4_t_f4x4(v, m):
    m = get_gpu_value(m)
    v = get_gpu_value(v)
    m, v = change_to_same_f64(m, v)
    return torch.matmul(torch.broadcast_to(v, (4, len(v), 4)).transpose(0, 1), m).transpose(0, 1)[0, ...]

def h_max_n_n(a, b):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.maximum(a, b)
def h_max_v_n(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.maximum(a, b)
def h_max_n_v(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.maximum(a, b)
def h_max_v_v(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.maximum(a, b)
def h_max_v_t_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_t_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_t_n_n(a, b):
    m = len(a)
    b = torch.broadcast_to(get_s_gpu_tensor(b), [m])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_n_t_n(a, b):
    m = len(b)
    a = torch.broadcast_to(get_s_gpu_tensor(a), [m])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_t_n_t_n(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_t_v_t_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_t_v_n(a, b):
    m = len(a)
    n = len(a[0])
    b = torch.broadcast_to(get_s_gpu_tensor(b), (m, n))
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.maximum(a, b)
def h_max_n_t_v(a, b):
    return torch.maximum(a, b)

def h_min_n_n(a, b):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.minimum(a, b)
def h_min_v_n(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.minimum(a, b)
def h_min_n_v(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.minimum(a, b)
def h_min_v_v(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.minimum(a, b)
def h_min_v_t_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.minimum(a, b)
def h_min_t_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.minimum(a, b)
def h_min_t_n_n(a, b):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    b = get_gpu_value(b)
    return torch.minimum(a, b)
def h_min_n_t_n(a, b):
    a = torch.broadcast_to(get_s_gpu_tensor(a), [len(b)])
    a = get_gpu_value(a)
    return torch.minimum(a, b)
def h_min_t_n_t_n(a, b):
    return torch.minimum(a, b)
def h_min_t_v_t_v(a, b):
    return torch.minimum(a, b)
def h_min_t_v_n(a, b):
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return torch.minimum(a, b)
def h_min_n_t_v(a, b):
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.minimum(a, b)

def h_where_n_n_n(b, y, n):
    return y if b else n
def h_where_n_v_v(b, y, n):
    return y if b else n
def h_where_n_t_n_t_n(b, y, n):
    return y if b else n
def h_where_n_t_v_t_v(b, y, n):
    return y if b else n
def h_where_n_n_t_n(b, y, n):
    ct = len(n)
    y = torch.broadcast_to(get_s_gpu_tensor(y), [ct])
    return y if b else n
def h_where_n_t_n_n(b, y, n):
    ct = len(y)
    n = torch.broadcast_to(get_s_gpu_tensor(n), [ct])
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return y if b else n
def h_where_n_v_t_v(b, y, n):
    ct = len(n)
    y = torch.broadcast_to(get_vm_gpu_tensor(y), (ct, len(y)))
    return y if b else n
def h_where_n_t_v_v(b, y, n):
    ct = len(y)
    n = torch.broadcast_to(n, (ct, len(n)))
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return y if b else n
def h_where_t_n_t_n_t_n(b, y, n):
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_t_v_t_v(b, y, n):
    ct = len(y[0])
    b = torch.broadcast_to(b, (ct, len(b))).T
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_t_v_v(b, y, n):
    ct = len(y[0])
    b = torch.broadcast_to(b, (ct, len(b))).T
    n = torch.broadcast_to(n, (len(y), ct))
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_v_t_v(b, y, n):
    ct = len(n[0])
    b = torch.broadcast_to(b, (ct, len(b))).T
    y = torch.broadcast_to(y, (len(n), ct))
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_v_v(b, y, n):
    ct = len(n)
    b = torch.broadcast_to(b, (ct, len(b))).T
    y = torch.broadcast_to(y, (len(b), ct))
    n = torch.broadcast_to(n, (len(b), ct))
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_t_n_n(b, y, n):
    ct = len(y)
    n = torch.broadcast_to(get_s_gpu_tensor(n), [ct])
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_n_t_n(b, y, n):
    ct = len(n)
    y = torch.broadcast_to(get_s_gpu_tensor(y), [ct])
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_n_n(b, y, n):
    ct = len(b)
    y = torch.broadcast_to(get_s_gpu_tensor(y), [ct])
    n = torch.broadcast_to(get_s_gpu_tensor(n), [ct])
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_t_n_t_m_t_m(b, y, n):
    ct = len(b)
    m1 = len(y[0])
    m2 = len(y[0][0])
    b = torch.broadcast_to(b, (m2, m1, ct)).T
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    return torch.where(b, y, n)
def h_where_n_m_m(b, y, n):
    return y if b else n
def h_where_n_t_an_t_an(b, y, n):
    return y if b else n
def h_where_n_t_an_an(b, y, n):
    m1 = len(n)
    m2 = len(y[0])
    n = torch.broadcast_to(n, (m2, m1)).T
    return y if b else n
def h_where_n_an_t_an(b, y, n):
    m1 = len(y)
    m2 = len(n[0])
    y = torch.broadcast_to(y, (m2, m1)).T
    return y if b else n
def h_where_t_n_t_an_t_an(b, y, n):
    b = get_gpu_value(b)
    y = get_gpu_value(y)
    n = get_gpu_value(n)
    ct = len(b)
    m = len(y)
    b = torch.broadcast_to(b, (m, ct))
    return torch.where(b, y, n)
def h_where_t_n_t_av_t_av(b, y, n):
    ct = len(b)
    m1 = len(y)
    m2 = len(y[0][0])
    b = torch.broadcast_to(b, (m1, m2, ct)).transpose(2, 1)
    return torch.where(b, y, n)
def h_where_n_t_av_t_av(b, y, n):
    return y if b else n
def h_where_v_v_v(b, y, n):
    return torch.where(b, y, n)
def h_where_t_v_v_t_v(b, y, n):
    b = get_gpu_value(b)
    n = get_gpu_value(n)
    m = len(b)
    y = torch.broadcast_to(get_vm_gpu_tensor(y), (m, len(y)))
    y = get_gpu_value(y)
    return torch.where(b, y, n)

def h_step_n_n(y, x):
    x = get_s_gpu_tensor(x)
    y = get_s_gpu_tensor(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_v_n(y, x):
    x = get_s_gpu_tensor(x)
    y = get_vm_gpu_tensor(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_n_v(y, x):
    x = get_vm_gpu_tensor(x)
    y = get_s_gpu_tensor(y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_v_v(y, x):
    x = get_vm_gpu_tensor(x)
    y = get_vm_gpu_tensor(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_t_v_v(y, x):
    x = torch.broadcast_to(x, (len(y), len(x)))
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_v_t_v(y, x):
    y = torch.broadcast_to(y, (len(x), len(y)))
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_t_v_t_v(y, x):
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_n_t_v(y, x):
    y = torch.broadcast_to(get_s_gpu_tensor(y), (len(x), len(x[0])))
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_n_t_n(y, x):
    y = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_t_n_n(y, x):
    x = torch.broadcast_to(get_s_gpu_tensor(x), [len(y)])
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)
def h_step_t_n_t_n(y, x):
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    x, y = change_to_same_f64(x, y)
    return torch.heaviside(x - y, gpu_one_tensor)

def h_abs_n(v):
    v = get_s_gpu_tensor(v)
    return torch.abs(v)
def h_abs_v(v):
    return torch.abs(v)
def h_abs_t_n(v):
    return torch.abs(v)
def h_abs_t_v(v):
    return torch.abs(v)

def h_any_n(v):
    v = get_s_gpu_tensor(v)
    return torch.any(v)
def h_any_v(v):
    return torch.any(v)
def h_any_t_n(v):
    ct = len(v)
    v = torch.broadcast_to(v, (1, ct))
    return torch.any(v, 0)
def h_any_t_v(v):
    return torch.any(v, 1)

def h_all_n(v):
    v = get_s_gpu_tensor(v)
    return torch.all(v)
def h_all_v(v):
    return torch.all(v)
def h_all_t_n(v):
    return torch.all(v)
def h_all_t_v(v):
    return torch.all(v)

def array_init_an(arr):
    return torch.asarray(arr, device=device)
def array_init_an2(arr):
    return torch.stack(arr)
def array_init_an3(arr):
    return torch.stack(arr)
def array_init_an4(arr):
    return torch.stack(arr)
def array_init_t_an(arr):
    return torch.stack(arr)
def array_init_t_an2(arr):
    return torch.stack(arr)
def array_init_t_an3(arr):
    return torch.stack(arr)
def array_init_t_an4(arr):
    return torch.stack(arr)
def array_set_an_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an_n(arr, ix):
    return arr[ix]
def array_set_t_an_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_t_an_n(arr, ix):
    return arr[ix]
def array_set_an2_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an2_n(arr, ix):
    return arr[ix]
def array_set_an3_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an3_n(arr, ix):
    return arr[ix]
def array_set_an4_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an4_n(arr, ix):
    return arr[ix]
def array_set_t_an2_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_t_an2_n(arr, ix):
    r = arr[ix]
    return r
def array_set_t_an3_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_t_an3_n(arr, ix):
    r = arr[ix]
    return r
def array_set_t_an4_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_t_an4_n(arr, ix):
    r = arr[ix]
    return r
def array_set_an_t_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an_t_n(arr, ix):
    r = arr[ix]
    return r
def array_set_an2_t_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an2_t_n(arr, ix):
    r = arr[ix]
    return r
def array_set_an3_t_n(arr, ix, v):
    arr[ix.long()] = v
    return v
def array_get_an3_t_n(arr, ix):
    r = arr[ix.long()]
    return r
def array_set_an4_t_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_an4_t_n(arr, ix):
    r = arr[ix]
    return r
def array_set_t_an_t_n(arr, ix, v):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    v = get_gpu_value(v)
    arr, v = change_to_same_f64(arr, v)
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
def array_get_t_an_t_n(arr, ix):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
def array_set_t_an2_t_n(arr, ix, v):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    v = get_gpu_value(v)
    arr, v = change_to_same_f64(arr, v)
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
def array_get_t_an2_t_n(arr, ix):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
def array_set_t_an3_t_n(arr, ix, v):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    v = get_gpu_value(v)
    arr, v = change_to_same_f64(arr, v)
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
def array_get_t_an3_t_n(arr, ix):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
def array_set_t_an4_t_n(arr, ix, v):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    v = get_gpu_value(v)
    arr, v = change_to_same_f64(arr, v)
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
def array_get_t_an4_t_n(arr, ix):
    arr = get_gpu_value(arr)
    ix = get_gpu_value(ix)
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
def array_set_n2_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_n2_n(arr, ix):
    r = arr[ix]
    return r
def array_set_n3_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_n3_n(arr, ix):
    r = arr[ix]
    return r
def array_set_n4_n(arr, ix, v):
    arr[ix] = v
    return v
def array_get_n4_n(arr, ix):
    r = arr[ix]
    return r
def array_set_n2x2_n(m, ix, val):
    m[ix][0] = val[0]
    m[ix][1] = val[1]
def array_get_n2x2_n(m, ix):
    arr = m[ix]
    return arr
def array_set_n3x3_n(m, ix, val):
    m[ix][0] = val[0]
    m[ix][1] = val[1]
    m[ix][2] = val[2]
def array_get_n3x3_n(m, ix):
    arr = m[ix]
    return arr
def array_set_n4x4_n(m, ix, val):
    m[ix][0] = val[0]
    m[ix][1] = val[1]
    m[ix][2] = val[2]
    m[ix][3] = val[3]
def array_get_n4x4_n(m, ix):
    arr = m[ix]
    return arr
def array_set_t_n2_n(m, ix, val):
    m[..., ix] = val
    return val
def array_get_t_n2_n(m, ix):
    v = m[..., ix]
    return v
def array_set_t_n3_n(m, ix, val):
    m[..., ix] = val
    return val
def array_get_t_n3_n(m, ix):
    v = m[..., ix]
    return v
def array_set_t_n4_n(m, ix, val):
    m[..., ix] = val
    return val
def array_get_t_n4_n(m, ix):
    v = m[..., ix]
    return v
def array_set_t_n2x2_n(m, ix, val):
    m.swapaxes(1,2)[..., ix][..., 0] = val[..., 0]
    m.swapaxes(1,2)[..., ix][..., 1] = val[..., 1]
    return val
def array_get_t_n2x2_n(m, ix):
    v = m.swapaxes(1,2)[..., ix]
    return v
def array_set_t_n3x3_n(m, ix, val):
    m.swapaxes(1,2)[..., ix][..., 0] = val[..., 0]
    m.swapaxes(1,2)[..., ix][..., 1] = val[..., 1]
    m.swapaxes(1,2)[..., ix][..., 2] = val[..., 2]
    return val
def array_get_t_n3x3_n(m, ix):
    v = m.swapaxes(1,2)[..., ix]
    return v
def array_set_t_n4x4_n(m, ix, val):
    m.swapaxes(1,2)[..., ix][..., 0]  = val[..., 0]
    m.swapaxes(1,2)[..., ix][..., 1]  = val[..., 1]
    m.swapaxes(1,2)[..., ix][..., 2]  = val[..., 2]
    m.swapaxes(1,2)[..., ix][..., 3]  = val[..., 3]
    return val
def array_get_t_n4x4_n(m, ix):
    v = m.swapaxes(1,2)[..., ix]
    return v
def array_set_t_n2_n(m, ix, val):
    m[..., ix] = val
    return val
def array_get_t_n2_n(m, ix):
    v = m[..., ix]
    return v
def array_set_t_n3_n(m, ix, val):
    m[..., ix] = val
    return val
def array_get_t_n3_n(m, ix):
    v = m[..., ix]
    return v
def array_set_t_n4_n(m, ix, val):
    m[..., ix] = val
    return val
def array_get_t_n4_n(m, ix):
    v = m[..., ix]
    return v

def array_set_and_broadcast_an_n(arr, ix, v):
    m = len(v)
    arr = torch.tile(arr, (m, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr
def array_set_and_broadcast_an2_n(arr, ix, v):
    m = len(v)
    arr = torch.tile(arr, (m, 1, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr
def array_set_and_broadcast_an3_n(arr, ix, v):
    m = len(v)
    arr = torch.tile(arr, (m, 1, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr
def array_set_and_broadcast_an4_n(arr, ix, v):
    m = len(v)
    arr = torch.tile(arr, (m, 1, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr

def array_broadcast_an(ct, arr):
    arr = torch.tile(arr, (ct, 1)).swapaxes(0, 1)
    return arr
def array_broadcast_an2(ct, arr):
    arr = torch.tile(arr, (ct, 1, 1)).swapaxes(0, 1)
    return arr
def array_broadcast_an3(ct, arr):
    arr = torch.tile(arr, (ct, 1, 1)).swapaxes(0, 1)
    return arr
def array_broadcast_an4(ct, arr):
    arr = torch.tile(arr, (ct, 1, 1)).swapaxes(0, 1)
    return arr

def swizzle(v, m, dim, dim2 = None):
    if maybe_vec_mat_array(v):
        if dim == 2 and dim2 is None and m == "xy":
            nv = torch.clone(v)
            return nv
        elif dim == 3 and dim2 is None and m == "xyz":
            nv = torch.clone(v)
            return nv
        elif dim == 4 and dim2 is None and m == "xyzw":
            nv = torch.clone(v)
            return nv
        elif m == "x":
            return v[..., 0]
        elif m == "y":
            return v[..., 1]
        elif m == "z":
            return v[..., 2]
        elif m == "w":
            return v[..., 3]
        else:
            nv = list()
            for c in m:
                if c == "x":
                    nv.append(v[..., 0])
                elif c == "y":
                    nv.append(v[..., 1])
                elif c == "z":
                    nv.append(v[..., 2])
                elif c == "w":
                    nv.append(v[..., 3])
            return torch.column_stack(nv)
    else:
        if dim == 2 and dim2 is None and m == "xy":
            return v
        elif dim == 3 and dim2 is None and m == "xyz":
            return v
        elif dim == 4 and dim2 is None and m == "xyzw":
            return v
        elif m == "x":
            return v[0]
        elif m == "y":
            return v[1]
        elif m == "z":
            return v[2]
        elif m == "w":
            return v[3]
        else:
            nv = list()
            for c in m:
                if c == "x":
                    nv.append(v[0])
                elif c == "y":
                    nv.append(v[1])
                elif c == "z":
                    nv.append(v[2])
                elif c == "w":
                    nv.append(v[3])
            return get_vm_gpu_tensor(nv)

def swizzle_set(v, m, val, dim, dim2 = None):
    if maybe_vec_mat_array(v):
        if dim == 2 and dim2 is None and m == "xy":
            torch.copyto(v, val)
        elif dim == 3 and dim2 is None and m == "xyz":
            torch.copyto(v, val)
        elif dim == 4 and dim2 is None and m == "xyzw":
            torch.copyto(v, val)
        elif m == "x":
            v[..., 0] = val
        elif m == "y":
            v[..., 1] = val
        elif m == "z":
            v[..., 2] = val
        elif m == "w":
            v[..., 3] = val
        else:
            ns = len(m)
            ix = 0
            for c in m:
                if c == "x":
                    v[..., 0] = val[..., ix]
                elif c == "y":
                    v[..., 1] = val[..., ix]
                elif c == "z":
                    v[..., 2] = val[..., ix]
                elif c == "w":
                    v[..., 3] = val[..., ix]
                ix = ix + 1
    else:
        if dim == 2 and dim2 is None and m == "xy":
            torch.copyto(v, val)
        elif dim == 3 and dim2 is None and m == "xyz":
            torch.copyto(v, val)
        elif dim == 4 and dim2 is None and m == "xyzw":
            torch.copyto(v, val)
        elif m == "x":
            v[0] = val
        elif m == "y":
            v[1] = val
        elif m == "z":
            v[2] = val
        elif m == "w":
            v[3] = val
        else:
            ns = len(m)
            ix = 0
            for c in m:
                if c == "x":
                    v[0] = val[ix]
                elif c == "y":
                    v[1] = val[ix]
                elif c == "z":
                    v[2] = val[ix]
                elif c == "w":
                    v[3] = val[ix]
                ix = ix + 1

class SwizzleObject:
    def __init__(self, ndarr, dim, dim2 = None):
        self.dim = dim
        self.dim2 = dim2
        self.arr = ndarr
    def __getattr__(self, name):
        if name == "dim" or name == "dim2" or name == "arr":
            return object.__getattribute__(self, name)
        if self.dim == 2 and self.dim2 is None:
            return swizzle(self.arr, name, 2)
        elif self.dim == 3 and self.dim2 is None:
            return swizzle(self.arr, name, 3)
        elif self.dim == 4 and self.dim2 is None:
            return swizzle(self.arr, name, 4)
        else:
            return object.__getattribute__(self, name)
    def __setattr__(self, name, val):
        if name == "dim" or name == "dim2" or name == "arr":
            object.__setattr__(self, name, val)
            return
        if self.dim == 2 and self.dim2 is None:
            swizzle_set(self.arr, name, val, 2)
        elif self.dim == 3 and self.dim2 is None:
            swizzle_set(self.arr, name, val, 3)
        elif self.dim == 4 and self.dim2 is None:
            swizzle_set(self.arr, name, val, 4)
        else:
            object.__setattr__(self, name, val)


def h_inc_i(v):
    return v + 1
def h_dec_i(v):
    return v - 1
def h_inc_vi(v):
    return v + 1
def h_dec_vi(v):
    return v - 1
def h_inc_t_i(v):
    return v + 1
def h_dec_t_i(v):
    return v - 1
def h_inc_t_vi(v):
    return v + 1
def h_dec_t_vi(v):
    return v - 1
def h_inc_f(v):
    return v + 1
def h_dec_f(v):
    return v - 1
def h_inc_t_f(v):
    return v + 1
def h_dec_t_f(v):
    return v + 1

def h_add_f(v):
    return v
def h_sub_f(v):
    return -v
def h_not_n(v):
    return not v
def h_bitnot_i(v):
    return ~v
def h_bitnot_u(v):
    return (~v) + (1<<32)
def h_bitnot_vi(v):
    return ~v
def h_bitnot_vi(v):
    return ~v
def h_bitnot_vi(v):
    return ~v

def h_add_t_f(v):
    return +v
def h_sub_t_f(v):
    return -v
def h_add_t_vf(v):
    return +v
def h_sub_t_vf(v):
    return -v
def h_add_vf(v):
    return v
def h_sub_vf(v):
    return -v


def h_not_v(v):
    return not v
def h_not_t_n(v):
    return torch.bitwise_not(v.bool())

def h_add_f_f(a, b):
    return a + b
def h_sub_f_f(a, b):
    return a - b
def h_mul_f_f(a, b):
    return a * b
def h_div_f_f(a, b):
    if b == 0:
        return sys.float_info.max
    return a / b
def h_mod_i_i(a, b):
    return a % b

def h_add_vf_f(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.add(a, b)
def h_sub_vf_f(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.sub(a, b)
def h_mul_vf_f(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.mul(a, b)
def h_div_vf_f(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.div(a, b)
def h_mod_vi_i(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return torch.remainder(a, b)

def h_add_f_vf(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return a + b
def h_sub_f_vf(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return a - b
def h_mul_f_vf(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return a * b
def h_div_f_vf(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return a / b
def h_mod_i_vi(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return a % b

def h_add_vf_vf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.add(a, b)
def h_sub_vf_vf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.sub(a, b)
def h_mul_vf_vf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.mul(a, b)
def h_div_vf_vf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.div(a, b)
def h_mod_vi_vi(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return torch.remainder(a, b)

def h_and_n_n(a, b):
    return a and b
def h_or_n_n(a, b):
    return a or b

def h_and_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a and b
def h_or_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a or b

def h_and_t_n_n(a, b):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.logical_and(a, b)
    return r
def h_and_n_t_n(a, b):
    a = torch.broadcast_to(a, [len(b)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.logical_and(a, b)
    return r
def h_and_t_n_t_n(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.logical_and(a, b)
    return r
def h_or_t_n_n(a, b):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.logical_or(a, b)
    return r
def h_or_n_t_n(a, b):
    a = torch.broadcast_to(get_s_gpu_tensor(a), [len(b)])
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.logical_or(a, b)
    return r
def h_or_t_n_t_n(a, b):
    r = torch.logical_or(a, b)
    return r

def h_add_t_f_t_f(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.add(a, b)
def h_add_t_vf_t_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.add(a, b)
def h_add_f_t_f(a, b):
    return torch.add(a, b)
def h_add_t_f_f(a, b):
    return torch.add(a, b)
def h_add_t_i_i(a, b):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    r = a + b
    return r
def h_add_t_u_i(a, b):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    r = a + b
    return r
def h_add_t_u_u(a, b):
    b = torch.broadcast_to(get_s_gpu_tensor(b), [len(a)])
    r = a + b
    return r
def h_add_t_u_t_u(a, b):
    r = a + b
    return r
def h_add_f_t_vf(a, b):
    return a + b
def h_add_t_vf_f(a, b):
    return a + b
def h_add_t_f_vf(a, b):
    m = len(a)
    n = len(b)
    b = torch.broadcast_to(get_vm_gpu_tensor(b), (m, n))
    a = torch.broadcast_to(a, (n, m)).T
    a = get_gpu_value(a)
    return a + b
def h_add_vf_t_f(a, b):
    m = len(b)
    n = len(a)
    a = torch.broadcast_to(get_vm_gpu_tensor(a), (m, n))
    b = torch.broadcast_to(b, (n, m)).T
    b = get_gpu_value(b)
    return torch.add(a, b)
def h_add_t_vf_t_f(a, b):
    m = len(b)
    n = len(a[0])
    b = torch.broadcast_to(b, (n, m)).T
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a + b
def h_add_t_f_t_vf(a, b):
    m = len(a)
    n = len(b[0])
    a = torch.broadcast_to(a, (n, m)).T
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a + b
def h_add_vf_t_vf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.add(a, b)
def h_add_t_vf_vf(a, b):
    a = get_gpu_value(a)
    b = get_vm_gpu_tensor(b)
    return torch.add(a, b)
def h_add_t_vu_t_vu(a, b):
    r = a + b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_add_vu_vu(a, b):
    r = a + b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_add_i_i(a, b):
    return a + b

def h_sub_i_i(a, b):
    return a - b
def h_sub_i_t_i(a, b):
    return a - b
def h_sub_vi_vi(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_gpu_value(b)
    return a - b
def h_sub_vi_t_vi(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a - b
def h_sub_t_f_t_f(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.sub(a, b)
def h_sub_t_vf_t_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.sub(a, b)
def h_sub_f_t_f(a, b):
    return torch.sub(a, b)
def h_sub_t_f_f(a, b):
    return torch.sub(a, b)
def h_sub_f_t_vf(a, b):
    return a - b
def h_sub_t_vf_f(a, b):
    return a - b
def h_sub_t_f_vf(a, b):
    m = len(a)
    n = len(b)
    a = torch.broadcast_to(a, (n, m)).T
    return a - b
def h_sub_vf_t_f(a, b):
    m = len(b)
    n = len(a)
    b = torch.broadcast_to(b, (n, m)).T
    return a - b
def h_sub_vf_t_vf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.sub(a, b)
def h_sub_t_vf_vf(a, b):
    a = get_gpu_value(a)
    b = get_vm_gpu_tensor(b)
    return torch.sub(a, b)
def h_sub_t_vf_t_f(a, b):
    m = len(b)
    n = len(a[0])
    b = torch.broadcast_to(b, (n, m)).T
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a - b
def h_sub_t_f_t_vf(a, b):
    m = len(a)
    n = len(b[0])
    a = torch.broadcast_to(a, (n, m)).T
    return a - b
def h_sub_mf_mf(a, b):
    return a - b
def h_sub_t_mf_mf(a, b):
    return a - b
def h_sub_i_f(a, b):
    return a - b

def h_mul_f_mf(a, b):
    a = get_s_gpu_tensor(a)
    b = get_vm_gpu_tensor(b)
    return a * b
def h_mul_mf_f(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return a * b
def h_mul_mf_t_mf(a, b):
    a = get_vm_gpu_tensor(a)
    b = get_gpu_value(b)
    return a * b
def h_mul_i_f(a, b):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return a * b
def h_mul_i_i(a, b):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return a * b
def h_mul_u_u(a, b):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return a * b
def h_mul_u_t_u(a, b):
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return a * b
def h_mul_t_f_i(a, b):
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return a * b
def h_mul_t_f_t_f(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.mul(a, b)
def h_mul_t_vf_t_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.mul(a, b)
def h_mul_t_vf_t_f(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.mul(a.T, b).T
def h_mul_t_f_t_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.mul(a, b.T).T
def h_mul_f_t_f(a, b):
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.mul(a, b)
def h_mul_t_f_f(a, b):
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return torch.mul(a, b)
def h_mul_f_t_i(a, b):
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.mul(a, b)
def h_mul_f_t_vf(a, b):
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return a * b
def h_mul_t_vf_f(a, b):
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return a * b
def h_mul_t_f_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    a = a.reshape(len(a), 1)
    return torch.mul(a, b)
def h_mul_vf_t_f(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    b = b.reshape(len(b), 1)
    return torch.mul(a, b)
def h_mul_vf_t_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.mul(a, b)
def h_mul_t_vf_vf(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.mul(a, b)
def h_mul_t_vu_vu(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.mul(a, b)
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_mul_t_vu_t_vu(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.mul(a, b)
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_mul_vu_vu(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.mul(a, b)
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_mul_vi_vi(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.mul(a, b)
    r = torch.bitwise_and(r, 0xffffffff)
    return int(r)
def h_mul_vd_vd(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.mul(a, b)
    return r
def h_mul_vb_vb(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.mul(a, b)
    return r
def h_mul_t_i_f(a, b):
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    r = a * b
    return r
def h_mul_t_i_t_f(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = a * b
    return r
def h_mul_t_f_t_i(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = a * b
    return r
def h_mul_t_u_i(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = a * b
    return r
def h_mul_t_u_t_u(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = a * b
    return r
def h_mul_t_u_t_vu(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = a * b
    return r

def h_div_i_i(a, b):
    a = get_s_gpu_tensor(a)
    b = get_s_gpu_tensor(b)
    return a // b
def h_div_t_i_i(a, b):
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return a // b
def h_div_t_f_t_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.div(a, b)
def h_div_t_vf_t_vf(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.div(a, b)
def h_div_f_t_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return torch.div(a, b)
def h_div_t_f_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return torch.div(a, b)
def h_div_f_t_vf(a, b):
    b = b + sys.float_info.epsilon
    a = get_s_gpu_tensor(a)
    b = get_gpu_value(b)
    return a / b
def h_div_t_vf_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_s_gpu_tensor(b)
    return a / b
def h_div_t_f_vf(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    a = a.reshape(len(a), 1)
    return torch.div(a, b)
def h_div_vf_t_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    b = b.reshape(len(b), 1)
    return torch.div(a, b)
def h_div_vf_t_vf(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.div(a, b)
def h_div_t_vf_vf(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.div(a, b)
def h_div_t_vf_t_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.div(a.T, b).T
    return r
def h_div_t_mf_t_f(a, b):
    b = b + sys.float_info.epsilon
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    r = torch.div(a.T, b).T
    return r

def h_mod_t_i_t_i(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.remainder(a, b)
def h_mod_t_vi_t_vi(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.remainder(a, b)
def h_mod_i_t_i(a, b):
    return torch.remainder(a, b)
def h_mod_t_i_i(a, b):
    return torch.remainder(a, b)
def h_mod_i_t_vi(a, b):
    return a % b
def h_mod_t_vi_i(a, b):
    return a % b
def h_mod_t_i_vi(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    a = a.reshape(len(a), 1)
    return torch.remainder(a, b)
def h_mod_vi_t_i(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    b = b.reshape(len(b), 1)
    return torch.remainder(a, b)
def h_mod_vi_t_vi(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.remainder(a, b)
def h_mod_t_vi_vi(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.remainder(a, b)

def h_bitand_i_i(a, b):
    return a & b
def h_bitand_t_i_i(a, b):
    return a & b
def h_bitand_t_vi_i(a, b):
    return a & b
def h_bitand_t_u_u(a, b):
    return a & b
def h_bitand_t_vu_vu(a, b):
    return a & b
def h_bitor_i_i(a, b):
    return a | b
def h_bitxor_i_i(a, b):
    return a ^ b
def h_bitxor_t_u_t_u(a, b):
    r = a ^ b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_bitxor_t_vu_t_vu(a, b):
    r = a ^ b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_bitxor_vu_vu(a, b):
    r = a ^ b
    r = torch.bitwise_and(r, 0xffffffff)
    return r

def h_lshift_i_i(a, b):
    return a << b
def h_rshift_i_i(a, b):
    return a >> b
def h_lshift_t_u_i(a, b):
    r = a << b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_rshift_t_u_i(a, b):
    r = a >> b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_rshift_t_vu_i(a, b):
    r = a >> b
    r = torch.bitwise_and(r, 0xffffffff)
    return r
def h_rshift_vu_i(a, b):
    r = a >> b
    r = torch.bitwise_and(r, 0xffffffff)
    return r

def h_less_than_n_n(a, b):
    return a < b
def h_greater_than_n_n(a, b):
    return a > b
def h_less_equal_than_n_n(a, b):
    return a <= b
def h_greater_equal_than_n_n(a, b):
    return a >= b

def h_less_than_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a < b
def h_greater_than_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a > b
def h_less_equal_than_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a <= b
def h_greater_equal_than_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a >= b
def h_less_than_t_n_t_n(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a < b
def h_greater_than_t_n_t_n(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a > b
def h_less_equal_than_t_n_t_n(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a <= b
def h_greater_equal_than_t_n_t_n(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return a >= b
def h_less_than_t_n_n(a, b):
    return a < b
def h_greater_than_t_n_n(a, b):
    return a > b
def h_less_equal_than_t_n_n(a, b):
    return a <= b
def h_greater_equal_than_t_n_n(a, b):
    return a >= b
def h_less_than_n_t_n(a, b):
    return a < b
def h_greater_than_n_t_n(a, b):
    return a > b
def h_less_equal_than_n_t_n(a, b):
    return a <= b
def h_greater_equal_than_n_t_n(a, b):
    return a >= b

def h_equal_n_n(a, b):
    return a == b
def h_not_equal_n_n(a, b):
    return a != b
def h_equal_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.all(a == b)
def h_not_equal_v_v(a, b):
    a = get_gpu_value(a)
    b = get_gpu_value(b)
    return torch.any(a != b)

def h_equal_t_n_n(a, b):
    return a == b
def h_not_equal_t_n_n(a, b):
    return a != b
def h_equal_t_v_v(a, b):
    return a == b
def h_not_equal_t_v_v(a, b):
    return a != b
def h_equal_n_t_n(a, b):
    return a == b
def h_not_equal_n_t_n(a, b):
    return a != b
def h_equal_t_n_t_n(a, b):
    return a == b
def h_not_equal_t_n_t_n(a, b):
    return a != b

def h_broadcast_t_f_f(t, v):
    return torch.broadcast_to(get_s_gpu_tensor(v), [len(t)])
def h_broadcast_t_b_b(t, v):
    return torch.broadcast_to(get_s_gpu_tensor(v), [len(t)])
def h_broadcast_t_i_i(t, v):
    return torch.broadcast_to(get_s_gpu_tensor(v), [len(t)])
def h_broadcast_t_f2_f2(t, v):
    return torch.broadcast_to(get_vm_gpu_tensor(v), (len(t), 2))
def h_broadcast_t_f3_f3(t, v):
    return torch.broadcast_to(get_vm_gpu_tensor(v), (len(t), 3))
def h_broadcast_t_f4_f4(t, v):
    return torch.broadcast_to(get_vm_gpu_tensor(v), (len(t), 4))
def h_broadcast_f(ct, v):
    return torch.broadcast_to(get_s_gpu_tensor(v), [ct])
def h_broadcast_b(ct, v):
    return torch.broadcast_to(get_s_gpu_tensor(v), [ct])
def h_broadcast_i(ct, v):
    return torch.broadcast_to(get_s_gpu_tensor(v), [ct])
def h_broadcast_f2(ct, v):
    return torch.broadcast_to(get_vm_gpu_tensor(v), (ct, 2))
def h_broadcast_f3(ct, v):
    return torch.broadcast_to(get_vm_gpu_tensor(v), (ct, 3))
def h_broadcast_f4(ct, v):
    return torch.broadcast_to(get_vm_gpu_tensor(v), (ct, 4))
def h_copy_f(v):
    return v
def h_copy_f2(v):
    return torch.clone(v)
def h_copy_f3(v):
    return torch.clone(v)
def h_copy_f4(v):
    return torch.clone(v)
def h_copy_t_f(v):
    return torch.clone(v)
def h_copy_t_f2(v):
    return torch.clone(v)
def h_copy_t_f3(v):
    return torch.clone(v)
def h_copy_t_f4(v):
    return torch.clone(v)
def h_copy_t_i(v):
    return torch.clone(v)
def h_copy_t_i2(v):
    return torch.clone(v)
def h_copy_t_i3(v):
    return torch.clone(v)
def h_copy_t_i4(v):
    return torch.clone(v)

def h_cast_f_h(v):
    return float(v)
def h_cast_f_d(v):
    return float(v)
def h_cast_f_i(v):
    return float(v)
def h_cast_f_u(v):
    return float(v)
def h_cast_f_b(v):
    return 1.0 if v else 0.0

def h_cast_i_h(v):
    return int(v)
def h_cast_i_f(v):
    return int(v)
def h_cast_i_d(v):
    return int(v)
def h_cast_i_u(v):
    return int(v)
def h_cast_i_b(v):
    return 1 if v else 0

def h_cast_u_h(v):
    return int(v)+(1<<32)
def h_cast_u_f(v):
    return int(v)+(1<<32)
def h_cast_u_d(v):
    return int(v)+(1<<32)
def h_cast_u_i(v):
    r = v if v >= 0 else v + (1 << 32)
    return r
def h_cast_u_b(v):
    return 1 if v else 0

def h_cast_b_h(v):
    return int(v) != 0
def h_cast_b_f(v):
    return int(v) != 0
def h_cast_b_d(v):
    return int(v) != 0
def h_cast_b_i(v):
    return int(v) != 0
def h_cast_b_u(v):
    return int(v) != 0

def h_cast_f2_i2(v):
    return v.float()
def h_cast_f2_h2(v):
    return v.float()
def h_cast_f2_d2(v):
    return v.float()
def h_cast_f2_u2(v):
    return v.float()
def h_cast_f2_b2(v):
    return v.float()
def h_cast_f2_f(v):
    return get_vm_gpu_tensor([v, v])

def h_cast_f3_i3(v):
    return v.float()
def h_cast_f3_h3(v):
    return v.float()
def h_cast_f3_d3(v):
    return v.float()
def h_cast_f3_u3(v):
    return v.float()
def h_cast_f3_b3(v):
    return v.float()

def h_cast_f4_i4(v):
    return v.float()
def h_cast_f4_h4(v):
    return v.float()
def h_cast_f4_d4(v):
    return v.float()
def h_cast_f4_u4(v):
    return v.float()
def h_cast_f4_b4(v):
    return v.float()

def h_cast_d2_i2(v):
    return v.float()
def h_cast_d2_h2(v):
    return v.float()
def h_cast_d2_f2(v):
    return v.float()
def h_cast_d2_u2(v):
    return v.float()
def h_cast_d2_b2(v):
    return v.float()

def h_cast_d3_i3(v):
    return v.float()
def h_cast_d3_h3(v):
    return v.float()
def h_cast_d3_f3(v):
    return v.float()
def h_cast_d3_u3(v):
    return v.float()
def h_cast_d3_b3(v):
    return v.float()

def h_cast_d4_i4(v):
    return v.float()
def h_cast_d4_h4(v):
    return v.float()
def h_cast_d4_f4(v):
    return v.float()
def h_cast_d4_u4(v):
    return v.float()
def h_cast_d4_b4(v):
    return v.float()

def h_cast_i2_h2(v):
    return v.int()
def h_cast_i2_f2(v):
    return v.int()
def h_cast_i2_d2(v):
    return v.int()
def h_cast_i2_u2(v):
    return v.int()
def h_cast_i2_b2(v):
    return v.int()

def h_cast_i3_h3(v):
    return v.int()
def h_cast_i3_f3(v):
    return v.int()
def h_cast_i3_d3(v):
    return v.int()
def h_cast_i3_u3(v):
    return v.int()
def h_cast_i3_b3(v):
    return v.int()

def h_cast_i4_h4(v):
    return v.int()
def h_cast_i4_f4(v):
    return v.int()
def h_cast_i4_d4(v):
    return v.int()
def h_cast_i4_u4(v):
    return v.int()
def h_cast_i4_b4(v):
    return v.int()

def h_cast_u2_h2(v):
    return v.int()
def h_cast_u2_f2(v):
    return v.int()
def h_cast_u2_d2(v):
    return v.int()
def h_cast_u2_i2(v):
    return v.int()
def h_cast_u2_b2(v):
    return v.int()

def h_cast_t_u_t_i(v):
    return torch.abs(v).type(torch.int64)
def h_cast_t_u2_t_i2(v):
    return torch.abs(v).type(torch.int64)
def h_cast_t_u_t_f(v):
    return torch.abs(v).type(torch.int64)
def h_cast_t_u2_t_f2(v):
    return torch.abs(v).type(torch.int64)

def h_cast_u3_h3(v):
    return v.int()
def h_cast_u3_f3(v):
    return v.int()
def h_cast_u3_d3(v):
    return v.int()
def h_cast_u3_i3(v):
    return v.int()
def h_cast_u3_b3(v):
    return v.int()

def h_cast_u4_h4(v):
    return v.int()
def h_cast_u4_f4(v):
    return v.int()
def h_cast_u4_d4(v):
    return v.int()
def h_cast_u4_i4(v):
    return v.int()
def h_cast_u4_b4(v):
    return v.int()

def h_cast_h2_f2(v):
    return v.float()
def h_cast_h2_d2(v):
    return v.float()
def h_cast_h2_i2(v):
    return v.float()
def h_cast_h2_u2(v):
    return v.float()
def h_cast_h2_b2(v):
    return v.float()

def h_cast_h3_f3(v):
    return v.float()
def h_cast_h3_d3(v):
    return v.float()
def h_cast_h3_i3(v):
    return v.float()
def h_cast_h3_u3(v):
    return v.float()
def h_cast_h3_b3(v):
    return v.float()

def h_cast_h4_f4(v):
    return v.float()
def h_cast_h4_d4(v):
    return v.float()
def h_cast_h4_i4(v):
    return v.float()
def h_cast_h4_u4(v):
    return v.float()
def h_cast_h4_b4(v):
    return v.float()

def h_cast_b2_h2(v):
    return v.int()
def h_cast_b2_f2(v):
    return v.int()
def h_cast_b2_d2(v):
    return v.int()
def h_cast_b2_i2(v):
    return v.int()
def h_cast_b2_u2(v):
    return v.int()

def h_cast_b3_h3(v):
    return v.int()
def h_cast_b3_f3(v):
    return v.int()
def h_cast_b3_d3(v):
    return v.int()
def h_cast_b3_i3(v):
    return v.int()
def h_cast_b3_u3(v):
    return v.int()

def h_cast_b4_h4(v):
    return v.int()
def h_cast_b4_f4(v):
    return v.int()
def h_cast_b4_d4(v):
    return v.int()
def h_cast_b4_i4(v):
    return v.int()
def h_cast_b4_u4(v):
    return v.int()

def h_cast_f_f(v):
    return v
def h_cast_f2_f2(v):
    return v
def h_cast_f3_f3(v):
    return v
def h_cast_f4_f4(v):
    return v

def h_cast_f2_f3(v):
    return torch.asarray([v[0], v[1]])
def h_cast_f3_f4(v):
    return torch.asarray([v[0], v[1], v[2]])

def h_cast_i_i(v):
    return v
def h_cast_i2_i2(v):
    return v
def h_cast_i3_i3(v):
    return v
def h_cast_i4_i4(v):
    return v

def h_cast_b_b(v):
    return v
def h_cast_b2_b2(v):
    return v
def h_cast_b3_b3(v):
    return v
def h_cast_b4_b4(v):
    return v

def h_cast_f3x3_i_x9(v):
    return h_f3x3_n_n_n_n_n_n_n_n_n(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
def h_cast_t_f3x3_t_f_x9(v):
    return h_t_f3x3_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
def h_cast_f4x4_i_x16(v):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15])

def h_cast_t_f_t_f(v):
    return v
def h_cast_t_f2_t_f2(v):
    return v
def h_cast_t_f3_t_f3(v):
    return v
def h_cast_t_f4_t_f4(v):
    return v
def h_cast_t_i_t_i(v):
    return v
def h_cast_t_i2_t_i2(v):
    return v
def h_cast_t_i3_t_i3(v):
    return v
def h_cast_t_i4_t_i4(v):
    return v
def h_cast_t_f_t_i(v):
    return v.float()
def h_cast_t_f_t_u(v):
    return v.float()
def h_cast_t_i_t_f(v):
    return v.int()
def h_cast_t_i_t_f(v):
    return v.int()
def h_cast_t_i2_t_f2(v):
    return v.int()
def h_cast_t_i3_t_f3(v):
    return v.int()
def h_cast_t_i4_t_f4(v):
    return v.int()
def h_cast_t_f2_t_u2(v):
    return v.float()
def h_cast_t_f3_t_u3(v):
    return v.float()
def h_cast_t_f_t_b(v):
    return v.float()


def h_f2_n_n(x, y):
    return torch.asarray([x, y], device=device)

def h_t_f2_t_n_n(x, y):
    xs = x
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    xs = get_gpu_value(xs)
    ys = get_gpu_value(ys)
    nv = torch.column_stack((xs, ys))
    return nv

def h_t_f2_n_t_n(x, y):    
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(y)])
    ys = y
    xs = get_gpu_value(xs)
    ys = get_gpu_value(ys)
    nv = torch.column_stack((xs, ys))
    return nv
def h_t_f2_t_n_t_n(x, y):
    nv = torch.column_stack((x, y))
    nv = get_gpu_value(nv)
    return nv

def h_f3_x_y(x, y):
    xisv = maybe_svm_array(x)
    yisv = maybe_svm_array(y)
    if xisv or yisv:
        nx = 0
        ny = 0
        if xisv:
            nx = len(x)
        if yisv:
            ny = len(y)
        mn = max(nx, ny)
        xs = x
        if not xisv:
            xs = torch.broadcast_to(get_s_gpu_tensor(x), [mn])
        ys = y
        if not yisv:
            ys = torch.broadcast_to(get_s_gpu_tensor(y), [mn])
        xs = get_gpu_value(xs)
        ys = get_gpu_value(ys)
        nv = torch.column_stack((xs, ys))
        return nv
    else:
        if type(x) == torch.Tensor and x.dim() > 0 and len(x) >= 2:
            return torch.asarray([x[0], x[1], y], device=device)
        elif type(y) == torch.Tensor and y.dim() > 0 and len(y) >= 2:
            return torch.asarray([x, y[0], y[1]], device=device)
        else:
            return torch.asarray([x, y, 1.0], device=device)
def h_f3_x_y_z(x, y, z):
    xisv = maybe_svm_array(x)
    yisv = maybe_svm_array(y)
    zisv = maybe_svm_array(z)
    if xisv or yisv or zisv:
        nx = 0
        ny = 0
        nz = 0
        if xisv:
            nx = len(x)
        if yisv:
            ny = len(y)
        if zisv:
            nz = len(z)
        mn = max(nx, ny, nz)
        xs = x
        if not xisv:
            xs = torch.broadcast_to(get_s_gpu_tensor(x), [mn])
        ys = y
        if not yisv:
            ys = torch.broadcast_to(get_s_gpu_tensor(y), [mn])
        zs = z
        if not zisv:
            zs = torch.broadcast_to(get_s_gpu_tensor(z), [mn])
        nv = torch.column_stack((xs, ys, zs))
        return nv
    else:
        return torch.asarray([x, y, z], device=device)
def h_f3_n2_n(x, y):
    return torch.asarray([x[0], x[1], y], device=device)
def h_f3_n_n2(x, y):
    return torch.asarray([x, y[0], y[1]], device=device)
def h_f3_n_n(x, y):
    return torch.asarray([x, y, 1.0], device=device)
def h_f3_n_n_n(x, y, z):
    return torch.asarray([x, y, z], device=device)

def h_t_f3_t_n_t_n_t_n(x, y, z):
    return torch.column_stack((x, y, z))
def h_t_f3_t_n2_t_n(x, y):
    xs = get_gpu_value(x)
    ys = get_gpu_value(y)
    nv = torch.column_stack((xs, ys))
    return nv
def h_t_f3_t_n_t_n2(x, y):
    xs = get_gpu_value(x)
    ys = get_gpu_value(y)
    nv = torch.column_stack((xs, ys))
    return nv
def h_t_f3_t_n2_n(x, y):
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    ys = get_gpu_value(ys)
    return torch.column_stack((x, ys))
def h_t_f3_t_n_t_n_n(x, y, z):
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(x)])
    zs = get_gpu_value(zs)
    return torch.column_stack((x, y, zs))
def h_t_f3_t_n_n_n(x, y, z):
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(x)])
    ys = get_gpu_value(ys)
    zs = get_gpu_value(zs)
    return torch.column_stack((x, ys, zs))
def h_t_f3_t_n_n_t_n(x, y, z):
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    ys = get_gpu_value(ys)
    return torch.column_stack((x, ys, z))
def h_t_f3_n_t_n_n(x, y, z):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(y)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(y)])
    xs = get_gpu_value(xs)
    zs = get_gpu_value(zs)
    return torch.column_stack((xs, y, zs))
def h_t_f3_n_t_n_t_n(x, y, z):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(y)])
    xs = get_gpu_value(xs)
    return torch.column_stack((xs, y, z))
def h_t_f3_n_t_n2(x, yz):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(yz)])
    xs = get_gpu_value(xs)
    return torch.column_stack((xs, yz))

def h_f4_x_y(x, y):
    xisv = maybe_svm_array(x)
    yisv = maybe_svm_array(y)
    if xisv or yisv:
        nx = 0
        ny = 0
        if xisv:
            nx = len(x)
        if yisv:
            ny = len(y)
        mn = max(nx, ny)
        xs = x
        if not xisv:
            xs = torch.repeat_interleave(x, mn)
        ys = y
        if not yisv:
            ys = torch.repeat_interleave(y, mn)
        nv = torch.column_stack((xs, ys))
        return nv
    else:
        if type(x) == torch.Tensor and x.dim() > 0 and type(y) == torch.Tensor and y.dim() > 0:
            return torch.asarray([x[0], x[1], y[0], y[1]], device=device)
        elif type(x) == torch.Tensor and x.dim() > 0:
            return torch.asarray([x[0], x[1], x[2], y], device=device)
        elif type(y) == torch.Tensor and y.dim() > 0:
            return torch.asarray([x, y[0], y[1], y[2]], device=device)
def h_f4_x_y_z(x, y, z):
    xisv = maybe_svm_array(x)
    yisv = maybe_svm_array(y)
    zisv = maybe_svm_array(z)
    if xisv or yisv or zisv:
        nx = 0
        ny = 0
        nz = 0
        if xisv:
            nx = len(x)
        if yisv:
            ny = len(y)
        if zisv:
            nz = len(z)
        mn = max(nx, ny, nz)
        xs = x
        if not xisv:
            xs = torch.repeat_interleave(x, mn)
        ys = y
        if not yisv:
            ys = torch.repeat_interleave(y, mn)
        zs = z
        if not zisv:
            zs = torch.repeat_interleave(z, mn)
        ws = torch.repeat_interleave(1.0, mn)
        nv = torch.column_stack((xs, ys, zs, ws))
        return nv
    else:
        return torch.asarray([x, y, z, 1.0], device=device)
def h_f4_x_y_z_w(x, y, z, w):
    xisv = maybe_svm_array(x)
    yisv = maybe_svm_array(y)
    zisv = maybe_svm_array(z)
    wisv = maybe_svm_array(w)
    if xisv or yisv or zisv or wisv:
        nx = 0
        ny = 0
        nz = 0
        nw = 0
        if xisv:
            nx = len(x)
        if yisv:
            ny = len(y)
        if zisv:
            nz = len(z)
        if wisv:
            nw = len(w)
        mn = max(nx, ny, nz, nw)
        xs = x
        if not xisv:
            xs = torch.repeat_interleave(x, mn)
        ys = y
        if not yisv:
            ys = torch.repeat_interleave(y, mn)
        zs = z
        if not zisv:
            zs = torch.repeat_interleave(z, mn)
        ws = w
        if not wisv:
            ws = torch.repeat_interleave(w, mn)
        nv = torch.column_stack((xs, ys, zs, ws))
        return nv
    else:
        return torch.asarray([x, y, z, w], device=device)

def h_f4_n3_n(x, y):
    return torch.asarray([x[0], x[1], x[2], y], device=device)
def h_f4_n2_n2(x, y):
    return torch.asarray([x[0], x[1], y[0], y[1]], device=device)
def h_f4_n_n3(x, y):
    return torch.asarray([x, y[0], y[1], y[2]], device=device)
def h_f4_n2_n_n(x, y, z):
    return torch.asarray([x[0], x[1], y, z], device=device)
def h_f4_n_n_n(x, y, z):
    return torch.asarray([x, y, z, 1.0], device=device)
def h_f4_n_n_n_n(x, y, z, w):
    return torch.asarray([x, y, z, w], device=device)

def h_t_f4_t_n_t_n_t_n_t_n(x, y, z, w):
    return torch.column_stack((x, y, z, w))
def h_t_f4_t_n_t_n_t_n_n(x, y, z, w):
    w = torch.broadcast_to(get_s_gpu_tensor(w), [len(x)])
    w = get_gpu_value(w)
    return torch.column_stack((x, y, z, w))
def h_t_f4_t_n2_t_n2(x, y):
    return torch.column_stack((x, y))
def h_t_f4_t_n2_t_n_n(xy, z, w):
    w = torch.broadcast_to(get_s_gpu_tensor(w), [len(xy)])
    w = get_gpu_value(w)
    return torch.column_stack((xy, z, w))
def h_t_f4_t_n2_n_t_n(xy, z, w):
    z = torch.broadcast_to(get_s_gpu_tensor(z), [len(xy)])
    return torch.column_stack((xy, z, w))
def h_t_f4_t_n2_n_n(xy, z, w):
    z = torch.broadcast_to(get_s_gpu_tensor(z), [len(xy)])
    w = torch.broadcast_to(get_s_gpu_tensor(w), [len(xy)])
    return torch.column_stack((xy, z, w))
def h_t_f4_t_n3_n(x, y):
    y = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    x = get_gpu_value(x)
    y = get_gpu_value(y)
    return torch.column_stack((x, y))
def h_t_f4_t_n_t_n3(x, y):
    return torch.column_stack((x, y))
def h_t_f4_t_n3_t_n(x, y):
    return torch.column_stack((x, y))
def h_t_f4_t_n_t_n_n_n(x, y, z, w):
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(x)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(x)])
    zs = get_gpu_value(zs)
    ws = get_gpu_value(ws)
    return torch.column_stack((x, y, zs, ws))

def h_t_f4_t_n_n_t_n_n(x, y, z, w):
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(x)])
    ys = get_gpu_value(ys)
    ws = get_gpu_value(ws)
    return torch.column_stack((x, ys, z, ws))
def h_t_f4_t_n_n_n_t_n(x, y, z, w):
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(x)])
    ys = get_gpu_value(ys)
    zs = get_gpu_value(zs)
    return torch.column_stack((x, ys, zs, w))
def h_t_f4_n_t_n_t_n_n(x, y, z, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(y)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(y)])
    xs = get_gpu_value(xs)
    ws = get_gpu_value(ws)
    return torch.column_stack((xs, y, z, ws))
def h_t_f4_n_t_n_n_t_n(x, y, z, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(w)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(w)])
    xs = get_gpu_value(xs)
    zs = get_gpu_value(zs)
    return torch.column_stack((xs, y, zs, w))
def h_t_f4_n_n_t_n_t_n(x, y, z, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(w)])
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(w)])
    xs = get_gpu_value(xs)
    ys = get_gpu_value(ys)
    return torch.column_stack((xs, ys, z, w))
def h_t_f4_t_n_n_n_n(x, y, z, w):
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(x)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(x)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(x)])
    ys = get_gpu_value(ys)
    zs = get_gpu_value(zs)
    ws = get_gpu_value(ws)
    return torch.column_stack((x, ys, zs, ws))
def h_t_f4_n_t_n_n_n(x, y, z, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(y)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(y)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(y)])
    xs = get_gpu_value(xs)
    zs = get_gpu_value(zs)
    ws = get_gpu_value(ws)
    return torch.column_stack((xs, y, zs, ws))
def h_t_f4_n_n_t_n_n(x, y, z, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(z)])
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(z)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(z)])
    xs = get_gpu_value(xs)
    ys = get_gpu_value(ys)
    ws = get_gpu_value(ws)
    return torch.column_stack((xs, ys, z, ws))
def h_t_f4_n_n_n_t_n(x, y, z, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(w)])
    ys = torch.broadcast_to(get_s_gpu_tensor(y), [len(w)])
    zs = torch.broadcast_to(get_s_gpu_tensor(z), [len(w)])
    xs = get_gpu_value(xs)
    ys = get_gpu_value(ys)
    zs = get_gpu_value(zs)
    return torch.column_stack((xs, ys, zs, w))
def h_t_f4_n3_t_n(xyz, w):
    xyz = torch.broadcast_to(get_vm_gpu_tensor(xyz), (len(w), 3))
    xyz = get_gpu_value(xyz)
    w = get_gpu_value(w)
    return torch.column_stack((xyz, w))
def h_t_f4_t_n2_t_n_n(xy, z, w):
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(z)])
    ws = get_gpu_value(ws)
    return torch.column_stack((xy, z, ws))
def h_t_f4_n_t_n2_n(x, yz, w):
    xs = torch.broadcast_to(get_s_gpu_tensor(x), [len(yz)])
    ws = torch.broadcast_to(get_s_gpu_tensor(w), [len(yz)])
    xs = get_gpu_value(xs)
    ws = get_gpu_value(ws)
    return torch.column_stack((xs, yz, ws))

def h_d2_n_n(x, y):
    return h_f2_n_n(x, y)

def h_d3_n2_n(x, y):
    return h_f3_n2_n(x, y)
def h_d3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)

def h_d4_n3_n(x, y):
    return h_f4_n3_n(x, y)
def h_d4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
def h_d4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

def h_h2_n_n(x, y):
    return h_f2_n_n(x, y)

def h_h3_n2_n(x, y):
    return h_f3_n2_n(x, y)
def h_h3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)

def h_h4_n3_n(x, y):
    return h_f4_n3_n(x, y)
def h_h4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
def h_h4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

def h_i2_n_n(x, y):
    return h_f2_n_n(x, y)

def h_i3_n2_n(x, y):
    return h_f3_n2_n(x, y)
def h_i3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)
def h_t_i3_t_n2_n(x, y):
    return h_t_f3_t_n2_n(x, y)

def h_i4_n3_n(x, y):
    return h_f4_n3_n(x, y)
def h_i4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
def h_i4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

def h_b2_n_n(x, y):
    return h_f2_n_n(x, y)

def h_b3_n2_n(x, y):
    return h_f3_n2_n(x, y)
def h_b3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)

def h_b4_n3_n(x, y):
    return h_f4_n3_n(x, y)
def h_b4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
def h_b4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

def h_u2_n_n(x, y):
    return h_f2_n_n(x, y)
def h_t_u2_t_n_t_n(x, y):
    return torch.column_stack((x, y))

def h_u3_n2_n(x, y):
    return h_f3_n2_n(x, y)
def h_u3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)
def h_t_u3_t_n_t_n_t_n(x, y, z):
    return torch.column_stack((x, y, z))

def h_u4_n3_n(x, y):
    return h_f4_n3_n(x, y)
def h_u4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
def h_u4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

def h_f2x2_n_n_n_n(x1, y1, x2, y2):
    return torch.asarray([[x1, y1], [x2, y2]], device=device)
def h_t_f2x2_t_n_t_n_t_n_t_n(x1, y1, x2, y2):
    r1 = torch.column_stack((x1, y1))
    r2 = torch.column_stack((x2, y2))
    return torch.stack((r1, r2), axis=1)

def h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return torch.asarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], device=device)
def h_f3x3_n3_n3_n3(x, y, z):
    return torch.stack((x, y, z))
def h_t_f3x3_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    r1 = torch.column_stack((x1, y1, z1))
    r2 = torch.column_stack((x2, y2, z2))
    r3 = torch.column_stack((x3, y3, z3))
    return torch.stack((r1, r2, r3), axis=1)
def h_t_f3x3_t_n3_t_n3_t_n3(v1, v2, v3):
    return torch.stack((v1, v2, v3), axis=1)

def h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return torch.asarray([[x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]], device=device)
def h_f4x4_n4_n4_n4_n4(x, y, z, w):
    return torch.stack((x, y, z, w))
def h_t_f4x4_t_n_t_n_t_n_n_t_n_t_n_t_n_n_t_n_t_n_t_n_n_n_n_n_n(xs1, ys1, zs1, w1, xs2, ys2, zs2, w2, xs3, ys3, zs3, w3, x4, y4, z4, w4):
    n = len(xs1)
    ws1 = torch.broadcast_to(get_s_gpu_tensor(w1), [n])
    ws2 = torch.broadcast_to(get_s_gpu_tensor(w2), [n])
    ws3 = torch.broadcast_to(get_s_gpu_tensor(w3), [n])
    xs4 = torch.broadcast_to(get_s_gpu_tensor(x4), [n])
    ys4 = torch.broadcast_to(get_s_gpu_tensor(y4), [n])
    zs4 = torch.broadcast_to(get_s_gpu_tensor(z4), [n])
    ws4 = torch.broadcast_to(get_s_gpu_tensor(w4), [n])
    r1 = torch.column_stack((xs1, ys1, zs1, ws1))
    r2 = torch.column_stack((xs2, ys2, zs2, ws2))
    r3 = torch.column_stack((xs3, ys3, zs3, ws3))
    r4 = torch.column_stack((xs4, ys4, zs4, ws4))
    return torch.stack((r1, r2, r3, r4), axis=1)

def h_d2x2_n_n_n_n(x1, y1, x2, y2):
    return torch.asarray([[x1, y1], [x2, y2]], device=device)

def h_d3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return torch.asarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], device=device)

def h_d4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return torch.asarray([[x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]], device=device)

def h_h2x2_n_n_n_n(x1, y1, x2, y2):
    return torch.asarray([[x1, y1], [x2, y2]], device=device)

def h_h3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return torch.asarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], device=device)

def h_h4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return torch.asarray([[x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]], device=device)

def h_i2x2_n_n_n_n(x1, y1, x2, y2):
    return h_f2x2_n_n_n_n(x1, y1, x2, y2)
def h_i3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3)
def h_i4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4)

def h_b2x2_n_n_n_n(x1, y1, x2, y2):
    return h_f2x2_n_n_n_n(x1, y1, x2, y2)
def h_b3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3)
def h_b4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4)

def h_u2x2_n_n_n_n(x1, y1, x2, y2):
    return h_f2x2_n_n_n_n(x1, y1, x2, y2)
def h_u3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3)
def h_u4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4)

def h_f2_defval():
    return new_zero_tensor(2)
def h_f3_defval():
    return new_zero_tensor(3)
def h_f4_defval():
    return new_zero_tensor(4)
def h_f2x2_defval():
    return new_zero_tensor(2, 2)
def h_f3x3_defval():
    return new_zero_tensor(3, 3)
def h_f4x4_defval():
    return new_zero_tensor(4, 4)
def h_af_defval(num):
    return new_zero_tensor(num)
def h_af2_defval(num):
    return new_zero_tensor(num, 2)
def h_af3_defval(num):
    return new_zero_tensor(num, 3)
def h_af4_defval(num):
    return new_zero_tensor(num, 4)
def h_ab_defval(num):
    return new_false_tensor(num)
def h_ai_defval(num):
    return new_zero_int_tensor(num)

'''
def h_f2_defval():
    return torch.asarray([0.0, 0.0], device=device)
def h_f3_defval():
    return torch.asarray([0.0, 0.0, 0.0], device=device)
def h_f4_defval():
    return torch.asarray([0.0, 0.0, 0.0, 0.0], device=device)
def h_f2x2_defval():
    return torch.asarray([[0.0, 0.0], [0.0, 0.0]], device=device)
def h_f3x3_defval():
    return torch.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
def h_f4x4_defval():
    return torch.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], device=device)
def h_af3_defval(num):
    return torch.repeat_interleave([0.0, 0.0, 0.0], [num], device=device)
def h_af_defval(num):
    return torch.repeat_interleave(0.0, [num], device=device)
def h_ab_defval(num):
    return torch.repeat_interleave(False, [num], device=device)
'''

def Texture2D_Sample_n_v(tex, sampler, coord):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:2]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).int(), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == torch.Tensor:
        if col.dim() == 1 and len(col) == 3:
            return get_gpu_value(torch.asarray([col[0], col[1], col[2], 1.0]))
        else:
            return col
    return torch.as_tensor([col, col, col, 1.0], device=device)
def Texture2D_SampleBias_n_v_n(tex, sampler, coord, bias):
    return Texture2D_Sample_n_v(tex, sampler, coord)
def Texture2D_Sample_n_t_v(tex, sampler, coord):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:2]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).long(), texSize)
    icoord = tuple(torch.transpose(icoord, 0, 1))
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = torch.column_stack((cols, torch.broadcast_to(gpu_one_tensor, [len(cols)])))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        ones = get_gpu_ones(len(cols))
        nv = torch.column_stack((cols, cols, cols, ones))
        return nv

def Texture2D_SampleBias_n_t_v_n(tex, sampler, coord, bias):
    return Texture2D_Sample_n_t_v(tex, sampler, coord)
def Texture2D_SampleLevel_n_v_n(tex, sampler, coord, level):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:2]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).int(), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == torch.Tensor:
        if col.dim() == 1 and len(col) == 3:
            return get_gpu_value(torch.asarray([col[0], col[1], col[2], 1.0]))
        else:
            return col
    return torch.as_tensor([col, col, col, 1.0], device=device)

def Texture2D_SampleLevel_n_t_v_n(tex, sampler, coord, level):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:2]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).long(), texSize)
    icoord = tuple(torch.transpose(icoord, 0, 1))
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = torch.column_stack((cols, torch.broadcast_to(gpu_one_tensor, [len(cols)])))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        ones = get_gpu_ones(len(cols))
        nv = torch.column_stack((cols, cols, cols, ones))
        return nv

def Texture2D_SampleGrad_n_t_v_t_v_t_v(tex, sampler, coord, ddx, ddy):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:2]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).long(), texSize)
    icoord = tuple(torch.transpose(icoord, 0, 1))
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = torch.column_stack((cols, torch.broadcast_to(gpu_one_tensor, [len(cols)])))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        ones = get_gpu_ones(len(cols))
        nv = torch.column_stack((cols, cols, cols, ones))
        return nv
def Texture2D_Load_t_v(tex, coord):
    texSize = get_vm_gpu_tensor(tex.shape)
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    n = len(coord)
    icoord = torch.remainder(coord[..., 0:2].int(), texSize[0:2])
    icoord = tuple(torch.transpose(icoord, 0, 1))
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = torch.column_stack((cols, torch.broadcast_to(gpu_one_tensor, [len(cols)])))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        ones = get_gpu_ones(len(cols))
        nv = torch.column_stack((cols, cols, cols, ones))
        return nv
def TextureCube_Sample_n_v(tex, sampler, coord):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:3]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).int(), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == torch.Tensor:
        if col.dim() == 1 and len(col) == 3:
            return get_gpu_value(torch.asarray([col[0], col[1], col[2], 1.0]))
        else:
            return col
    return torch.as_tensor([col, col, col, 1.0], device=device)

def TextureCube_Sample_n_t_v(tex, sampler, coord):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:3]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).long(), texSize)
    icoord = tuple(torch.transpose(icoord, 0, 1))
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = torch.column_stack((cols, torch.broadcast_to(gpu_one_tensor, [len(cols)])))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        ones = get_gpu_ones(len(cols))
        nv = torch.column_stack((cols, cols, cols, ones))
        return nv

def TextureCube_SampleLevel_n_t_v_n(tex, sampler, coord, level):
    return TextureCube_Sample_n_t_v(tex, sampler, coord)
def Texture3D_Sample_n_v(tex, sampler, coord):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:3]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).int(), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == torch.Tensor:
        if col.dim() == 1 and len(col) == 3:
            return get_gpu_value(torch.asarray([col[0], col[1], col[2], 1.0]))
        else:
            return col
    return torch.as_tensor([col, col, col, 1.0], device=device)
def Texture3D_Sample_n_t_v(tex, sampler, coord):
    coord[torch.isnan(coord[...])] = 0
    coord = torch.remainder(coord, 1.0)
    tex = get_gpu_value(tex)
    texSize = get_vm_gpu_tensor(tex.shape)[0:3]
    texSize = get_gpu_value(texSize)
    coord = get_gpu_value(coord)
    icoord = torch.remainder((coord * texSize).long(), texSize)
    icoord = tuple(torch.transpose(icoord, 0, 1))
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = torch.column_stack((cols, torch.broadcast_to(gpu_one_tensor, [len(cols)])))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        ones = get_gpu_ones(len(cols))
        nv = torch.column_stack((cols, cols, cols, ones))
        return nv

