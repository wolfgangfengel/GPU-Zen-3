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


#----------------------------------------
# these code generated from gen_hlsl_lib_numpy_swizzle.dsl
#---begin---

g_x_index = torch.asarray([0], device=device).squeeze(0)
g_y_index = torch.asarray([1], device=device).squeeze(0)
g_z_index = torch.asarray([2], device=device).squeeze(0)
g_w_index = torch.asarray([3], device=device).squeeze(0)
g_xx_index = torch.asarray([0, 0], device=device)
g_xy_index = torch.asarray([0, 1], device=device)
g_xz_index = torch.asarray([0, 2], device=device)
g_xw_index = torch.asarray([0, 3], device=device)
g_yx_index = torch.asarray([1, 0], device=device)
g_yy_index = torch.asarray([1, 1], device=device)
g_yz_index = torch.asarray([1, 2], device=device)
g_yw_index = torch.asarray([1, 3], device=device)
g_zx_index = torch.asarray([2, 0], device=device)
g_zy_index = torch.asarray([2, 1], device=device)
g_zz_index = torch.asarray([2, 2], device=device)
g_zw_index = torch.asarray([2, 3], device=device)
g_wx_index = torch.asarray([3, 0], device=device)
g_wy_index = torch.asarray([3, 1], device=device)
g_wz_index = torch.asarray([3, 2], device=device)
g_ww_index = torch.asarray([3, 3], device=device)
g_xxx_index = torch.asarray([0, 0, 0], device=device)
g_xxy_index = torch.asarray([0, 0, 1], device=device)
g_xxz_index = torch.asarray([0, 0, 2], device=device)
g_xxw_index = torch.asarray([0, 0, 3], device=device)
g_xyx_index = torch.asarray([0, 1, 0], device=device)
g_xyy_index = torch.asarray([0, 1, 1], device=device)
g_xyz_index = torch.asarray([0, 1, 2], device=device)
g_xyw_index = torch.asarray([0, 1, 3], device=device)
g_xzx_index = torch.asarray([0, 2, 0], device=device)
g_xzy_index = torch.asarray([0, 2, 1], device=device)
g_xzz_index = torch.asarray([0, 2, 2], device=device)
g_xzw_index = torch.asarray([0, 2, 3], device=device)
g_xwx_index = torch.asarray([0, 3, 0], device=device)
g_xwy_index = torch.asarray([0, 3, 1], device=device)
g_xwz_index = torch.asarray([0, 3, 2], device=device)
g_xww_index = torch.asarray([0, 3, 3], device=device)
g_yxx_index = torch.asarray([1, 0, 0], device=device)
g_yxy_index = torch.asarray([1, 0, 1], device=device)
g_yxz_index = torch.asarray([1, 0, 2], device=device)
g_yxw_index = torch.asarray([1, 0, 3], device=device)
g_yyx_index = torch.asarray([1, 1, 0], device=device)
g_yyy_index = torch.asarray([1, 1, 1], device=device)
g_yyz_index = torch.asarray([1, 1, 2], device=device)
g_yyw_index = torch.asarray([1, 1, 3], device=device)
g_yzx_index = torch.asarray([1, 2, 0], device=device)
g_yzy_index = torch.asarray([1, 2, 1], device=device)
g_yzz_index = torch.asarray([1, 2, 2], device=device)
g_yzw_index = torch.asarray([1, 2, 3], device=device)
g_ywx_index = torch.asarray([1, 3, 0], device=device)
g_ywy_index = torch.asarray([1, 3, 1], device=device)
g_ywz_index = torch.asarray([1, 3, 2], device=device)
g_yww_index = torch.asarray([1, 3, 3], device=device)
g_zxx_index = torch.asarray([2, 0, 0], device=device)
g_zxy_index = torch.asarray([2, 0, 1], device=device)
g_zxz_index = torch.asarray([2, 0, 2], device=device)
g_zxw_index = torch.asarray([2, 0, 3], device=device)
g_zyx_index = torch.asarray([2, 1, 0], device=device)
g_zyy_index = torch.asarray([2, 1, 1], device=device)
g_zyz_index = torch.asarray([2, 1, 2], device=device)
g_zyw_index = torch.asarray([2, 1, 3], device=device)
g_zzx_index = torch.asarray([2, 2, 0], device=device)
g_zzy_index = torch.asarray([2, 2, 1], device=device)
g_zzz_index = torch.asarray([2, 2, 2], device=device)
g_zzw_index = torch.asarray([2, 2, 3], device=device)
g_zwx_index = torch.asarray([2, 3, 0], device=device)
g_zwy_index = torch.asarray([2, 3, 1], device=device)
g_zwz_index = torch.asarray([2, 3, 2], device=device)
g_zww_index = torch.asarray([2, 3, 3], device=device)
g_wxx_index = torch.asarray([3, 0, 0], device=device)
g_wxy_index = torch.asarray([3, 0, 1], device=device)
g_wxz_index = torch.asarray([3, 0, 2], device=device)
g_wxw_index = torch.asarray([3, 0, 3], device=device)
g_wyx_index = torch.asarray([3, 1, 0], device=device)
g_wyy_index = torch.asarray([3, 1, 1], device=device)
g_wyz_index = torch.asarray([3, 1, 2], device=device)
g_wyw_index = torch.asarray([3, 1, 3], device=device)
g_wzx_index = torch.asarray([3, 2, 0], device=device)
g_wzy_index = torch.asarray([3, 2, 1], device=device)
g_wzz_index = torch.asarray([3, 2, 2], device=device)
g_wzw_index = torch.asarray([3, 2, 3], device=device)
g_wwx_index = torch.asarray([3, 3, 0], device=device)
g_wwy_index = torch.asarray([3, 3, 1], device=device)
g_wwz_index = torch.asarray([3, 3, 2], device=device)
g_www_index = torch.asarray([3, 3, 3], device=device)
g_xxxx_index = torch.asarray([0, 0, 0, 0], device=device)
g_xxxy_index = torch.asarray([0, 0, 0, 1], device=device)
g_xxxz_index = torch.asarray([0, 0, 0, 2], device=device)
g_xxxw_index = torch.asarray([0, 0, 0, 3], device=device)
g_xxyx_index = torch.asarray([0, 0, 1, 0], device=device)
g_xxyy_index = torch.asarray([0, 0, 1, 1], device=device)
g_xxyz_index = torch.asarray([0, 0, 1, 2], device=device)
g_xxyw_index = torch.asarray([0, 0, 1, 3], device=device)
g_xxzx_index = torch.asarray([0, 0, 2, 0], device=device)
g_xxzy_index = torch.asarray([0, 0, 2, 1], device=device)
g_xxzz_index = torch.asarray([0, 0, 2, 2], device=device)
g_xxzw_index = torch.asarray([0, 0, 2, 3], device=device)
g_xxwx_index = torch.asarray([0, 0, 3, 0], device=device)
g_xxwy_index = torch.asarray([0, 0, 3, 1], device=device)
g_xxwz_index = torch.asarray([0, 0, 3, 2], device=device)
g_xxww_index = torch.asarray([0, 0, 3, 3], device=device)
g_xyxx_index = torch.asarray([0, 1, 0, 0], device=device)
g_xyxy_index = torch.asarray([0, 1, 0, 1], device=device)
g_xyxz_index = torch.asarray([0, 1, 0, 2], device=device)
g_xyxw_index = torch.asarray([0, 1, 0, 3], device=device)
g_xyyx_index = torch.asarray([0, 1, 1, 0], device=device)
g_xyyy_index = torch.asarray([0, 1, 1, 1], device=device)
g_xyyz_index = torch.asarray([0, 1, 1, 2], device=device)
g_xyyw_index = torch.asarray([0, 1, 1, 3], device=device)
g_xyzx_index = torch.asarray([0, 1, 2, 0], device=device)
g_xyzy_index = torch.asarray([0, 1, 2, 1], device=device)
g_xyzz_index = torch.asarray([0, 1, 2, 2], device=device)
g_xyzw_index = torch.asarray([0, 1, 2, 3], device=device)
g_xywx_index = torch.asarray([0, 1, 3, 0], device=device)
g_xywy_index = torch.asarray([0, 1, 3, 1], device=device)
g_xywz_index = torch.asarray([0, 1, 3, 2], device=device)
g_xyww_index = torch.asarray([0, 1, 3, 3], device=device)
g_xzxx_index = torch.asarray([0, 2, 0, 0], device=device)
g_xzxy_index = torch.asarray([0, 2, 0, 1], device=device)
g_xzxz_index = torch.asarray([0, 2, 0, 2], device=device)
g_xzxw_index = torch.asarray([0, 2, 0, 3], device=device)
g_xzyx_index = torch.asarray([0, 2, 1, 0], device=device)
g_xzyy_index = torch.asarray([0, 2, 1, 1], device=device)
g_xzyz_index = torch.asarray([0, 2, 1, 2], device=device)
g_xzyw_index = torch.asarray([0, 2, 1, 3], device=device)
g_xzzx_index = torch.asarray([0, 2, 2, 0], device=device)
g_xzzy_index = torch.asarray([0, 2, 2, 1], device=device)
g_xzzz_index = torch.asarray([0, 2, 2, 2], device=device)
g_xzzw_index = torch.asarray([0, 2, 2, 3], device=device)
g_xzwx_index = torch.asarray([0, 2, 3, 0], device=device)
g_xzwy_index = torch.asarray([0, 2, 3, 1], device=device)
g_xzwz_index = torch.asarray([0, 2, 3, 2], device=device)
g_xzww_index = torch.asarray([0, 2, 3, 3], device=device)
g_xwxx_index = torch.asarray([0, 3, 0, 0], device=device)
g_xwxy_index = torch.asarray([0, 3, 0, 1], device=device)
g_xwxz_index = torch.asarray([0, 3, 0, 2], device=device)
g_xwxw_index = torch.asarray([0, 3, 0, 3], device=device)
g_xwyx_index = torch.asarray([0, 3, 1, 0], device=device)
g_xwyy_index = torch.asarray([0, 3, 1, 1], device=device)
g_xwyz_index = torch.asarray([0, 3, 1, 2], device=device)
g_xwyw_index = torch.asarray([0, 3, 1, 3], device=device)
g_xwzx_index = torch.asarray([0, 3, 2, 0], device=device)
g_xwzy_index = torch.asarray([0, 3, 2, 1], device=device)
g_xwzz_index = torch.asarray([0, 3, 2, 2], device=device)
g_xwzw_index = torch.asarray([0, 3, 2, 3], device=device)
g_xwwx_index = torch.asarray([0, 3, 3, 0], device=device)
g_xwwy_index = torch.asarray([0, 3, 3, 1], device=device)
g_xwwz_index = torch.asarray([0, 3, 3, 2], device=device)
g_xwww_index = torch.asarray([0, 3, 3, 3], device=device)
g_yxxx_index = torch.asarray([1, 0, 0, 0], device=device)
g_yxxy_index = torch.asarray([1, 0, 0, 1], device=device)
g_yxxz_index = torch.asarray([1, 0, 0, 2], device=device)
g_yxxw_index = torch.asarray([1, 0, 0, 3], device=device)
g_yxyx_index = torch.asarray([1, 0, 1, 0], device=device)
g_yxyy_index = torch.asarray([1, 0, 1, 1], device=device)
g_yxyz_index = torch.asarray([1, 0, 1, 2], device=device)
g_yxyw_index = torch.asarray([1, 0, 1, 3], device=device)
g_yxzx_index = torch.asarray([1, 0, 2, 0], device=device)
g_yxzy_index = torch.asarray([1, 0, 2, 1], device=device)
g_yxzz_index = torch.asarray([1, 0, 2, 2], device=device)
g_yxzw_index = torch.asarray([1, 0, 2, 3], device=device)
g_yxwx_index = torch.asarray([1, 0, 3, 0], device=device)
g_yxwy_index = torch.asarray([1, 0, 3, 1], device=device)
g_yxwz_index = torch.asarray([1, 0, 3, 2], device=device)
g_yxww_index = torch.asarray([1, 0, 3, 3], device=device)
g_yyxx_index = torch.asarray([1, 1, 0, 0], device=device)
g_yyxy_index = torch.asarray([1, 1, 0, 1], device=device)
g_yyxz_index = torch.asarray([1, 1, 0, 2], device=device)
g_yyxw_index = torch.asarray([1, 1, 0, 3], device=device)
g_yyyx_index = torch.asarray([1, 1, 1, 0], device=device)
g_yyyy_index = torch.asarray([1, 1, 1, 1], device=device)
g_yyyz_index = torch.asarray([1, 1, 1, 2], device=device)
g_yyyw_index = torch.asarray([1, 1, 1, 3], device=device)
g_yyzx_index = torch.asarray([1, 1, 2, 0], device=device)
g_yyzy_index = torch.asarray([1, 1, 2, 1], device=device)
g_yyzz_index = torch.asarray([1, 1, 2, 2], device=device)
g_yyzw_index = torch.asarray([1, 1, 2, 3], device=device)
g_yywx_index = torch.asarray([1, 1, 3, 0], device=device)
g_yywy_index = torch.asarray([1, 1, 3, 1], device=device)
g_yywz_index = torch.asarray([1, 1, 3, 2], device=device)
g_yyww_index = torch.asarray([1, 1, 3, 3], device=device)
g_yzxx_index = torch.asarray([1, 2, 0, 0], device=device)
g_yzxy_index = torch.asarray([1, 2, 0, 1], device=device)
g_yzxz_index = torch.asarray([1, 2, 0, 2], device=device)
g_yzxw_index = torch.asarray([1, 2, 0, 3], device=device)
g_yzyx_index = torch.asarray([1, 2, 1, 0], device=device)
g_yzyy_index = torch.asarray([1, 2, 1, 1], device=device)
g_yzyz_index = torch.asarray([1, 2, 1, 2], device=device)
g_yzyw_index = torch.asarray([1, 2, 1, 3], device=device)
g_yzzx_index = torch.asarray([1, 2, 2, 0], device=device)
g_yzzy_index = torch.asarray([1, 2, 2, 1], device=device)
g_yzzz_index = torch.asarray([1, 2, 2, 2], device=device)
g_yzzw_index = torch.asarray([1, 2, 2, 3], device=device)
g_yzwx_index = torch.asarray([1, 2, 3, 0], device=device)
g_yzwy_index = torch.asarray([1, 2, 3, 1], device=device)
g_yzwz_index = torch.asarray([1, 2, 3, 2], device=device)
g_yzww_index = torch.asarray([1, 2, 3, 3], device=device)
g_ywxx_index = torch.asarray([1, 3, 0, 0], device=device)
g_ywxy_index = torch.asarray([1, 3, 0, 1], device=device)
g_ywxz_index = torch.asarray([1, 3, 0, 2], device=device)
g_ywxw_index = torch.asarray([1, 3, 0, 3], device=device)
g_ywyx_index = torch.asarray([1, 3, 1, 0], device=device)
g_ywyy_index = torch.asarray([1, 3, 1, 1], device=device)
g_ywyz_index = torch.asarray([1, 3, 1, 2], device=device)
g_ywyw_index = torch.asarray([1, 3, 1, 3], device=device)
g_ywzx_index = torch.asarray([1, 3, 2, 0], device=device)
g_ywzy_index = torch.asarray([1, 3, 2, 1], device=device)
g_ywzz_index = torch.asarray([1, 3, 2, 2], device=device)
g_ywzw_index = torch.asarray([1, 3, 2, 3], device=device)
g_ywwx_index = torch.asarray([1, 3, 3, 0], device=device)
g_ywwy_index = torch.asarray([1, 3, 3, 1], device=device)
g_ywwz_index = torch.asarray([1, 3, 3, 2], device=device)
g_ywww_index = torch.asarray([1, 3, 3, 3], device=device)
g_zxxx_index = torch.asarray([2, 0, 0, 0], device=device)
g_zxxy_index = torch.asarray([2, 0, 0, 1], device=device)
g_zxxz_index = torch.asarray([2, 0, 0, 2], device=device)
g_zxxw_index = torch.asarray([2, 0, 0, 3], device=device)
g_zxyx_index = torch.asarray([2, 0, 1, 0], device=device)
g_zxyy_index = torch.asarray([2, 0, 1, 1], device=device)
g_zxyz_index = torch.asarray([2, 0, 1, 2], device=device)
g_zxyw_index = torch.asarray([2, 0, 1, 3], device=device)
g_zxzx_index = torch.asarray([2, 0, 2, 0], device=device)
g_zxzy_index = torch.asarray([2, 0, 2, 1], device=device)
g_zxzz_index = torch.asarray([2, 0, 2, 2], device=device)
g_zxzw_index = torch.asarray([2, 0, 2, 3], device=device)
g_zxwx_index = torch.asarray([2, 0, 3, 0], device=device)
g_zxwy_index = torch.asarray([2, 0, 3, 1], device=device)
g_zxwz_index = torch.asarray([2, 0, 3, 2], device=device)
g_zxww_index = torch.asarray([2, 0, 3, 3], device=device)
g_zyxx_index = torch.asarray([2, 1, 0, 0], device=device)
g_zyxy_index = torch.asarray([2, 1, 0, 1], device=device)
g_zyxz_index = torch.asarray([2, 1, 0, 2], device=device)
g_zyxw_index = torch.asarray([2, 1, 0, 3], device=device)
g_zyyx_index = torch.asarray([2, 1, 1, 0], device=device)
g_zyyy_index = torch.asarray([2, 1, 1, 1], device=device)
g_zyyz_index = torch.asarray([2, 1, 1, 2], device=device)
g_zyyw_index = torch.asarray([2, 1, 1, 3], device=device)
g_zyzx_index = torch.asarray([2, 1, 2, 0], device=device)
g_zyzy_index = torch.asarray([2, 1, 2, 1], device=device)
g_zyzz_index = torch.asarray([2, 1, 2, 2], device=device)
g_zyzw_index = torch.asarray([2, 1, 2, 3], device=device)
g_zywx_index = torch.asarray([2, 1, 3, 0], device=device)
g_zywy_index = torch.asarray([2, 1, 3, 1], device=device)
g_zywz_index = torch.asarray([2, 1, 3, 2], device=device)
g_zyww_index = torch.asarray([2, 1, 3, 3], device=device)
g_zzxx_index = torch.asarray([2, 2, 0, 0], device=device)
g_zzxy_index = torch.asarray([2, 2, 0, 1], device=device)
g_zzxz_index = torch.asarray([2, 2, 0, 2], device=device)
g_zzxw_index = torch.asarray([2, 2, 0, 3], device=device)
g_zzyx_index = torch.asarray([2, 2, 1, 0], device=device)
g_zzyy_index = torch.asarray([2, 2, 1, 1], device=device)
g_zzyz_index = torch.asarray([2, 2, 1, 2], device=device)
g_zzyw_index = torch.asarray([2, 2, 1, 3], device=device)
g_zzzx_index = torch.asarray([2, 2, 2, 0], device=device)
g_zzzy_index = torch.asarray([2, 2, 2, 1], device=device)
g_zzzz_index = torch.asarray([2, 2, 2, 2], device=device)
g_zzzw_index = torch.asarray([2, 2, 2, 3], device=device)
g_zzwx_index = torch.asarray([2, 2, 3, 0], device=device)
g_zzwy_index = torch.asarray([2, 2, 3, 1], device=device)
g_zzwz_index = torch.asarray([2, 2, 3, 2], device=device)
g_zzww_index = torch.asarray([2, 2, 3, 3], device=device)
g_zwxx_index = torch.asarray([2, 3, 0, 0], device=device)
g_zwxy_index = torch.asarray([2, 3, 0, 1], device=device)
g_zwxz_index = torch.asarray([2, 3, 0, 2], device=device)
g_zwxw_index = torch.asarray([2, 3, 0, 3], device=device)
g_zwyx_index = torch.asarray([2, 3, 1, 0], device=device)
g_zwyy_index = torch.asarray([2, 3, 1, 1], device=device)
g_zwyz_index = torch.asarray([2, 3, 1, 2], device=device)
g_zwyw_index = torch.asarray([2, 3, 1, 3], device=device)
g_zwzx_index = torch.asarray([2, 3, 2, 0], device=device)
g_zwzy_index = torch.asarray([2, 3, 2, 1], device=device)
g_zwzz_index = torch.asarray([2, 3, 2, 2], device=device)
g_zwzw_index = torch.asarray([2, 3, 2, 3], device=device)
g_zwwx_index = torch.asarray([2, 3, 3, 0], device=device)
g_zwwy_index = torch.asarray([2, 3, 3, 1], device=device)
g_zwwz_index = torch.asarray([2, 3, 3, 2], device=device)
g_zwww_index = torch.asarray([2, 3, 3, 3], device=device)
g_wxxx_index = torch.asarray([3, 0, 0, 0], device=device)
g_wxxy_index = torch.asarray([3, 0, 0, 1], device=device)
g_wxxz_index = torch.asarray([3, 0, 0, 2], device=device)
g_wxxw_index = torch.asarray([3, 0, 0, 3], device=device)
g_wxyx_index = torch.asarray([3, 0, 1, 0], device=device)
g_wxyy_index = torch.asarray([3, 0, 1, 1], device=device)
g_wxyz_index = torch.asarray([3, 0, 1, 2], device=device)
g_wxyw_index = torch.asarray([3, 0, 1, 3], device=device)
g_wxzx_index = torch.asarray([3, 0, 2, 0], device=device)
g_wxzy_index = torch.asarray([3, 0, 2, 1], device=device)
g_wxzz_index = torch.asarray([3, 0, 2, 2], device=device)
g_wxzw_index = torch.asarray([3, 0, 2, 3], device=device)
g_wxwx_index = torch.asarray([3, 0, 3, 0], device=device)
g_wxwy_index = torch.asarray([3, 0, 3, 1], device=device)
g_wxwz_index = torch.asarray([3, 0, 3, 2], device=device)
g_wxww_index = torch.asarray([3, 0, 3, 3], device=device)
g_wyxx_index = torch.asarray([3, 1, 0, 0], device=device)
g_wyxy_index = torch.asarray([3, 1, 0, 1], device=device)
g_wyxz_index = torch.asarray([3, 1, 0, 2], device=device)
g_wyxw_index = torch.asarray([3, 1, 0, 3], device=device)
g_wyyx_index = torch.asarray([3, 1, 1, 0], device=device)
g_wyyy_index = torch.asarray([3, 1, 1, 1], device=device)
g_wyyz_index = torch.asarray([3, 1, 1, 2], device=device)
g_wyyw_index = torch.asarray([3, 1, 1, 3], device=device)
g_wyzx_index = torch.asarray([3, 1, 2, 0], device=device)
g_wyzy_index = torch.asarray([3, 1, 2, 1], device=device)
g_wyzz_index = torch.asarray([3, 1, 2, 2], device=device)
g_wyzw_index = torch.asarray([3, 1, 2, 3], device=device)
g_wywx_index = torch.asarray([3, 1, 3, 0], device=device)
g_wywy_index = torch.asarray([3, 1, 3, 1], device=device)
g_wywz_index = torch.asarray([3, 1, 3, 2], device=device)
g_wyww_index = torch.asarray([3, 1, 3, 3], device=device)
g_wzxx_index = torch.asarray([3, 2, 0, 0], device=device)
g_wzxy_index = torch.asarray([3, 2, 0, 1], device=device)
g_wzxz_index = torch.asarray([3, 2, 0, 2], device=device)
g_wzxw_index = torch.asarray([3, 2, 0, 3], device=device)
g_wzyx_index = torch.asarray([3, 2, 1, 0], device=device)
g_wzyy_index = torch.asarray([3, 2, 1, 1], device=device)
g_wzyz_index = torch.asarray([3, 2, 1, 2], device=device)
g_wzyw_index = torch.asarray([3, 2, 1, 3], device=device)
g_wzzx_index = torch.asarray([3, 2, 2, 0], device=device)
g_wzzy_index = torch.asarray([3, 2, 2, 1], device=device)
g_wzzz_index = torch.asarray([3, 2, 2, 2], device=device)
g_wzzw_index = torch.asarray([3, 2, 2, 3], device=device)
g_wzwx_index = torch.asarray([3, 2, 3, 0], device=device)
g_wzwy_index = torch.asarray([3, 2, 3, 1], device=device)
g_wzwz_index = torch.asarray([3, 2, 3, 2], device=device)
g_wzww_index = torch.asarray([3, 2, 3, 3], device=device)
g_wwxx_index = torch.asarray([3, 3, 0, 0], device=device)
g_wwxy_index = torch.asarray([3, 3, 0, 1], device=device)
g_wwxz_index = torch.asarray([3, 3, 0, 2], device=device)
g_wwxw_index = torch.asarray([3, 3, 0, 3], device=device)
g_wwyx_index = torch.asarray([3, 3, 1, 0], device=device)
g_wwyy_index = torch.asarray([3, 3, 1, 1], device=device)
g_wwyz_index = torch.asarray([3, 3, 1, 2], device=device)
g_wwyw_index = torch.asarray([3, 3, 1, 3], device=device)
g_wwzx_index = torch.asarray([3, 3, 2, 0], device=device)
g_wwzy_index = torch.asarray([3, 3, 2, 1], device=device)
g_wwzz_index = torch.asarray([3, 3, 2, 2], device=device)
g_wwzw_index = torch.asarray([3, 3, 2, 3], device=device)
g_wwwx_index = torch.asarray([3, 3, 3, 0], device=device)
g_wwwy_index = torch.asarray([3, 3, 3, 1], device=device)
g_wwwz_index = torch.asarray([3, 3, 3, 2], device=device)
g_wwww_index = torch.asarray([3, 3, 3, 3], device=device)

def swizzle_n_x(v):
    return v
def swizzle_n_xx(v):
    return torch.asarray([v, v], device=device)
def swizzle_n_xxx(v):
    return torch.asarray([v, v, v], device=device)
def swizzle_n_xxxx(v):
    return torch.asarray([v, v, v, v], device=device)
def swizzle_n2_x(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_x_index).squeeze(0)
def swizzle_n2_y(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_y_index).squeeze(0)
def swizzle_n2_xx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xx_index)
def swizzle_n2_xy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xy_index)
def swizzle_n2_yx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yx_index)
def swizzle_n2_yy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yy_index)
def swizzle_n2_xxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxx_index)
def swizzle_n2_xxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxy_index)
def swizzle_n2_xyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyx_index)
def swizzle_n2_xyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyy_index)
def swizzle_n2_yxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxx_index)
def swizzle_n2_yxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxy_index)
def swizzle_n2_yyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyx_index)
def swizzle_n2_yyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyy_index)
def swizzle_n2_xxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxx_index)
def swizzle_n2_xxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxy_index)
def swizzle_n2_xxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyx_index)
def swizzle_n2_xxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyy_index)
def swizzle_n2_xyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxx_index)
def swizzle_n2_xyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxy_index)
def swizzle_n2_xyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyx_index)
def swizzle_n2_xyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyy_index)
def swizzle_n2_yxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxx_index)
def swizzle_n2_yxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxy_index)
def swizzle_n2_yxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyx_index)
def swizzle_n2_yxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyy_index)
def swizzle_n2_yyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxx_index)
def swizzle_n2_yyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxy_index)
def swizzle_n2_yyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyx_index)
def swizzle_n2_yyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyy_index)
def swizzle_n3_x(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_x_index).squeeze(0)
def swizzle_n3_y(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_y_index).squeeze(0)
def swizzle_n3_z(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_z_index).squeeze(0)
def swizzle_n3_xx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xx_index)
def swizzle_n3_xy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xy_index)
def swizzle_n3_xz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xz_index)
def swizzle_n3_yx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yx_index)
def swizzle_n3_yy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yy_index)
def swizzle_n3_yz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yz_index)
def swizzle_n3_zx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zx_index)
def swizzle_n3_zy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zy_index)
def swizzle_n3_zz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zz_index)
def swizzle_n3_xxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxx_index)
def swizzle_n3_xxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxy_index)
def swizzle_n3_xxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxz_index)
def swizzle_n3_xyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyx_index)
def swizzle_n3_xyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyy_index)
def swizzle_n3_xyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyz_index)
def swizzle_n3_xzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzx_index)
def swizzle_n3_xzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzy_index)
def swizzle_n3_xzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzz_index)
def swizzle_n3_yxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxx_index)
def swizzle_n3_yxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxy_index)
def swizzle_n3_yxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxz_index)
def swizzle_n3_yyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyx_index)
def swizzle_n3_yyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyy_index)
def swizzle_n3_yyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyz_index)
def swizzle_n3_yzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzx_index)
def swizzle_n3_yzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzy_index)
def swizzle_n3_yzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzz_index)
def swizzle_n3_zxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxx_index)
def swizzle_n3_zxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxy_index)
def swizzle_n3_zxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxz_index)
def swizzle_n3_zyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyx_index)
def swizzle_n3_zyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyy_index)
def swizzle_n3_zyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyz_index)
def swizzle_n3_zzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzx_index)
def swizzle_n3_zzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzy_index)
def swizzle_n3_zzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzz_index)
def swizzle_n3_xxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxx_index)
def swizzle_n3_xxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxy_index)
def swizzle_n3_xxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxz_index)
def swizzle_n3_xxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyx_index)
def swizzle_n3_xxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyy_index)
def swizzle_n3_xxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyz_index)
def swizzle_n3_xxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzx_index)
def swizzle_n3_xxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzy_index)
def swizzle_n3_xxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzz_index)
def swizzle_n3_xyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxx_index)
def swizzle_n3_xyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxy_index)
def swizzle_n3_xyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxz_index)
def swizzle_n3_xyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyx_index)
def swizzle_n3_xyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyy_index)
def swizzle_n3_xyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyz_index)
def swizzle_n3_xyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzx_index)
def swizzle_n3_xyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzy_index)
def swizzle_n3_xyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzz_index)
def swizzle_n3_xzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxx_index)
def swizzle_n3_xzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxy_index)
def swizzle_n3_xzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxz_index)
def swizzle_n3_xzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyx_index)
def swizzle_n3_xzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyy_index)
def swizzle_n3_xzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyz_index)
def swizzle_n3_xzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzx_index)
def swizzle_n3_xzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzy_index)
def swizzle_n3_xzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzz_index)
def swizzle_n3_yxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxx_index)
def swizzle_n3_yxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxy_index)
def swizzle_n3_yxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxz_index)
def swizzle_n3_yxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyx_index)
def swizzle_n3_yxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyy_index)
def swizzle_n3_yxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyz_index)
def swizzle_n3_yxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzx_index)
def swizzle_n3_yxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzy_index)
def swizzle_n3_yxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzz_index)
def swizzle_n3_yyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxx_index)
def swizzle_n3_yyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxy_index)
def swizzle_n3_yyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxz_index)
def swizzle_n3_yyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyx_index)
def swizzle_n3_yyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyy_index)
def swizzle_n3_yyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyz_index)
def swizzle_n3_yyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzx_index)
def swizzle_n3_yyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzy_index)
def swizzle_n3_yyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzz_index)
def swizzle_n3_yzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxx_index)
def swizzle_n3_yzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxy_index)
def swizzle_n3_yzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxz_index)
def swizzle_n3_yzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyx_index)
def swizzle_n3_yzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyy_index)
def swizzle_n3_yzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyz_index)
def swizzle_n3_yzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzx_index)
def swizzle_n3_yzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzy_index)
def swizzle_n3_yzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzz_index)
def swizzle_n3_zxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxx_index)
def swizzle_n3_zxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxy_index)
def swizzle_n3_zxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxz_index)
def swizzle_n3_zxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyx_index)
def swizzle_n3_zxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyy_index)
def swizzle_n3_zxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyz_index)
def swizzle_n3_zxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzx_index)
def swizzle_n3_zxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzy_index)
def swizzle_n3_zxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzz_index)
def swizzle_n3_zyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxx_index)
def swizzle_n3_zyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxy_index)
def swizzle_n3_zyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxz_index)
def swizzle_n3_zyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyx_index)
def swizzle_n3_zyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyy_index)
def swizzle_n3_zyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyz_index)
def swizzle_n3_zyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzx_index)
def swizzle_n3_zyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzy_index)
def swizzle_n3_zyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzz_index)
def swizzle_n3_zzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxx_index)
def swizzle_n3_zzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxy_index)
def swizzle_n3_zzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxz_index)
def swizzle_n3_zzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyx_index)
def swizzle_n3_zzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyy_index)
def swizzle_n3_zzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyz_index)
def swizzle_n3_zzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzx_index)
def swizzle_n3_zzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzy_index)
def swizzle_n3_zzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzz_index)
def swizzle_n4_x(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_x_index).squeeze(0)
def swizzle_n4_y(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_y_index).squeeze(0)
def swizzle_n4_z(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_z_index).squeeze(0)
def swizzle_n4_w(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_w_index).squeeze(0)
def swizzle_n4_xx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xx_index)
def swizzle_n4_xy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xy_index)
def swizzle_n4_xz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xz_index)
def swizzle_n4_xw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xw_index)
def swizzle_n4_yx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yx_index)
def swizzle_n4_yy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yy_index)
def swizzle_n4_yz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yz_index)
def swizzle_n4_yw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yw_index)
def swizzle_n4_zx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zx_index)
def swizzle_n4_zy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zy_index)
def swizzle_n4_zz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zz_index)
def swizzle_n4_zw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zw_index)
def swizzle_n4_wx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wx_index)
def swizzle_n4_wy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wy_index)
def swizzle_n4_wz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wz_index)
def swizzle_n4_ww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ww_index)
def swizzle_n4_xxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxx_index)
def swizzle_n4_xxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxy_index)
def swizzle_n4_xxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxz_index)
def swizzle_n4_xxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxw_index)
def swizzle_n4_xyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyx_index)
def swizzle_n4_xyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyy_index)
def swizzle_n4_xyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyz_index)
def swizzle_n4_xyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyw_index)
def swizzle_n4_xzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzx_index)
def swizzle_n4_xzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzy_index)
def swizzle_n4_xzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzz_index)
def swizzle_n4_xzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzw_index)
def swizzle_n4_xwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwx_index)
def swizzle_n4_xwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwy_index)
def swizzle_n4_xwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwz_index)
def swizzle_n4_xww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xww_index)
def swizzle_n4_yxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxx_index)
def swizzle_n4_yxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxy_index)
def swizzle_n4_yxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxz_index)
def swizzle_n4_yxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxw_index)
def swizzle_n4_yyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyx_index)
def swizzle_n4_yyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyy_index)
def swizzle_n4_yyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyz_index)
def swizzle_n4_yyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyw_index)
def swizzle_n4_yzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzx_index)
def swizzle_n4_yzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzy_index)
def swizzle_n4_yzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzz_index)
def swizzle_n4_yzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzw_index)
def swizzle_n4_ywx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywx_index)
def swizzle_n4_ywy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywy_index)
def swizzle_n4_ywz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywz_index)
def swizzle_n4_yww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yww_index)
def swizzle_n4_zxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxx_index)
def swizzle_n4_zxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxy_index)
def swizzle_n4_zxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxz_index)
def swizzle_n4_zxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxw_index)
def swizzle_n4_zyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyx_index)
def swizzle_n4_zyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyy_index)
def swizzle_n4_zyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyz_index)
def swizzle_n4_zyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyw_index)
def swizzle_n4_zzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzx_index)
def swizzle_n4_zzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzy_index)
def swizzle_n4_zzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzz_index)
def swizzle_n4_zzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzw_index)
def swizzle_n4_zwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwx_index)
def swizzle_n4_zwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwy_index)
def swizzle_n4_zwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwz_index)
def swizzle_n4_zww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zww_index)
def swizzle_n4_wxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxx_index)
def swizzle_n4_wxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxy_index)
def swizzle_n4_wxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxz_index)
def swizzle_n4_wxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxw_index)
def swizzle_n4_wyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyx_index)
def swizzle_n4_wyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyy_index)
def swizzle_n4_wyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyz_index)
def swizzle_n4_wyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyw_index)
def swizzle_n4_wzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzx_index)
def swizzle_n4_wzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzy_index)
def swizzle_n4_wzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzz_index)
def swizzle_n4_wzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzw_index)
def swizzle_n4_wwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwx_index)
def swizzle_n4_wwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwy_index)
def swizzle_n4_wwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwz_index)
def swizzle_n4_www(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_www_index)
def swizzle_n4_xxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxx_index)
def swizzle_n4_xxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxy_index)
def swizzle_n4_xxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxz_index)
def swizzle_n4_xxxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxxw_index)
def swizzle_n4_xxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyx_index)
def swizzle_n4_xxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyy_index)
def swizzle_n4_xxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyz_index)
def swizzle_n4_xxyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxyw_index)
def swizzle_n4_xxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzx_index)
def swizzle_n4_xxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzy_index)
def swizzle_n4_xxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzz_index)
def swizzle_n4_xxzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxzw_index)
def swizzle_n4_xxwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxwx_index)
def swizzle_n4_xxwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxwy_index)
def swizzle_n4_xxwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxwz_index)
def swizzle_n4_xxww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xxww_index)
def swizzle_n4_xyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxx_index)
def swizzle_n4_xyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxy_index)
def swizzle_n4_xyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxz_index)
def swizzle_n4_xyxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyxw_index)
def swizzle_n4_xyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyx_index)
def swizzle_n4_xyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyy_index)
def swizzle_n4_xyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyz_index)
def swizzle_n4_xyyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyyw_index)
def swizzle_n4_xyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzx_index)
def swizzle_n4_xyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzy_index)
def swizzle_n4_xyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzz_index)
def swizzle_n4_xyzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyzw_index)
def swizzle_n4_xywx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xywx_index)
def swizzle_n4_xywy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xywy_index)
def swizzle_n4_xywz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xywz_index)
def swizzle_n4_xyww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xyww_index)
def swizzle_n4_xzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxx_index)
def swizzle_n4_xzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxy_index)
def swizzle_n4_xzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxz_index)
def swizzle_n4_xzxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzxw_index)
def swizzle_n4_xzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyx_index)
def swizzle_n4_xzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyy_index)
def swizzle_n4_xzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyz_index)
def swizzle_n4_xzyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzyw_index)
def swizzle_n4_xzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzx_index)
def swizzle_n4_xzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzy_index)
def swizzle_n4_xzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzz_index)
def swizzle_n4_xzzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzzw_index)
def swizzle_n4_xzwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzwx_index)
def swizzle_n4_xzwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzwy_index)
def swizzle_n4_xzwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzwz_index)
def swizzle_n4_xzww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xzww_index)
def swizzle_n4_xwxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwxx_index)
def swizzle_n4_xwxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwxy_index)
def swizzle_n4_xwxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwxz_index)
def swizzle_n4_xwxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwxw_index)
def swizzle_n4_xwyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwyx_index)
def swizzle_n4_xwyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwyy_index)
def swizzle_n4_xwyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwyz_index)
def swizzle_n4_xwyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwyw_index)
def swizzle_n4_xwzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwzx_index)
def swizzle_n4_xwzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwzy_index)
def swizzle_n4_xwzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwzz_index)
def swizzle_n4_xwzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwzw_index)
def swizzle_n4_xwwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwwx_index)
def swizzle_n4_xwwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwwy_index)
def swizzle_n4_xwwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwwz_index)
def swizzle_n4_xwww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_xwww_index)
def swizzle_n4_yxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxx_index)
def swizzle_n4_yxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxy_index)
def swizzle_n4_yxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxz_index)
def swizzle_n4_yxxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxxw_index)
def swizzle_n4_yxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyx_index)
def swizzle_n4_yxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyy_index)
def swizzle_n4_yxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyz_index)
def swizzle_n4_yxyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxyw_index)
def swizzle_n4_yxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzx_index)
def swizzle_n4_yxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzy_index)
def swizzle_n4_yxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzz_index)
def swizzle_n4_yxzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxzw_index)
def swizzle_n4_yxwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxwx_index)
def swizzle_n4_yxwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxwy_index)
def swizzle_n4_yxwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxwz_index)
def swizzle_n4_yxww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yxww_index)
def swizzle_n4_yyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxx_index)
def swizzle_n4_yyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxy_index)
def swizzle_n4_yyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxz_index)
def swizzle_n4_yyxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyxw_index)
def swizzle_n4_yyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyx_index)
def swizzle_n4_yyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyy_index)
def swizzle_n4_yyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyz_index)
def swizzle_n4_yyyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyyw_index)
def swizzle_n4_yyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzx_index)
def swizzle_n4_yyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzy_index)
def swizzle_n4_yyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzz_index)
def swizzle_n4_yyzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyzw_index)
def swizzle_n4_yywx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yywx_index)
def swizzle_n4_yywy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yywy_index)
def swizzle_n4_yywz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yywz_index)
def swizzle_n4_yyww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yyww_index)
def swizzle_n4_yzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxx_index)
def swizzle_n4_yzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxy_index)
def swizzle_n4_yzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxz_index)
def swizzle_n4_yzxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzxw_index)
def swizzle_n4_yzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyx_index)
def swizzle_n4_yzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyy_index)
def swizzle_n4_yzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyz_index)
def swizzle_n4_yzyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzyw_index)
def swizzle_n4_yzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzx_index)
def swizzle_n4_yzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzy_index)
def swizzle_n4_yzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzz_index)
def swizzle_n4_yzzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzzw_index)
def swizzle_n4_yzwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzwx_index)
def swizzle_n4_yzwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzwy_index)
def swizzle_n4_yzwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzwz_index)
def swizzle_n4_yzww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_yzww_index)
def swizzle_n4_ywxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywxx_index)
def swizzle_n4_ywxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywxy_index)
def swizzle_n4_ywxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywxz_index)
def swizzle_n4_ywxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywxw_index)
def swizzle_n4_ywyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywyx_index)
def swizzle_n4_ywyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywyy_index)
def swizzle_n4_ywyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywyz_index)
def swizzle_n4_ywyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywyw_index)
def swizzle_n4_ywzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywzx_index)
def swizzle_n4_ywzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywzy_index)
def swizzle_n4_ywzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywzz_index)
def swizzle_n4_ywzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywzw_index)
def swizzle_n4_ywwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywwx_index)
def swizzle_n4_ywwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywwy_index)
def swizzle_n4_ywwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywwz_index)
def swizzle_n4_ywww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_ywww_index)
def swizzle_n4_zxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxx_index)
def swizzle_n4_zxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxy_index)
def swizzle_n4_zxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxz_index)
def swizzle_n4_zxxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxxw_index)
def swizzle_n4_zxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyx_index)
def swizzle_n4_zxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyy_index)
def swizzle_n4_zxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyz_index)
def swizzle_n4_zxyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxyw_index)
def swizzle_n4_zxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzx_index)
def swizzle_n4_zxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzy_index)
def swizzle_n4_zxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzz_index)
def swizzle_n4_zxzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxzw_index)
def swizzle_n4_zxwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxwx_index)
def swizzle_n4_zxwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxwy_index)
def swizzle_n4_zxwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxwz_index)
def swizzle_n4_zxww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zxww_index)
def swizzle_n4_zyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxx_index)
def swizzle_n4_zyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxy_index)
def swizzle_n4_zyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxz_index)
def swizzle_n4_zyxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyxw_index)
def swizzle_n4_zyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyx_index)
def swizzle_n4_zyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyy_index)
def swizzle_n4_zyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyz_index)
def swizzle_n4_zyyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyyw_index)
def swizzle_n4_zyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzx_index)
def swizzle_n4_zyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzy_index)
def swizzle_n4_zyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzz_index)
def swizzle_n4_zyzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyzw_index)
def swizzle_n4_zywx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zywx_index)
def swizzle_n4_zywy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zywy_index)
def swizzle_n4_zywz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zywz_index)
def swizzle_n4_zyww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zyww_index)
def swizzle_n4_zzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxx_index)
def swizzle_n4_zzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxy_index)
def swizzle_n4_zzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxz_index)
def swizzle_n4_zzxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzxw_index)
def swizzle_n4_zzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyx_index)
def swizzle_n4_zzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyy_index)
def swizzle_n4_zzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyz_index)
def swizzle_n4_zzyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzyw_index)
def swizzle_n4_zzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzx_index)
def swizzle_n4_zzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzy_index)
def swizzle_n4_zzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzz_index)
def swizzle_n4_zzzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzzw_index)
def swizzle_n4_zzwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzwx_index)
def swizzle_n4_zzwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzwy_index)
def swizzle_n4_zzwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzwz_index)
def swizzle_n4_zzww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zzww_index)
def swizzle_n4_zwxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwxx_index)
def swizzle_n4_zwxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwxy_index)
def swizzle_n4_zwxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwxz_index)
def swizzle_n4_zwxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwxw_index)
def swizzle_n4_zwyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwyx_index)
def swizzle_n4_zwyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwyy_index)
def swizzle_n4_zwyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwyz_index)
def swizzle_n4_zwyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwyw_index)
def swizzle_n4_zwzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwzx_index)
def swizzle_n4_zwzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwzy_index)
def swizzle_n4_zwzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwzz_index)
def swizzle_n4_zwzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwzw_index)
def swizzle_n4_zwwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwwx_index)
def swizzle_n4_zwwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwwy_index)
def swizzle_n4_zwwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwwz_index)
def swizzle_n4_zwww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_zwww_index)
def swizzle_n4_wxxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxxx_index)
def swizzle_n4_wxxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxxy_index)
def swizzle_n4_wxxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxxz_index)
def swizzle_n4_wxxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxxw_index)
def swizzle_n4_wxyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxyx_index)
def swizzle_n4_wxyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxyy_index)
def swizzle_n4_wxyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxyz_index)
def swizzle_n4_wxyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxyw_index)
def swizzle_n4_wxzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxzx_index)
def swizzle_n4_wxzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxzy_index)
def swizzle_n4_wxzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxzz_index)
def swizzle_n4_wxzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxzw_index)
def swizzle_n4_wxwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxwx_index)
def swizzle_n4_wxwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxwy_index)
def swizzle_n4_wxwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxwz_index)
def swizzle_n4_wxww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wxww_index)
def swizzle_n4_wyxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyxx_index)
def swizzle_n4_wyxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyxy_index)
def swizzle_n4_wyxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyxz_index)
def swizzle_n4_wyxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyxw_index)
def swizzle_n4_wyyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyyx_index)
def swizzle_n4_wyyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyyy_index)
def swizzle_n4_wyyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyyz_index)
def swizzle_n4_wyyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyyw_index)
def swizzle_n4_wyzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyzx_index)
def swizzle_n4_wyzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyzy_index)
def swizzle_n4_wyzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyzz_index)
def swizzle_n4_wyzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyzw_index)
def swizzle_n4_wywx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wywx_index)
def swizzle_n4_wywy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wywy_index)
def swizzle_n4_wywz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wywz_index)
def swizzle_n4_wyww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wyww_index)
def swizzle_n4_wzxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzxx_index)
def swizzle_n4_wzxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzxy_index)
def swizzle_n4_wzxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzxz_index)
def swizzle_n4_wzxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzxw_index)
def swizzle_n4_wzyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzyx_index)
def swizzle_n4_wzyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzyy_index)
def swizzle_n4_wzyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzyz_index)
def swizzle_n4_wzyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzyw_index)
def swizzle_n4_wzzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzzx_index)
def swizzle_n4_wzzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzzy_index)
def swizzle_n4_wzzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzzz_index)
def swizzle_n4_wzzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzzw_index)
def swizzle_n4_wzwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzwx_index)
def swizzle_n4_wzwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzwy_index)
def swizzle_n4_wzwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzwz_index)
def swizzle_n4_wzww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wzww_index)
def swizzle_n4_wwxx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwxx_index)
def swizzle_n4_wwxy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwxy_index)
def swizzle_n4_wwxz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwxz_index)
def swizzle_n4_wwxw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwxw_index)
def swizzle_n4_wwyx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwyx_index)
def swizzle_n4_wwyy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwyy_index)
def swizzle_n4_wwyz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwyz_index)
def swizzle_n4_wwyw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwyw_index)
def swizzle_n4_wwzx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwzx_index)
def swizzle_n4_wwzy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwzy_index)
def swizzle_n4_wwzz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwzz_index)
def swizzle_n4_wwzw(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwzw_index)
def swizzle_n4_wwwx(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwwx_index)
def swizzle_n4_wwwy(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwwy_index)
def swizzle_n4_wwwz(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwwz_index)
def swizzle_n4_wwww(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_wwww_index)
def swizzle_set_n_x(v, val):
    raise
def swizzle_set_n2_x(v, val):
    v[0] = val
    return val
def swizzle_set_n2_y(v, val):
    v[1] = val
    return val
def swizzle_set_n2_xy(v, val):
    v[0] = val[0]
    v[1] = val[1]
    return val
def swizzle_set_n2_yx(v, val):
    v[1] = val[0]
    v[0] = val[1]
    return val
def swizzle_set_n3_x(v, val):
    v[0] = val
    return val
def swizzle_set_n3_y(v, val):
    v[1] = val
    return val
def swizzle_set_n3_z(v, val):
    v[2] = val
    return val
def swizzle_set_n3_xy(v, val):
    v[0] = val[0]
    v[1] = val[1]
    return val
def swizzle_set_n3_xz(v, val):
    v[0] = val[0]
    v[2] = val[1]
    return val
def swizzle_set_n3_yx(v, val):
    v[1] = val[0]
    v[0] = val[1]
    return val
def swizzle_set_n3_yz(v, val):
    v[1] = val[0]
    v[2] = val[1]
    return val
def swizzle_set_n3_zx(v, val):
    v[2] = val[0]
    v[0] = val[1]
    return val
def swizzle_set_n3_zy(v, val):
    v[2] = val[0]
    v[1] = val[1]
    return val
def swizzle_set_n3_xyz(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n3_xzy(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n3_yxz(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n3_yzx(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n3_zxy(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n3_zyx(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_x(v, val):
    v[0] = val
    return val
def swizzle_set_n4_y(v, val):
    v[1] = val
    return val
def swizzle_set_n4_z(v, val):
    v[2] = val
    return val
def swizzle_set_n4_w(v, val):
    v[3] = val
    return val
def swizzle_set_n4_xy(v, val):
    v[0] = val[0]
    v[1] = val[1]
    return val
def swizzle_set_n4_xz(v, val):
    v[0] = val[0]
    v[2] = val[1]
    return val
def swizzle_set_n4_xw(v, val):
    v[0] = val[0]
    v[3] = val[1]
    return val
def swizzle_set_n4_yx(v, val):
    v[1] = val[0]
    v[0] = val[1]
    return val
def swizzle_set_n4_yz(v, val):
    v[1] = val[0]
    v[2] = val[1]
    return val
def swizzle_set_n4_yw(v, val):
    v[1] = val[0]
    v[3] = val[1]
    return val
def swizzle_set_n4_zx(v, val):
    v[2] = val[0]
    v[0] = val[1]
    return val
def swizzle_set_n4_zy(v, val):
    v[2] = val[0]
    v[1] = val[1]
    return val
def swizzle_set_n4_zw(v, val):
    v[2] = val[0]
    v[3] = val[1]
    return val
def swizzle_set_n4_wx(v, val):
    v[3] = val[0]
    v[0] = val[1]
    return val
def swizzle_set_n4_wy(v, val):
    v[3] = val[0]
    v[1] = val[1]
    return val
def swizzle_set_n4_wz(v, val):
    v[3] = val[0]
    v[2] = val[1]
    return val
def swizzle_set_n4_xyz(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n4_xyw(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    return val
def swizzle_set_n4_xzy(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n4_xzw(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    return val
def swizzle_set_n4_xwy(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n4_xwz(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n4_yxz(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n4_yxw(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    return val
def swizzle_set_n4_yzx(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_yzw(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    return val
def swizzle_set_n4_ywx(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_ywz(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n4_zxy(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n4_zxw(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    return val
def swizzle_set_n4_zyx(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_zyw(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    return val
def swizzle_set_n4_zwx(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_zwy(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n4_wxy(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n4_wxz(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n4_wyx(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_wyz(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    return val
def swizzle_set_n4_wzx(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    return val
def swizzle_set_n4_wzy(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    return val
def swizzle_set_n4_xyzw(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    v[3] = val[3]
    return val
def swizzle_set_n4_xywz(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    v[2] = val[3]
    return val
def swizzle_set_n4_xzyw(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    v[3] = val[3]
    return val
def swizzle_set_n4_xzwy(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    v[1] = val[3]
    return val
def swizzle_set_n4_xwyz(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    v[2] = val[3]
    return val
def swizzle_set_n4_xwzy(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    v[1] = val[3]
    return val
def swizzle_set_n4_yxzw(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    v[3] = val[3]
    return val
def swizzle_set_n4_yxwz(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    v[2] = val[3]
    return val
def swizzle_set_n4_yzxw(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    v[3] = val[3]
    return val
def swizzle_set_n4_yzwx(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    v[0] = val[3]
    return val
def swizzle_set_n4_ywxz(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    v[2] = val[3]
    return val
def swizzle_set_n4_ywzx(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    v[0] = val[3]
    return val
def swizzle_set_n4_zxyw(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    v[3] = val[3]
    return val
def swizzle_set_n4_zxwy(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    v[1] = val[3]
    return val
def swizzle_set_n4_zyxw(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    v[3] = val[3]
    return val
def swizzle_set_n4_zywx(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    v[0] = val[3]
    return val
def swizzle_set_n4_zwxy(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    v[1] = val[3]
    return val
def swizzle_set_n4_zwyx(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    v[0] = val[3]
    return val
def swizzle_set_n4_wxyz(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    v[2] = val[3]
    return val
def swizzle_set_n4_wxzy(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    v[1] = val[3]
    return val
def swizzle_set_n4_wyxz(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    v[2] = val[3]
    return val
def swizzle_set_n4_wyzx(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    v[0] = val[3]
    return val
def swizzle_set_n4_wzxy(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    v[1] = val[3]
    return val
def swizzle_set_n4_wzyx(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    v[0] = val[3]
    return val
def swizzle_t_n_x(v):
    return v
def swizzle_t_n_xx(v):
    return v.unsqueeze(1).index_select(1, g_xx_index)
def swizzle_t_n_xxx(v):
    return v.unsqueeze(1).index_select(1, g_xxx_index)
def swizzle_t_n_xxxx(v):
    return v.unsqueeze(1).index_select(1, g_xxxx_index)
def swizzle_t_n2_x(v):
    return torch.clone(v[..., 0])
def swizzle_t_n2_y(v):
    return torch.clone(v[..., 1])
def swizzle_t_n2_xx(v):
    return v.index_select(1, g_xx_index)
def swizzle_t_n2_xy(v):
    return v.index_select(1, g_xy_index)
def swizzle_t_n2_yx(v):
    return v.index_select(1, g_yx_index)
def swizzle_t_n2_yy(v):
    return v.index_select(1, g_yy_index)
def swizzle_t_n2_xxx(v):
    return v.index_select(1, g_xxx_index)
def swizzle_t_n2_xxy(v):
    return v.index_select(1, g_xxy_index)
def swizzle_t_n2_xyx(v):
    return v.index_select(1, g_xyx_index)
def swizzle_t_n2_xyy(v):
    return v.index_select(1, g_xyy_index)
def swizzle_t_n2_yxx(v):
    return v.index_select(1, g_yxx_index)
def swizzle_t_n2_yxy(v):
    return v.index_select(1, g_yxy_index)
def swizzle_t_n2_yyx(v):
    return v.index_select(1, g_yyx_index)
def swizzle_t_n2_yyy(v):
    return v.index_select(1, g_yyy_index)
def swizzle_t_n2_xxxx(v):
    return v.index_select(1, g_xxxx_index)
def swizzle_t_n2_xxxy(v):
    return v.index_select(1, g_xxxy_index)
def swizzle_t_n2_xxyx(v):
    return v.index_select(1, g_xxyx_index)
def swizzle_t_n2_xxyy(v):
    return v.index_select(1, g_xxyy_index)
def swizzle_t_n2_xyxx(v):
    return v.index_select(1, g_xyxx_index)
def swizzle_t_n2_xyxy(v):
    return v.index_select(1, g_xyxy_index)
def swizzle_t_n2_xyyx(v):
    return v.index_select(1, g_xyyx_index)
def swizzle_t_n2_xyyy(v):
    return v.index_select(1, g_xyyy_index)
def swizzle_t_n2_yxxx(v):
    return v.index_select(1, g_yxxx_index)
def swizzle_t_n2_yxxy(v):
    return v.index_select(1, g_yxxy_index)
def swizzle_t_n2_yxyx(v):
    return v.index_select(1, g_yxyx_index)
def swizzle_t_n2_yxyy(v):
    return v.index_select(1, g_yxyy_index)
def swizzle_t_n2_yyxx(v):
    return v.index_select(1, g_yyxx_index)
def swizzle_t_n2_yyxy(v):
    return v.index_select(1, g_yyxy_index)
def swizzle_t_n2_yyyx(v):
    return v.index_select(1, g_yyyx_index)
def swizzle_t_n2_yyyy(v):
    return v.index_select(1, g_yyyy_index)
def swizzle_t_n3_x(v):
    return torch.clone(v[..., 0])
def swizzle_t_n3_y(v):
    return torch.clone(v[..., 1])
def swizzle_t_n3_z(v):
    return torch.clone(v[..., 2])
def swizzle_t_n3_xx(v):
    return v.index_select(1, g_xx_index)
def swizzle_t_n3_xy(v):
    return v.index_select(1, g_xy_index)
def swizzle_t_n3_xz(v):
    return v.index_select(1, g_xz_index)
def swizzle_t_n3_yx(v):
    return v.index_select(1, g_yx_index)
def swizzle_t_n3_yy(v):
    return v.index_select(1, g_yy_index)
def swizzle_t_n3_yz(v):
    return v.index_select(1, g_yz_index)
def swizzle_t_n3_zx(v):
    return v.index_select(1, g_zx_index)
def swizzle_t_n3_zy(v):
    return v.index_select(1, g_zy_index)
def swizzle_t_n3_zz(v):
    return v.index_select(1, g_zz_index)
def swizzle_t_n3_xxx(v):
    return v.index_select(1, g_xxx_index)
def swizzle_t_n3_xxy(v):
    return v.index_select(1, g_xxy_index)
def swizzle_t_n3_xxz(v):
    return v.index_select(1, g_xxz_index)
def swizzle_t_n3_xyx(v):
    return v.index_select(1, g_xyx_index)
def swizzle_t_n3_xyy(v):
    return v.index_select(1, g_xyy_index)
def swizzle_t_n3_xyz(v):
    return v.index_select(1, g_xyz_index)
def swizzle_t_n3_xzx(v):
    return v.index_select(1, g_xzx_index)
def swizzle_t_n3_xzy(v):
    return v.index_select(1, g_xzy_index)
def swizzle_t_n3_xzz(v):
    return v.index_select(1, g_xzz_index)
def swizzle_t_n3_yxx(v):
    return v.index_select(1, g_yxx_index)
def swizzle_t_n3_yxy(v):
    return v.index_select(1, g_yxy_index)
def swizzle_t_n3_yxz(v):
    return v.index_select(1, g_yxz_index)
def swizzle_t_n3_yyx(v):
    return v.index_select(1, g_yyx_index)
def swizzle_t_n3_yyy(v):
    return v.index_select(1, g_yyy_index)
def swizzle_t_n3_yyz(v):
    return v.index_select(1, g_yyz_index)
def swizzle_t_n3_yzx(v):
    return v.index_select(1, g_yzx_index)
def swizzle_t_n3_yzy(v):
    return v.index_select(1, g_yzy_index)
def swizzle_t_n3_yzz(v):
    return v.index_select(1, g_yzz_index)
def swizzle_t_n3_zxx(v):
    return v.index_select(1, g_zxx_index)
def swizzle_t_n3_zxy(v):
    return v.index_select(1, g_zxy_index)
def swizzle_t_n3_zxz(v):
    return v.index_select(1, g_zxz_index)
def swizzle_t_n3_zyx(v):
    return v.index_select(1, g_zyx_index)
def swizzle_t_n3_zyy(v):
    return v.index_select(1, g_zyy_index)
def swizzle_t_n3_zyz(v):
    return v.index_select(1, g_zyz_index)
def swizzle_t_n3_zzx(v):
    return v.index_select(1, g_zzx_index)
def swizzle_t_n3_zzy(v):
    return v.index_select(1, g_zzy_index)
def swizzle_t_n3_zzz(v):
    return v.index_select(1, g_zzz_index)
def swizzle_t_n3_xxxx(v):
    return v.index_select(1, g_xxxx_index)
def swizzle_t_n3_xxxy(v):
    return v.index_select(1, g_xxxy_index)
def swizzle_t_n3_xxxz(v):
    return v.index_select(1, g_xxxz_index)
def swizzle_t_n3_xxyx(v):
    return v.index_select(1, g_xxyx_index)
def swizzle_t_n3_xxyy(v):
    return v.index_select(1, g_xxyy_index)
def swizzle_t_n3_xxyz(v):
    return v.index_select(1, g_xxyz_index)
def swizzle_t_n3_xxzx(v):
    return v.index_select(1, g_xxzx_index)
def swizzle_t_n3_xxzy(v):
    return v.index_select(1, g_xxzy_index)
def swizzle_t_n3_xxzz(v):
    return v.index_select(1, g_xxzz_index)
def swizzle_t_n3_xyxx(v):
    return v.index_select(1, g_xyxx_index)
def swizzle_t_n3_xyxy(v):
    return v.index_select(1, g_xyxy_index)
def swizzle_t_n3_xyxz(v):
    return v.index_select(1, g_xyxz_index)
def swizzle_t_n3_xyyx(v):
    return v.index_select(1, g_xyyx_index)
def swizzle_t_n3_xyyy(v):
    return v.index_select(1, g_xyyy_index)
def swizzle_t_n3_xyyz(v):
    return v.index_select(1, g_xyyz_index)
def swizzle_t_n3_xyzx(v):
    return v.index_select(1, g_xyzx_index)
def swizzle_t_n3_xyzy(v):
    return v.index_select(1, g_xyzy_index)
def swizzle_t_n3_xyzz(v):
    return v.index_select(1, g_xyzz_index)
def swizzle_t_n3_xzxx(v):
    return v.index_select(1, g_xzxx_index)
def swizzle_t_n3_xzxy(v):
    return v.index_select(1, g_xzxy_index)
def swizzle_t_n3_xzxz(v):
    return v.index_select(1, g_xzxz_index)
def swizzle_t_n3_xzyx(v):
    return v.index_select(1, g_xzyx_index)
def swizzle_t_n3_xzyy(v):
    return v.index_select(1, g_xzyy_index)
def swizzle_t_n3_xzyz(v):
    return v.index_select(1, g_xzyz_index)
def swizzle_t_n3_xzzx(v):
    return v.index_select(1, g_xzzx_index)
def swizzle_t_n3_xzzy(v):
    return v.index_select(1, g_xzzy_index)
def swizzle_t_n3_xzzz(v):
    return v.index_select(1, g_xzzz_index)
def swizzle_t_n3_yxxx(v):
    return v.index_select(1, g_yxxx_index)
def swizzle_t_n3_yxxy(v):
    return v.index_select(1, g_yxxy_index)
def swizzle_t_n3_yxxz(v):
    return v.index_select(1, g_yxxz_index)
def swizzle_t_n3_yxyx(v):
    return v.index_select(1, g_yxyx_index)
def swizzle_t_n3_yxyy(v):
    return v.index_select(1, g_yxyy_index)
def swizzle_t_n3_yxyz(v):
    return v.index_select(1, g_yxyz_index)
def swizzle_t_n3_yxzx(v):
    return v.index_select(1, g_yxzx_index)
def swizzle_t_n3_yxzy(v):
    return v.index_select(1, g_yxzy_index)
def swizzle_t_n3_yxzz(v):
    return v.index_select(1, g_yxzz_index)
def swizzle_t_n3_yyxx(v):
    return v.index_select(1, g_yyxx_index)
def swizzle_t_n3_yyxy(v):
    return v.index_select(1, g_yyxy_index)
def swizzle_t_n3_yyxz(v):
    return v.index_select(1, g_yyxz_index)
def swizzle_t_n3_yyyx(v):
    return v.index_select(1, g_yyyx_index)
def swizzle_t_n3_yyyy(v):
    return v.index_select(1, g_yyyy_index)
def swizzle_t_n3_yyyz(v):
    return v.index_select(1, g_yyyz_index)
def swizzle_t_n3_yyzx(v):
    return v.index_select(1, g_yyzx_index)
def swizzle_t_n3_yyzy(v):
    return v.index_select(1, g_yyzy_index)
def swizzle_t_n3_yyzz(v):
    return v.index_select(1, g_yyzz_index)
def swizzle_t_n3_yzxx(v):
    return v.index_select(1, g_yzxx_index)
def swizzle_t_n3_yzxy(v):
    return v.index_select(1, g_yzxy_index)
def swizzle_t_n3_yzxz(v):
    return v.index_select(1, g_yzxz_index)
def swizzle_t_n3_yzyx(v):
    return v.index_select(1, g_yzyx_index)
def swizzle_t_n3_yzyy(v):
    return v.index_select(1, g_yzyy_index)
def swizzle_t_n3_yzyz(v):
    return v.index_select(1, g_yzyz_index)
def swizzle_t_n3_yzzx(v):
    return v.index_select(1, g_yzzx_index)
def swizzle_t_n3_yzzy(v):
    return v.index_select(1, g_yzzy_index)
def swizzle_t_n3_yzzz(v):
    return v.index_select(1, g_yzzz_index)
def swizzle_t_n3_zxxx(v):
    return v.index_select(1, g_zxxx_index)
def swizzle_t_n3_zxxy(v):
    return v.index_select(1, g_zxxy_index)
def swizzle_t_n3_zxxz(v):
    return v.index_select(1, g_zxxz_index)
def swizzle_t_n3_zxyx(v):
    return v.index_select(1, g_zxyx_index)
def swizzle_t_n3_zxyy(v):
    return v.index_select(1, g_zxyy_index)
def swizzle_t_n3_zxyz(v):
    return v.index_select(1, g_zxyz_index)
def swizzle_t_n3_zxzx(v):
    return v.index_select(1, g_zxzx_index)
def swizzle_t_n3_zxzy(v):
    return v.index_select(1, g_zxzy_index)
def swizzle_t_n3_zxzz(v):
    return v.index_select(1, g_zxzz_index)
def swizzle_t_n3_zyxx(v):
    return v.index_select(1, g_zyxx_index)
def swizzle_t_n3_zyxy(v):
    return v.index_select(1, g_zyxy_index)
def swizzle_t_n3_zyxz(v):
    return v.index_select(1, g_zyxz_index)
def swizzle_t_n3_zyyx(v):
    return v.index_select(1, g_zyyx_index)
def swizzle_t_n3_zyyy(v):
    return v.index_select(1, g_zyyy_index)
def swizzle_t_n3_zyyz(v):
    return v.index_select(1, g_zyyz_index)
def swizzle_t_n3_zyzx(v):
    return v.index_select(1, g_zyzx_index)
def swizzle_t_n3_zyzy(v):
    return v.index_select(1, g_zyzy_index)
def swizzle_t_n3_zyzz(v):
    return v.index_select(1, g_zyzz_index)
def swizzle_t_n3_zzxx(v):
    return v.index_select(1, g_zzxx_index)
def swizzle_t_n3_zzxy(v):
    return v.index_select(1, g_zzxy_index)
def swizzle_t_n3_zzxz(v):
    return v.index_select(1, g_zzxz_index)
def swizzle_t_n3_zzyx(v):
    return v.index_select(1, g_zzyx_index)
def swizzle_t_n3_zzyy(v):
    return v.index_select(1, g_zzyy_index)
def swizzle_t_n3_zzyz(v):
    return v.index_select(1, g_zzyz_index)
def swizzle_t_n3_zzzx(v):
    return v.index_select(1, g_zzzx_index)
def swizzle_t_n3_zzzy(v):
    return v.index_select(1, g_zzzy_index)
def swizzle_t_n3_zzzz(v):
    return v.index_select(1, g_zzzz_index)
def swizzle_t_n4_x(v):
    return torch.clone(v[..., 0])
def swizzle_t_n4_y(v):
    return torch.clone(v[..., 1])
def swizzle_t_n4_z(v):
    return torch.clone(v[..., 2])
def swizzle_t_n4_w(v):
    return torch.clone(v[..., 3])
def swizzle_t_n4_xx(v):
    return v.index_select(1, g_xx_index)
def swizzle_t_n4_xy(v):
    return v.index_select(1, g_xy_index)
def swizzle_t_n4_xz(v):
    return v.index_select(1, g_xz_index)
def swizzle_t_n4_xw(v):
    return v.index_select(1, g_xw_index)
def swizzle_t_n4_yx(v):
    return v.index_select(1, g_yx_index)
def swizzle_t_n4_yy(v):
    return v.index_select(1, g_yy_index)
def swizzle_t_n4_yz(v):
    return v.index_select(1, g_yz_index)
def swizzle_t_n4_yw(v):
    return v.index_select(1, g_yw_index)
def swizzle_t_n4_zx(v):
    return v.index_select(1, g_zx_index)
def swizzle_t_n4_zy(v):
    return v.index_select(1, g_zy_index)
def swizzle_t_n4_zz(v):
    return v.index_select(1, g_zz_index)
def swizzle_t_n4_zw(v):
    return v.index_select(1, g_zw_index)
def swizzle_t_n4_wx(v):
    return v.index_select(1, g_wx_index)
def swizzle_t_n4_wy(v):
    return v.index_select(1, g_wy_index)
def swizzle_t_n4_wz(v):
    return v.index_select(1, g_wz_index)
def swizzle_t_n4_ww(v):
    return v.index_select(1, g_ww_index)
def swizzle_t_n4_xxx(v):
    return v.index_select(1, g_xxx_index)
def swizzle_t_n4_xxy(v):
    return v.index_select(1, g_xxy_index)
def swizzle_t_n4_xxz(v):
    return v.index_select(1, g_xxz_index)
def swizzle_t_n4_xxw(v):
    return v.index_select(1, g_xxw_index)
def swizzle_t_n4_xyx(v):
    return v.index_select(1, g_xyx_index)
def swizzle_t_n4_xyy(v):
    return v.index_select(1, g_xyy_index)
def swizzle_t_n4_xyz(v):
    return v.index_select(1, g_xyz_index)
def swizzle_t_n4_xyw(v):
    return v.index_select(1, g_xyw_index)
def swizzle_t_n4_xzx(v):
    return v.index_select(1, g_xzx_index)
def swizzle_t_n4_xzy(v):
    return v.index_select(1, g_xzy_index)
def swizzle_t_n4_xzz(v):
    return v.index_select(1, g_xzz_index)
def swizzle_t_n4_xzw(v):
    return v.index_select(1, g_xzw_index)
def swizzle_t_n4_xwx(v):
    return v.index_select(1, g_xwx_index)
def swizzle_t_n4_xwy(v):
    return v.index_select(1, g_xwy_index)
def swizzle_t_n4_xwz(v):
    return v.index_select(1, g_xwz_index)
def swizzle_t_n4_xww(v):
    return v.index_select(1, g_xww_index)
def swizzle_t_n4_yxx(v):
    return v.index_select(1, g_yxx_index)
def swizzle_t_n4_yxy(v):
    return v.index_select(1, g_yxy_index)
def swizzle_t_n4_yxz(v):
    return v.index_select(1, g_yxz_index)
def swizzle_t_n4_yxw(v):
    return v.index_select(1, g_yxw_index)
def swizzle_t_n4_yyx(v):
    return v.index_select(1, g_yyx_index)
def swizzle_t_n4_yyy(v):
    return v.index_select(1, g_yyy_index)
def swizzle_t_n4_yyz(v):
    return v.index_select(1, g_yyz_index)
def swizzle_t_n4_yyw(v):
    return v.index_select(1, g_yyw_index)
def swizzle_t_n4_yzx(v):
    return v.index_select(1, g_yzx_index)
def swizzle_t_n4_yzy(v):
    return v.index_select(1, g_yzy_index)
def swizzle_t_n4_yzz(v):
    return v.index_select(1, g_yzz_index)
def swizzle_t_n4_yzw(v):
    return v.index_select(1, g_yzw_index)
def swizzle_t_n4_ywx(v):
    return v.index_select(1, g_ywx_index)
def swizzle_t_n4_ywy(v):
    return v.index_select(1, g_ywy_index)
def swizzle_t_n4_ywz(v):
    return v.index_select(1, g_ywz_index)
def swizzle_t_n4_yww(v):
    return v.index_select(1, g_yww_index)
def swizzle_t_n4_zxx(v):
    return v.index_select(1, g_zxx_index)
def swizzle_t_n4_zxy(v):
    return v.index_select(1, g_zxy_index)
def swizzle_t_n4_zxz(v):
    return v.index_select(1, g_zxz_index)
def swizzle_t_n4_zxw(v):
    return v.index_select(1, g_zxw_index)
def swizzle_t_n4_zyx(v):
    return v.index_select(1, g_zyx_index)
def swizzle_t_n4_zyy(v):
    return v.index_select(1, g_zyy_index)
def swizzle_t_n4_zyz(v):
    return v.index_select(1, g_zyz_index)
def swizzle_t_n4_zyw(v):
    return v.index_select(1, g_zyw_index)
def swizzle_t_n4_zzx(v):
    return v.index_select(1, g_zzx_index)
def swizzle_t_n4_zzy(v):
    return v.index_select(1, g_zzy_index)
def swizzle_t_n4_zzz(v):
    return v.index_select(1, g_zzz_index)
def swizzle_t_n4_zzw(v):
    return v.index_select(1, g_zzw_index)
def swizzle_t_n4_zwx(v):
    return v.index_select(1, g_zwx_index)
def swizzle_t_n4_zwy(v):
    return v.index_select(1, g_zwy_index)
def swizzle_t_n4_zwz(v):
    return v.index_select(1, g_zwz_index)
def swizzle_t_n4_zww(v):
    return v.index_select(1, g_zww_index)
def swizzle_t_n4_wxx(v):
    return v.index_select(1, g_wxx_index)
def swizzle_t_n4_wxy(v):
    return v.index_select(1, g_wxy_index)
def swizzle_t_n4_wxz(v):
    return v.index_select(1, g_wxz_index)
def swizzle_t_n4_wxw(v):
    return v.index_select(1, g_wxw_index)
def swizzle_t_n4_wyx(v):
    return v.index_select(1, g_wyx_index)
def swizzle_t_n4_wyy(v):
    return v.index_select(1, g_wyy_index)
def swizzle_t_n4_wyz(v):
    return v.index_select(1, g_wyz_index)
def swizzle_t_n4_wyw(v):
    return v.index_select(1, g_wyw_index)
def swizzle_t_n4_wzx(v):
    return v.index_select(1, g_wzx_index)
def swizzle_t_n4_wzy(v):
    return v.index_select(1, g_wzy_index)
def swizzle_t_n4_wzz(v):
    return v.index_select(1, g_wzz_index)
def swizzle_t_n4_wzw(v):
    return v.index_select(1, g_wzw_index)
def swizzle_t_n4_wwx(v):
    return v.index_select(1, g_wwx_index)
def swizzle_t_n4_wwy(v):
    return v.index_select(1, g_wwy_index)
def swizzle_t_n4_wwz(v):
    return v.index_select(1, g_wwz_index)
def swizzle_t_n4_www(v):
    return v.index_select(1, g_www_index)
def swizzle_t_n4_xxxx(v):
    return v.index_select(1, g_xxxx_index)
def swizzle_t_n4_xxxy(v):
    return v.index_select(1, g_xxxy_index)
def swizzle_t_n4_xxxz(v):
    return v.index_select(1, g_xxxz_index)
def swizzle_t_n4_xxxw(v):
    return v.index_select(1, g_xxxw_index)
def swizzle_t_n4_xxyx(v):
    return v.index_select(1, g_xxyx_index)
def swizzle_t_n4_xxyy(v):
    return v.index_select(1, g_xxyy_index)
def swizzle_t_n4_xxyz(v):
    return v.index_select(1, g_xxyz_index)
def swizzle_t_n4_xxyw(v):
    return v.index_select(1, g_xxyw_index)
def swizzle_t_n4_xxzx(v):
    return v.index_select(1, g_xxzx_index)
def swizzle_t_n4_xxzy(v):
    return v.index_select(1, g_xxzy_index)
def swizzle_t_n4_xxzz(v):
    return v.index_select(1, g_xxzz_index)
def swizzle_t_n4_xxzw(v):
    return v.index_select(1, g_xxzw_index)
def swizzle_t_n4_xxwx(v):
    return v.index_select(1, g_xxwx_index)
def swizzle_t_n4_xxwy(v):
    return v.index_select(1, g_xxwy_index)
def swizzle_t_n4_xxwz(v):
    return v.index_select(1, g_xxwz_index)
def swizzle_t_n4_xxww(v):
    return v.index_select(1, g_xxww_index)
def swizzle_t_n4_xyxx(v):
    return v.index_select(1, g_xyxx_index)
def swizzle_t_n4_xyxy(v):
    return v.index_select(1, g_xyxy_index)
def swizzle_t_n4_xyxz(v):
    return v.index_select(1, g_xyxz_index)
def swizzle_t_n4_xyxw(v):
    return v.index_select(1, g_xyxw_index)
def swizzle_t_n4_xyyx(v):
    return v.index_select(1, g_xyyx_index)
def swizzle_t_n4_xyyy(v):
    return v.index_select(1, g_xyyy_index)
def swizzle_t_n4_xyyz(v):
    return v.index_select(1, g_xyyz_index)
def swizzle_t_n4_xyyw(v):
    return v.index_select(1, g_xyyw_index)
def swizzle_t_n4_xyzx(v):
    return v.index_select(1, g_xyzx_index)
def swizzle_t_n4_xyzy(v):
    return v.index_select(1, g_xyzy_index)
def swizzle_t_n4_xyzz(v):
    return v.index_select(1, g_xyzz_index)
def swizzle_t_n4_xyzw(v):
    return v.index_select(1, g_xyzw_index)
def swizzle_t_n4_xywx(v):
    return v.index_select(1, g_xywx_index)
def swizzle_t_n4_xywy(v):
    return v.index_select(1, g_xywy_index)
def swizzle_t_n4_xywz(v):
    return v.index_select(1, g_xywz_index)
def swizzle_t_n4_xyww(v):
    return v.index_select(1, g_xyww_index)
def swizzle_t_n4_xzxx(v):
    return v.index_select(1, g_xzxx_index)
def swizzle_t_n4_xzxy(v):
    return v.index_select(1, g_xzxy_index)
def swizzle_t_n4_xzxz(v):
    return v.index_select(1, g_xzxz_index)
def swizzle_t_n4_xzxw(v):
    return v.index_select(1, g_xzxw_index)
def swizzle_t_n4_xzyx(v):
    return v.index_select(1, g_xzyx_index)
def swizzle_t_n4_xzyy(v):
    return v.index_select(1, g_xzyy_index)
def swizzle_t_n4_xzyz(v):
    return v.index_select(1, g_xzyz_index)
def swizzle_t_n4_xzyw(v):
    return v.index_select(1, g_xzyw_index)
def swizzle_t_n4_xzzx(v):
    return v.index_select(1, g_xzzx_index)
def swizzle_t_n4_xzzy(v):
    return v.index_select(1, g_xzzy_index)
def swizzle_t_n4_xzzz(v):
    return v.index_select(1, g_xzzz_index)
def swizzle_t_n4_xzzw(v):
    return v.index_select(1, g_xzzw_index)
def swizzle_t_n4_xzwx(v):
    return v.index_select(1, g_xzwx_index)
def swizzle_t_n4_xzwy(v):
    return v.index_select(1, g_xzwy_index)
def swizzle_t_n4_xzwz(v):
    return v.index_select(1, g_xzwz_index)
def swizzle_t_n4_xzww(v):
    return v.index_select(1, g_xzww_index)
def swizzle_t_n4_xwxx(v):
    return v.index_select(1, g_xwxx_index)
def swizzle_t_n4_xwxy(v):
    return v.index_select(1, g_xwxy_index)
def swizzle_t_n4_xwxz(v):
    return v.index_select(1, g_xwxz_index)
def swizzle_t_n4_xwxw(v):
    return v.index_select(1, g_xwxw_index)
def swizzle_t_n4_xwyx(v):
    return v.index_select(1, g_xwyx_index)
def swizzle_t_n4_xwyy(v):
    return v.index_select(1, g_xwyy_index)
def swizzle_t_n4_xwyz(v):
    return v.index_select(1, g_xwyz_index)
def swizzle_t_n4_xwyw(v):
    return v.index_select(1, g_xwyw_index)
def swizzle_t_n4_xwzx(v):
    return v.index_select(1, g_xwzx_index)
def swizzle_t_n4_xwzy(v):
    return v.index_select(1, g_xwzy_index)
def swizzle_t_n4_xwzz(v):
    return v.index_select(1, g_xwzz_index)
def swizzle_t_n4_xwzw(v):
    return v.index_select(1, g_xwzw_index)
def swizzle_t_n4_xwwx(v):
    return v.index_select(1, g_xwwx_index)
def swizzle_t_n4_xwwy(v):
    return v.index_select(1, g_xwwy_index)
def swizzle_t_n4_xwwz(v):
    return v.index_select(1, g_xwwz_index)
def swizzle_t_n4_xwww(v):
    return v.index_select(1, g_xwww_index)
def swizzle_t_n4_yxxx(v):
    return v.index_select(1, g_yxxx_index)
def swizzle_t_n4_yxxy(v):
    return v.index_select(1, g_yxxy_index)
def swizzle_t_n4_yxxz(v):
    return v.index_select(1, g_yxxz_index)
def swizzle_t_n4_yxxw(v):
    return v.index_select(1, g_yxxw_index)
def swizzle_t_n4_yxyx(v):
    return v.index_select(1, g_yxyx_index)
def swizzle_t_n4_yxyy(v):
    return v.index_select(1, g_yxyy_index)
def swizzle_t_n4_yxyz(v):
    return v.index_select(1, g_yxyz_index)
def swizzle_t_n4_yxyw(v):
    return v.index_select(1, g_yxyw_index)
def swizzle_t_n4_yxzx(v):
    return v.index_select(1, g_yxzx_index)
def swizzle_t_n4_yxzy(v):
    return v.index_select(1, g_yxzy_index)
def swizzle_t_n4_yxzz(v):
    return v.index_select(1, g_yxzz_index)
def swizzle_t_n4_yxzw(v):
    return v.index_select(1, g_yxzw_index)
def swizzle_t_n4_yxwx(v):
    return v.index_select(1, g_yxwx_index)
def swizzle_t_n4_yxwy(v):
    return v.index_select(1, g_yxwy_index)
def swizzle_t_n4_yxwz(v):
    return v.index_select(1, g_yxwz_index)
def swizzle_t_n4_yxww(v):
    return v.index_select(1, g_yxww_index)
def swizzle_t_n4_yyxx(v):
    return v.index_select(1, g_yyxx_index)
def swizzle_t_n4_yyxy(v):
    return v.index_select(1, g_yyxy_index)
def swizzle_t_n4_yyxz(v):
    return v.index_select(1, g_yyxz_index)
def swizzle_t_n4_yyxw(v):
    return v.index_select(1, g_yyxw_index)
def swizzle_t_n4_yyyx(v):
    return v.index_select(1, g_yyyx_index)
def swizzle_t_n4_yyyy(v):
    return v.index_select(1, g_yyyy_index)
def swizzle_t_n4_yyyz(v):
    return v.index_select(1, g_yyyz_index)
def swizzle_t_n4_yyyw(v):
    return v.index_select(1, g_yyyw_index)
def swizzle_t_n4_yyzx(v):
    return v.index_select(1, g_yyzx_index)
def swizzle_t_n4_yyzy(v):
    return v.index_select(1, g_yyzy_index)
def swizzle_t_n4_yyzz(v):
    return v.index_select(1, g_yyzz_index)
def swizzle_t_n4_yyzw(v):
    return v.index_select(1, g_yyzw_index)
def swizzle_t_n4_yywx(v):
    return v.index_select(1, g_yywx_index)
def swizzle_t_n4_yywy(v):
    return v.index_select(1, g_yywy_index)
def swizzle_t_n4_yywz(v):
    return v.index_select(1, g_yywz_index)
def swizzle_t_n4_yyww(v):
    return v.index_select(1, g_yyww_index)
def swizzle_t_n4_yzxx(v):
    return v.index_select(1, g_yzxx_index)
def swizzle_t_n4_yzxy(v):
    return v.index_select(1, g_yzxy_index)
def swizzle_t_n4_yzxz(v):
    return v.index_select(1, g_yzxz_index)
def swizzle_t_n4_yzxw(v):
    return v.index_select(1, g_yzxw_index)
def swizzle_t_n4_yzyx(v):
    return v.index_select(1, g_yzyx_index)
def swizzle_t_n4_yzyy(v):
    return v.index_select(1, g_yzyy_index)
def swizzle_t_n4_yzyz(v):
    return v.index_select(1, g_yzyz_index)
def swizzle_t_n4_yzyw(v):
    return v.index_select(1, g_yzyw_index)
def swizzle_t_n4_yzzx(v):
    return v.index_select(1, g_yzzx_index)
def swizzle_t_n4_yzzy(v):
    return v.index_select(1, g_yzzy_index)
def swizzle_t_n4_yzzz(v):
    return v.index_select(1, g_yzzz_index)
def swizzle_t_n4_yzzw(v):
    return v.index_select(1, g_yzzw_index)
def swizzle_t_n4_yzwx(v):
    return v.index_select(1, g_yzwx_index)
def swizzle_t_n4_yzwy(v):
    return v.index_select(1, g_yzwy_index)
def swizzle_t_n4_yzwz(v):
    return v.index_select(1, g_yzwz_index)
def swizzle_t_n4_yzww(v):
    return v.index_select(1, g_yzww_index)
def swizzle_t_n4_ywxx(v):
    return v.index_select(1, g_ywxx_index)
def swizzle_t_n4_ywxy(v):
    return v.index_select(1, g_ywxy_index)
def swizzle_t_n4_ywxz(v):
    return v.index_select(1, g_ywxz_index)
def swizzle_t_n4_ywxw(v):
    return v.index_select(1, g_ywxw_index)
def swizzle_t_n4_ywyx(v):
    return v.index_select(1, g_ywyx_index)
def swizzle_t_n4_ywyy(v):
    return v.index_select(1, g_ywyy_index)
def swizzle_t_n4_ywyz(v):
    return v.index_select(1, g_ywyz_index)
def swizzle_t_n4_ywyw(v):
    return v.index_select(1, g_ywyw_index)
def swizzle_t_n4_ywzx(v):
    return v.index_select(1, g_ywzx_index)
def swizzle_t_n4_ywzy(v):
    return v.index_select(1, g_ywzy_index)
def swizzle_t_n4_ywzz(v):
    return v.index_select(1, g_ywzz_index)
def swizzle_t_n4_ywzw(v):
    return v.index_select(1, g_ywzw_index)
def swizzle_t_n4_ywwx(v):
    return v.index_select(1, g_ywwx_index)
def swizzle_t_n4_ywwy(v):
    return v.index_select(1, g_ywwy_index)
def swizzle_t_n4_ywwz(v):
    return v.index_select(1, g_ywwz_index)
def swizzle_t_n4_ywww(v):
    return v.index_select(1, g_ywww_index)
def swizzle_t_n4_zxxx(v):
    return v.index_select(1, g_zxxx_index)
def swizzle_t_n4_zxxy(v):
    return v.index_select(1, g_zxxy_index)
def swizzle_t_n4_zxxz(v):
    return v.index_select(1, g_zxxz_index)
def swizzle_t_n4_zxxw(v):
    return v.index_select(1, g_zxxw_index)
def swizzle_t_n4_zxyx(v):
    return v.index_select(1, g_zxyx_index)
def swizzle_t_n4_zxyy(v):
    return v.index_select(1, g_zxyy_index)
def swizzle_t_n4_zxyz(v):
    return v.index_select(1, g_zxyz_index)
def swizzle_t_n4_zxyw(v):
    return v.index_select(1, g_zxyw_index)
def swizzle_t_n4_zxzx(v):
    return v.index_select(1, g_zxzx_index)
def swizzle_t_n4_zxzy(v):
    return v.index_select(1, g_zxzy_index)
def swizzle_t_n4_zxzz(v):
    return v.index_select(1, g_zxzz_index)
def swizzle_t_n4_zxzw(v):
    return v.index_select(1, g_zxzw_index)
def swizzle_t_n4_zxwx(v):
    return v.index_select(1, g_zxwx_index)
def swizzle_t_n4_zxwy(v):
    return v.index_select(1, g_zxwy_index)
def swizzle_t_n4_zxwz(v):
    return v.index_select(1, g_zxwz_index)
def swizzle_t_n4_zxww(v):
    return v.index_select(1, g_zxww_index)
def swizzle_t_n4_zyxx(v):
    return v.index_select(1, g_zyxx_index)
def swizzle_t_n4_zyxy(v):
    return v.index_select(1, g_zyxy_index)
def swizzle_t_n4_zyxz(v):
    return v.index_select(1, g_zyxz_index)
def swizzle_t_n4_zyxw(v):
    return v.index_select(1, g_zyxw_index)
def swizzle_t_n4_zyyx(v):
    return v.index_select(1, g_zyyx_index)
def swizzle_t_n4_zyyy(v):
    return v.index_select(1, g_zyyy_index)
def swizzle_t_n4_zyyz(v):
    return v.index_select(1, g_zyyz_index)
def swizzle_t_n4_zyyw(v):
    return v.index_select(1, g_zyyw_index)
def swizzle_t_n4_zyzx(v):
    return v.index_select(1, g_zyzx_index)
def swizzle_t_n4_zyzy(v):
    return v.index_select(1, g_zyzy_index)
def swizzle_t_n4_zyzz(v):
    return v.index_select(1, g_zyzz_index)
def swizzle_t_n4_zyzw(v):
    return v.index_select(1, g_zyzw_index)
def swizzle_t_n4_zywx(v):
    return v.index_select(1, g_zywx_index)
def swizzle_t_n4_zywy(v):
    return v.index_select(1, g_zywy_index)
def swizzle_t_n4_zywz(v):
    return v.index_select(1, g_zywz_index)
def swizzle_t_n4_zyww(v):
    return v.index_select(1, g_zyww_index)
def swizzle_t_n4_zzxx(v):
    return v.index_select(1, g_zzxx_index)
def swizzle_t_n4_zzxy(v):
    return v.index_select(1, g_zzxy_index)
def swizzle_t_n4_zzxz(v):
    return v.index_select(1, g_zzxz_index)
def swizzle_t_n4_zzxw(v):
    return v.index_select(1, g_zzxw_index)
def swizzle_t_n4_zzyx(v):
    return v.index_select(1, g_zzyx_index)
def swizzle_t_n4_zzyy(v):
    return v.index_select(1, g_zzyy_index)
def swizzle_t_n4_zzyz(v):
    return v.index_select(1, g_zzyz_index)
def swizzle_t_n4_zzyw(v):
    return v.index_select(1, g_zzyw_index)
def swizzle_t_n4_zzzx(v):
    return v.index_select(1, g_zzzx_index)
def swizzle_t_n4_zzzy(v):
    return v.index_select(1, g_zzzy_index)
def swizzle_t_n4_zzzz(v):
    return v.index_select(1, g_zzzz_index)
def swizzle_t_n4_zzzw(v):
    return v.index_select(1, g_zzzw_index)
def swizzle_t_n4_zzwx(v):
    return v.index_select(1, g_zzwx_index)
def swizzle_t_n4_zzwy(v):
    return v.index_select(1, g_zzwy_index)
def swizzle_t_n4_zzwz(v):
    return v.index_select(1, g_zzwz_index)
def swizzle_t_n4_zzww(v):
    return v.index_select(1, g_zzww_index)
def swizzle_t_n4_zwxx(v):
    return v.index_select(1, g_zwxx_index)
def swizzle_t_n4_zwxy(v):
    return v.index_select(1, g_zwxy_index)
def swizzle_t_n4_zwxz(v):
    return v.index_select(1, g_zwxz_index)
def swizzle_t_n4_zwxw(v):
    return v.index_select(1, g_zwxw_index)
def swizzle_t_n4_zwyx(v):
    return v.index_select(1, g_zwyx_index)
def swizzle_t_n4_zwyy(v):
    return v.index_select(1, g_zwyy_index)
def swizzle_t_n4_zwyz(v):
    return v.index_select(1, g_zwyz_index)
def swizzle_t_n4_zwyw(v):
    return v.index_select(1, g_zwyw_index)
def swizzle_t_n4_zwzx(v):
    return v.index_select(1, g_zwzx_index)
def swizzle_t_n4_zwzy(v):
    return v.index_select(1, g_zwzy_index)
def swizzle_t_n4_zwzz(v):
    return v.index_select(1, g_zwzz_index)
def swizzle_t_n4_zwzw(v):
    return v.index_select(1, g_zwzw_index)
def swizzle_t_n4_zwwx(v):
    return v.index_select(1, g_zwwx_index)
def swizzle_t_n4_zwwy(v):
    return v.index_select(1, g_zwwy_index)
def swizzle_t_n4_zwwz(v):
    return v.index_select(1, g_zwwz_index)
def swizzle_t_n4_zwww(v):
    return v.index_select(1, g_zwww_index)
def swizzle_t_n4_wxxx(v):
    return v.index_select(1, g_wxxx_index)
def swizzle_t_n4_wxxy(v):
    return v.index_select(1, g_wxxy_index)
def swizzle_t_n4_wxxz(v):
    return v.index_select(1, g_wxxz_index)
def swizzle_t_n4_wxxw(v):
    return v.index_select(1, g_wxxw_index)
def swizzle_t_n4_wxyx(v):
    return v.index_select(1, g_wxyx_index)
def swizzle_t_n4_wxyy(v):
    return v.index_select(1, g_wxyy_index)
def swizzle_t_n4_wxyz(v):
    return v.index_select(1, g_wxyz_index)
def swizzle_t_n4_wxyw(v):
    return v.index_select(1, g_wxyw_index)
def swizzle_t_n4_wxzx(v):
    return v.index_select(1, g_wxzx_index)
def swizzle_t_n4_wxzy(v):
    return v.index_select(1, g_wxzy_index)
def swizzle_t_n4_wxzz(v):
    return v.index_select(1, g_wxzz_index)
def swizzle_t_n4_wxzw(v):
    return v.index_select(1, g_wxzw_index)
def swizzle_t_n4_wxwx(v):
    return v.index_select(1, g_wxwx_index)
def swizzle_t_n4_wxwy(v):
    return v.index_select(1, g_wxwy_index)
def swizzle_t_n4_wxwz(v):
    return v.index_select(1, g_wxwz_index)
def swizzle_t_n4_wxww(v):
    return v.index_select(1, g_wxww_index)
def swizzle_t_n4_wyxx(v):
    return v.index_select(1, g_wyxx_index)
def swizzle_t_n4_wyxy(v):
    return v.index_select(1, g_wyxy_index)
def swizzle_t_n4_wyxz(v):
    return v.index_select(1, g_wyxz_index)
def swizzle_t_n4_wyxw(v):
    return v.index_select(1, g_wyxw_index)
def swizzle_t_n4_wyyx(v):
    return v.index_select(1, g_wyyx_index)
def swizzle_t_n4_wyyy(v):
    return v.index_select(1, g_wyyy_index)
def swizzle_t_n4_wyyz(v):
    return v.index_select(1, g_wyyz_index)
def swizzle_t_n4_wyyw(v):
    return v.index_select(1, g_wyyw_index)
def swizzle_t_n4_wyzx(v):
    return v.index_select(1, g_wyzx_index)
def swizzle_t_n4_wyzy(v):
    return v.index_select(1, g_wyzy_index)
def swizzle_t_n4_wyzz(v):
    return v.index_select(1, g_wyzz_index)
def swizzle_t_n4_wyzw(v):
    return v.index_select(1, g_wyzw_index)
def swizzle_t_n4_wywx(v):
    return v.index_select(1, g_wywx_index)
def swizzle_t_n4_wywy(v):
    return v.index_select(1, g_wywy_index)
def swizzle_t_n4_wywz(v):
    return v.index_select(1, g_wywz_index)
def swizzle_t_n4_wyww(v):
    return v.index_select(1, g_wyww_index)
def swizzle_t_n4_wzxx(v):
    return v.index_select(1, g_wzxx_index)
def swizzle_t_n4_wzxy(v):
    return v.index_select(1, g_wzxy_index)
def swizzle_t_n4_wzxz(v):
    return v.index_select(1, g_wzxz_index)
def swizzle_t_n4_wzxw(v):
    return v.index_select(1, g_wzxw_index)
def swizzle_t_n4_wzyx(v):
    return v.index_select(1, g_wzyx_index)
def swizzle_t_n4_wzyy(v):
    return v.index_select(1, g_wzyy_index)
def swizzle_t_n4_wzyz(v):
    return v.index_select(1, g_wzyz_index)
def swizzle_t_n4_wzyw(v):
    return v.index_select(1, g_wzyw_index)
def swizzle_t_n4_wzzx(v):
    return v.index_select(1, g_wzzx_index)
def swizzle_t_n4_wzzy(v):
    return v.index_select(1, g_wzzy_index)
def swizzle_t_n4_wzzz(v):
    return v.index_select(1, g_wzzz_index)
def swizzle_t_n4_wzzw(v):
    return v.index_select(1, g_wzzw_index)
def swizzle_t_n4_wzwx(v):
    return v.index_select(1, g_wzwx_index)
def swizzle_t_n4_wzwy(v):
    return v.index_select(1, g_wzwy_index)
def swizzle_t_n4_wzwz(v):
    return v.index_select(1, g_wzwz_index)
def swizzle_t_n4_wzww(v):
    return v.index_select(1, g_wzww_index)
def swizzle_t_n4_wwxx(v):
    return v.index_select(1, g_wwxx_index)
def swizzle_t_n4_wwxy(v):
    return v.index_select(1, g_wwxy_index)
def swizzle_t_n4_wwxz(v):
    return v.index_select(1, g_wwxz_index)
def swizzle_t_n4_wwxw(v):
    return v.index_select(1, g_wwxw_index)
def swizzle_t_n4_wwyx(v):
    return v.index_select(1, g_wwyx_index)
def swizzle_t_n4_wwyy(v):
    return v.index_select(1, g_wwyy_index)
def swizzle_t_n4_wwyz(v):
    return v.index_select(1, g_wwyz_index)
def swizzle_t_n4_wwyw(v):
    return v.index_select(1, g_wwyw_index)
def swizzle_t_n4_wwzx(v):
    return v.index_select(1, g_wwzx_index)
def swizzle_t_n4_wwzy(v):
    return v.index_select(1, g_wwzy_index)
def swizzle_t_n4_wwzz(v):
    return v.index_select(1, g_wwzz_index)
def swizzle_t_n4_wwzw(v):
    return v.index_select(1, g_wwzw_index)
def swizzle_t_n4_wwwx(v):
    return v.index_select(1, g_wwwx_index)
def swizzle_t_n4_wwwy(v):
    return v.index_select(1, g_wwwy_index)
def swizzle_t_n4_wwwz(v):
    return v.index_select(1, g_wwwz_index)
def swizzle_t_n4_wwww(v):
    return v.index_select(1, g_wwww_index)
def swizzle_set_t_n_x(v, val):
    v.copy_(val)
    return val
def swizzle_set_t_n2_x(v, val):
    v[..., 0] = val
    return val
def swizzle_set_t_n2_y(v, val):
    v[..., 1] = val
    return val
def swizzle_set_t_n2_xy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
def swizzle_set_t_n2_yx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
def swizzle_set_t_n3_x(v, val):
    v[..., 0] = val
    return val
def swizzle_set_t_n3_y(v, val):
    v[..., 1] = val
    return val
def swizzle_set_t_n3_z(v, val):
    v[..., 2] = val
    return val
def swizzle_set_t_n3_xy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
def swizzle_set_t_n3_xz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
def swizzle_set_t_n3_yx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
def swizzle_set_t_n3_yz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
def swizzle_set_t_n3_zx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
def swizzle_set_t_n3_zy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
def swizzle_set_t_n3_xyz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n3_xzy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n3_yxz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n3_yzx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n3_zxy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n3_zyx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_x(v, val):
    v[..., 0] = val
    return val
def swizzle_set_t_n4_y(v, val):
    v[..., 1] = val
    return val
def swizzle_set_t_n4_z(v, val):
    v[..., 2] = val
    return val
def swizzle_set_t_n4_w(v, val):
    v[..., 3] = val
    return val
def swizzle_set_t_n4_xy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
def swizzle_set_t_n4_xz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
def swizzle_set_t_n4_xw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    return val
def swizzle_set_t_n4_yx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
def swizzle_set_t_n4_yz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
def swizzle_set_t_n4_yw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    return val
def swizzle_set_t_n4_zx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
def swizzle_set_t_n4_zy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
def swizzle_set_t_n4_zw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    return val
def swizzle_set_t_n4_wx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
def swizzle_set_t_n4_wy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
def swizzle_set_t_n4_wz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
def swizzle_set_t_n4_xyz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n4_xyw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
def swizzle_set_t_n4_xzy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n4_xzw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
def swizzle_set_t_n4_xwy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n4_xwz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n4_yxz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n4_yxw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
def swizzle_set_t_n4_yzx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_yzw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
def swizzle_set_t_n4_ywx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_ywz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n4_zxy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n4_zxw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
def swizzle_set_t_n4_zyx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_zyw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
def swizzle_set_t_n4_zwx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_zwy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n4_wxy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n4_wxz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n4_wyx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_wyz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
def swizzle_set_t_n4_wzx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
def swizzle_set_t_n4_wzy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
def swizzle_set_t_n4_xyzw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
def swizzle_set_t_n4_xywz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
def swizzle_set_t_n4_xzyw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
def swizzle_set_t_n4_xzwy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
def swizzle_set_t_n4_xwyz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
def swizzle_set_t_n4_xwzy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
def swizzle_set_t_n4_yxzw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
def swizzle_set_t_n4_yxwz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
def swizzle_set_t_n4_yzxw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
def swizzle_set_t_n4_yzwx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
def swizzle_set_t_n4_ywxz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
def swizzle_set_t_n4_ywzx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
def swizzle_set_t_n4_zxyw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
def swizzle_set_t_n4_zxwy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
def swizzle_set_t_n4_zyxw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
def swizzle_set_t_n4_zywx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
def swizzle_set_t_n4_zwxy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
def swizzle_set_t_n4_zwyx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
def swizzle_set_t_n4_wxyz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
def swizzle_set_t_n4_wxzy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
def swizzle_set_t_n4_wyxz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
def swizzle_set_t_n4_wyzx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
def swizzle_set_t_n4_wzxy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
def swizzle_set_t_n4_wzyx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
def swizzle_set_and_broadcast_n_x(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n_x(v, val), v
def swizzle_set_and_broadcast_n2_x(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n2_x(v, val), v
def swizzle_set_and_broadcast_n2_y(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n2_y(v, val), v
def swizzle_set_and_broadcast_n2_xy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n2_xy(v, val), v
def swizzle_set_and_broadcast_n2_yx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n2_yx(v, val), v
def swizzle_set_and_broadcast_n3_x(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_x(v, val), v
def swizzle_set_and_broadcast_n3_y(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_y(v, val), v
def swizzle_set_and_broadcast_n3_z(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_z(v, val), v
def swizzle_set_and_broadcast_n3_xy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xy(v, val), v
def swizzle_set_and_broadcast_n3_xz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xz(v, val), v
def swizzle_set_and_broadcast_n3_yx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yx(v, val), v
def swizzle_set_and_broadcast_n3_yz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yz(v, val), v
def swizzle_set_and_broadcast_n3_zx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zx(v, val), v
def swizzle_set_and_broadcast_n3_zy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zy(v, val), v
def swizzle_set_and_broadcast_n3_xyz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xyz(v, val), v
def swizzle_set_and_broadcast_n3_xzy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xzy(v, val), v
def swizzle_set_and_broadcast_n3_yxz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yxz(v, val), v
def swizzle_set_and_broadcast_n3_yzx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yzx(v, val), v
def swizzle_set_and_broadcast_n3_zxy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zxy(v, val), v
def swizzle_set_and_broadcast_n3_zyx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zyx(v, val), v
def swizzle_set_and_broadcast_n4_x(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_x(v, val), v
def swizzle_set_and_broadcast_n4_y(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_y(v, val), v
def swizzle_set_and_broadcast_n4_z(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_z(v, val), v
def swizzle_set_and_broadcast_n4_w(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_w(v, val), v
def swizzle_set_and_broadcast_n4_xy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xy(v, val), v
def swizzle_set_and_broadcast_n4_xz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xz(v, val), v
def swizzle_set_and_broadcast_n4_xw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xw(v, val), v
def swizzle_set_and_broadcast_n4_yx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yx(v, val), v
def swizzle_set_and_broadcast_n4_yz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yz(v, val), v
def swizzle_set_and_broadcast_n4_yw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yw(v, val), v
def swizzle_set_and_broadcast_n4_zx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zx(v, val), v
def swizzle_set_and_broadcast_n4_zy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zy(v, val), v
def swizzle_set_and_broadcast_n4_zw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zw(v, val), v
def swizzle_set_and_broadcast_n4_wx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wx(v, val), v
def swizzle_set_and_broadcast_n4_wy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wy(v, val), v
def swizzle_set_and_broadcast_n4_wz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wz(v, val), v
def swizzle_set_and_broadcast_n4_xyz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xyz(v, val), v
def swizzle_set_and_broadcast_n4_xyw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xyw(v, val), v
def swizzle_set_and_broadcast_n4_xzy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzy(v, val), v
def swizzle_set_and_broadcast_n4_xzw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzw(v, val), v
def swizzle_set_and_broadcast_n4_xwy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwy(v, val), v
def swizzle_set_and_broadcast_n4_xwz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwz(v, val), v
def swizzle_set_and_broadcast_n4_yxz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxz(v, val), v
def swizzle_set_and_broadcast_n4_yxw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxw(v, val), v
def swizzle_set_and_broadcast_n4_yzx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzx(v, val), v
def swizzle_set_and_broadcast_n4_yzw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzw(v, val), v
def swizzle_set_and_broadcast_n4_ywx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywx(v, val), v
def swizzle_set_and_broadcast_n4_ywz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywz(v, val), v
def swizzle_set_and_broadcast_n4_zxy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxy(v, val), v
def swizzle_set_and_broadcast_n4_zxw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxw(v, val), v
def swizzle_set_and_broadcast_n4_zyx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zyx(v, val), v
def swizzle_set_and_broadcast_n4_zyw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zyw(v, val), v
def swizzle_set_and_broadcast_n4_zwx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwx(v, val), v
def swizzle_set_and_broadcast_n4_zwy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwy(v, val), v
def swizzle_set_and_broadcast_n4_wxy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxy(v, val), v
def swizzle_set_and_broadcast_n4_wxz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxz(v, val), v
def swizzle_set_and_broadcast_n4_wyx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyx(v, val), v
def swizzle_set_and_broadcast_n4_wyz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyz(v, val), v
def swizzle_set_and_broadcast_n4_wzx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzx(v, val), v
def swizzle_set_and_broadcast_n4_wzy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzy(v, val), v
def swizzle_set_and_broadcast_n4_xyzw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xyzw(v, val), v
def swizzle_set_and_broadcast_n4_xywz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xywz(v, val), v
def swizzle_set_and_broadcast_n4_xzyw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzyw(v, val), v
def swizzle_set_and_broadcast_n4_xzwy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzwy(v, val), v
def swizzle_set_and_broadcast_n4_xwyz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwyz(v, val), v
def swizzle_set_and_broadcast_n4_xwzy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwzy(v, val), v
def swizzle_set_and_broadcast_n4_yxzw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxzw(v, val), v
def swizzle_set_and_broadcast_n4_yxwz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxwz(v, val), v
def swizzle_set_and_broadcast_n4_yzxw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzxw(v, val), v
def swizzle_set_and_broadcast_n4_yzwx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzwx(v, val), v
def swizzle_set_and_broadcast_n4_ywxz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywxz(v, val), v
def swizzle_set_and_broadcast_n4_ywzx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywzx(v, val), v
def swizzle_set_and_broadcast_n4_zxyw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxyw(v, val), v
def swizzle_set_and_broadcast_n4_zxwy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxwy(v, val), v
def swizzle_set_and_broadcast_n4_zyxw(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zyxw(v, val), v
def swizzle_set_and_broadcast_n4_zywx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zywx(v, val), v
def swizzle_set_and_broadcast_n4_zwxy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwxy(v, val), v
def swizzle_set_and_broadcast_n4_zwyx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwyx(v, val), v
def swizzle_set_and_broadcast_n4_wxyz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxyz(v, val), v
def swizzle_set_and_broadcast_n4_wxzy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxzy(v, val), v
def swizzle_set_and_broadcast_n4_wyxz(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyxz(v, val), v
def swizzle_set_and_broadcast_n4_wyzx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyzx(v, val), v
def swizzle_set_and_broadcast_n4_wzxy(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzxy(v, val), v
def swizzle_set_and_broadcast_n4_wzyx(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzyx(v, val), v
#---end---
#----------------------------------------


#pip install matplotlib numpy pyjion imageio PyOpenGL glfw
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

import matplotlib.pyplot as plt
import matplotlib.animation as mpanim
import imageio.v3 as iio
import OpenGL.GL as gl
import glfw
import nvdiffrast.torch as dr
import time
import functools
import cProfile
import pstats
import io
import os
from pstats import SortKey

def any_ifexp_true_n(v):
    return v
def not_all_ifexp_true_n(v):
    return not v
def any_ifexp_true_t_n(v):
    return torch.any(v)
def not_all_ifexp_true_t_n(v):
    return not torch.all(v)

def array_copy(v):
    return torch.clone(v)

def init_buffer():
    global iResolution
    return torch.broadcast_to(torch.asarray([[0.0, 0.0, 0.0, 1.0]], device=device), (iResolution[0] * iResolution[1], 4))
def buffer_to_tex(v):
    global iResolution
    img = torch.stack(torch.tensor_split(v, iResolution[1]))
    img = img.cpu().numpy()
    img = img[::-1, :, :] # Flip vertically.
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
    data = np.flip(np.transpose(img, (1, 0, 2)), 1)
    return torch.from_numpy(data.copy()).float().cuda()

def load_tex_2d(file):
    data = iio.imread(file)
    if len(data.shape) == 2:
        data = np.flip(np.transpose(data), 1)
    else:
        data = np.flip(np.transpose(data, (1, 0, 2)), 1)
    #return torch.as_tensor(data.astype(np.float64), device=device, requires_grad=True)
    return torch.from_numpy(data.copy()).float().cuda()

def load_tex_cube(file):
    nameAndExts = os.path.splitext(file)
    data = iio.imread(file)
    data1 = iio.imread(nameAndExts[0]+"_1"+nameAndExts[1])
    data2 = iio.imread(nameAndExts[0]+"_2"+nameAndExts[1])
    data3 = iio.imread(nameAndExts[0]+"_3"+nameAndExts[1])
    data4 = iio.imread(nameAndExts[0]+"_4"+nameAndExts[1])
    data5 = iio.imread(nameAndExts[0]+"_5"+nameAndExts[1])
    datas = [data, data1, data2, data3, data4, data5]
    for i in range(5):
        d = datas[i]
        if len(data.shape) == 2:
            d = np.flip(np.transpose(d), 1)
        else:
            d = np.flip(np.transpose(d, (1, 0, 2)), 1)
        datas[i] = d
    ds = np.moveaxis(np.stack(datas), 0, 2)
    return torch.from_numpy(ds.copy()).float().cuda()

def load_tex_3d(file):
    f = open(file, "rb")
    head = np.fromfile(f, dtype=np.uint32, count=5, offset=0)
    tag = head[0]
    w = head[1]
    h = head[2]
    d = head[3]
    data = np.fromfile(f, dtype=np.uint8, count=-1, offset=0)
    f.close()
    data = data.reshape(w, h, d)
    return torch.from_numpy(data.copy()).float().cuda()
    '''
    data = iio.imread(file)
    if len(data.shape) == 2:
        return np.flip(np.transpose(data), 1)
    else:
        return np.flip(np.transpose(data, (1, 0, 2)), 1)
    '''

def set_channel_resolution(ix, buf):
    global iChannelResolution
    if buf is not None:
        shape = buf.shape
        n = len(shape)
        for i in range(3):
            if i < n:
                iChannelResolution[ix][i] = shape[i] 

def mouse_button_callback(window, button, action, mods):
    global iMouse, iResolution
    x_pos, y_pos = glfw.get_cursor_pos(window)
    y_pos = iResolution[1] - y_pos
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        pass
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        print("mouse {0} press, ({1}, {2})".format(str(button), x_pos, y_pos))
        iMouse[0] = x_pos
        iMouse[1] = y_pos
        iMouse[2] = x_pos
        iMouse[3] = y_pos

def cursor_pos_callback(window, x_pos, y_pos):
    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
        print("mouse move, ({0}, {1})".format(x_pos, y_pos))
        iMouse[0] = x_pos
        iMouse[1] = y_pos
    elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        pass

def scroll_callback(window, x_offset, y_offset):
    x_pos, y_pos = glfw.get_cursor_pos(window)
    pass

def key_callback(window, key, scancode, action, mods):
    if action==glfw.PRESS:
        pass
    elif action==glfw.RELEASE:
        pass

def display_image(image, zoom=None, size=None, title=None): # HWC

    # Zoom image if requested.
    image = np.asarray(image)

    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global g_glfw_window
    if g_glfw_window is None:
        glfw.init()
        g_glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(g_glfw_window)
        glfw.show_window(g_glfw_window)
        glfw.swap_interval(0)
        glfw.set_mouse_button_callback(g_glfw_window, mouse_button_callback)
        glfw.set_cursor_pos_callback(g_glfw_window, cursor_pos_callback)
        glfw.set_scroll_callback(g_glfw_window, scroll_callback)
        glfw.set_key_callback(g_glfw_window, key_callback)
    else:
        glfw.make_context_current(g_glfw_window)
        glfw.set_window_title(g_glfw_window, title)
        glfw.set_window_size(g_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {4: gl.GL_RGBA, 3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(g_glfw_window)
    if glfw.window_should_close(g_glfw_window):
        return False
    return True

def save_image(fn, x):
    x = torch.rint(x * 255.0)
    x = torch.clip(x, 0, 255).astype(torch.uint8)
    iio.imsave(fn, x)

def update(frame, ax, fc, fcd):
    global iTime, iTimeDelta, iFrame, iFrameRate, iResolution, g_last_time
    curTime = time.time()
    iTimeDelta = curTime - g_last_time
    g_last_time = curTime
    iTime += iTimeDelta
    iFrame += 1
    if iTimeDelta > 0.01:
        iFrameRate = iFrameRate * 0.7 + 0.3 / iTimeDelta
    
    V = shader_main(fc, fcd)
    maxv = abs(V).max()
    minv = -abs(V).max()
    V = torch.stack(torch.tensor_split(V, iResolution[1]))
    V = get_cpu_value(V)
	
    ax.clear()
    info = "time:{0:.3f} frame time:{1:.3f} avg frame time:{2:.3f} iter:{3}".format(iTime, iTimeDelta, iTime/iFrame, iFrame)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(info)
    ax.text(0.0, 1.0, info)
    im = ax.imshow(V, interpolation='bilinear',
                   origin='lower', extent=[0, iResolution[0], 0, iResolution[1]],
                   vmax=maxv, vmin=-minv)
    tensor_pools.RecycleAll()

def on_press(event):
    global iMouse
    print("mouse {0} press, ({1}, {2})".format(str(event.button), event.xdata, event.ydata))
    iMouse[0] = event.xdata
    iMouse[1] = event.ydata
    iMouse[2] = event.xdata
    iMouse[3] = event.ydata
    
def on_release(event):
    print(str(event.button)+" release.")

def on_motion(event):
    if event.button == 1:
        print("mouse move: {0} {1} - {2} {3} button:{4}".format(event.x, event.y, event.xdata, event.ydata, event.button))
        iMouse[0] = event.xdata
        iMouse[1] = event.ydata
    elif event.button == 2:
        pass
    elif event.button == 3:
        pass

def on_scroll(event):
    #print("mouse scroll: {0}".format(event.step))
    pass

def on_key_press(event):
    print(event.key+" press.")

def on_key_release(event):
    print(event.key+" release.")

def main_entry():
    global iTime, iTimeDelta, iFrame, iFrameRate, iResolution, g_last_time, g_show_with_opengl, g_is_profiling, g_face_color, g_win_zoom, g_win_size
    torch.manual_seed(19680801)
    iTimeDelta = 0
    iTime = 0
    iFrame = 0
    
    coordx = torch.arange(0.0, iResolution[0])
    coordy = torch.arange(0.0, iResolution[1])
    X, Y = torch.meshgrid(coordx, coordy, indexing="xy")
    X = torch.reshape(X, (-1, ))
    Y = torch.reshape(Y, (-1, ))
    
    fcd = torch.column_stack((X, Y))
    #fc = torch.broadcast_to(hlsl_float4_n_n_n_n(0.5, 0.5, 0.5, 1.0), (iResolution[0], iResolution[1], 4), axis=0)
    fc = torch.as_tensor([0.5,0.5,0.5,1.0], device=device)

    fcd = fcd.cuda()
    fc = fc.cuda()

    print(fcd.is_cuda, fc.is_cuda)

    if g_show_with_opengl:
        g_last_time = time.time()
        iterCount = 10 if g_is_profiling else 1000
        for ct in range(iterCount):
            curTime = time.time()
            iTimeDelta = curTime - g_last_time
            iTime += iTimeDelta
            iFrame += 1
            if iTimeDelta > 0.01:
                iFrameRate = iFrameRate * 0.7 + 0.3 / iTimeDelta
            g_last_time = curTime

            V = shader_main(fc, fcd)

            img = torch.stack(torch.tensor_split(V, iResolution[1]))
            img = img.cpu().numpy()
            img = img[::-1, :, :] # Flip vertically.
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
			
            wtitle = "time:{0:.3f} frame time:{1:.3f} avg frame time:{2:.3f} iter:{3}".format(iTime, iTimeDelta, iTime/iFrame, ct)
            display_image(img, g_win_zoom, g_win_size, wtitle)
            tensor_pools.RecycleAll()
            time.sleep(0.033)
    else:
        fig, ax = plt.subplots()
        ax.set_facecolor(g_face_color)
        cidpress = fig.canvas.mpl_connect('button_press_event', on_press)
        cidrelease = fig.canvas.mpl_connect('button_release_event', on_release)
        cidmotion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        cidscroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
        kpid = fig.canvas.mpl_connect('key_press_event', on_key_press)
        krid = fig.canvas.mpl_connect('key_release_event', on_key_release)
        g_last_time = time.time()
        ani = mpanim.FuncAnimation(fig, functools.partial(update, ax = ax, fc = fc, fcd = fcd), interval = 100.0, repeat = not g_is_profiling)
        plt.show()
        #plt.pause(0.1)

class MyProfiler:
    def __init__(self) -> None:
        self.datas = {}
    def beginSample(self, tag):
        data = self.datas.get(tag)
        if data is None:
            data = [0, 0.0, 0.0]
            self.datas[tag] = data
        data[0] += 1
        data[1] = time.time()
    def endSample(self, tag, is_show=True):
        data = self.datas.get(tag)
        if data is not None:
            data[1] = time.time() - data[1]
            data[2] += data[1]
            if is_show:
                print("{0} ct:{1} time:{2} total:{3} avg:{4}".format(tag, data[0], data[1], data[2], data[2]/data[0]))
    def ShowTotal(self):
        print("[total:]")
        for tag, data in self.datas.items():
            print("{0} ct:{1} total:{2} avg:{3}".format(tag, data[0], data[2], data[2]/data[0]))

def profile_entry(real_entry):
    pr = cProfile.Profile()
    pr.enable()
    real_entry()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE #SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

def main_entry_autodiff():
    global iTime, iTimeDelta, iFrame, iFrameRate, iResolution, g_last_time, g_target_img, g_show_with_opengl, g_is_profiling, g_face_color, g_win_zoom, g_win_size
    torch.manual_seed(19680801)
    iTimeDelta = 0
    iTime = 0
    iFrame = 0

    coordx = torch.arange(0.0, iResolution[0])
    coordy = torch.arange(0.0, iResolution[1])
    X, Y = torch.meshgrid(coordx, coordy, indexing="xy")
    X = torch.reshape(X, (-1, ))
    Y = torch.reshape(Y, (-1, ))

    g_target_img = iio.imread("target.jpg")
    g_target_img = torch.from_numpy(g_target_img).float()
    g_target_img = get_gpu_value(g_target_img)

    target = torch.flatten(g_target_img)[0:(iResolution[0]*iResolution[1])]
    optimizer = torch.optim.SGD(params=[iChannel2], lr = 10000.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    loss_f = torch.nn.L1Loss()

    fcd = torch.column_stack((X, Y))
    #fc = torch.repeat_interleave(torch.as_tensor([[0.5, 0.5, 0.5, 1.0]], device=device), (iResolution[0] * iResolution[1]), dim=0)
    fc = torch.as_tensor([0.5, 0.5, 0.5, 1.0], device=device)

    fcd = fcd.cuda()
    fc = fc.cuda()

    print(fcd.is_cuda, fc.is_cuda)

    if g_show_with_opengl:
        g_last_time = time.time()
        epoch = 10
        iterCount = 1 if g_is_profiling else 1000
        for st in range(epoch):
            for ct in range(iterCount):
                curTime = time.time()
                #iTime += curTime - g_last_time
                g_last_time = curTime

                optimizer.zero_grad()
                V = shader_main(fc, fcd)
                vs = V[..., 0] * 255.0
                loss = loss_f(vs, target)
                loss.backward()
                optimizer.step()

                img = torch.stack(torch.tensor_split(V, iResolution[1]))
                img = img.cpu().detach().numpy()
                img = img[::-1, :, :] # Flip vertically.
                img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

                info = "epoch:" + str(st) + "iter:" + str(ct) + " grad:" + str(loss)
                print(info)
                display_image(img, g_win_zoom, g_win_size, info)
                tensor_pools.RecycleAll()
                time.sleep(0.033)
            scheduler.step()

        iio.imsave("autogen.jpg", iChannel2.cpu().detach().numpy())
    else:
        fig, ax = plt.subplots()
        g_last_time = time.time()
        ani = mpanim.FuncAnimation(fig, functools.partial(update, ax = ax, fc = fc, fcd = fcd), interval = 100.0, repeat = not g_is_profiling)
        plt.show()
        #plt.pause(0.1)

_ = None
g_last_time = 0.0
g_target_img = None
g_glfw_window = None
g_show_with_opengl = False
g_is_profiling = False
g_is_autodiff = False
g_is_full_vectorized = False

g_main_iChannel0 = None
g_main_iChannel1 = None
g_main_iChannel2 = None
g_main_iChannel3 = None

g_bufferA_iChannel0 = None
g_bufferA_iChannel1 = None
g_bufferA_iChannel2 = None
g_bufferA_iChannel3 = None

g_bufferB_iChannel0 = None
g_bufferB_iChannel1 = None
g_bufferB_iChannel2 = None
g_bufferB_iChannel3 = None

g_bufferC_iChannel0 = None
g_bufferC_iChannel1 = None
g_bufferC_iChannel2 = None
g_bufferC_iChannel3 = None

g_bufferD_iChannel0 = None
g_bufferD_iChannel1 = None
g_bufferD_iChannel2 = None
g_bufferD_iChannel3 = None

g_bufferCubemap_iChannel0 = None
g_bufferCubemap_iChannel1 = None
g_bufferCubemap_iChannel2 = None
g_bufferCubemap_iChannel3 = None

g_bufferSound_iChannel0 = None
g_bufferSound_iChannel1 = None
g_bufferSound_iChannel2 = None
g_bufferSound_iChannel3 = None

bufferA = None
bufferB = None
bufferC = None
bufferD = None
bufferCubemap = None
bufferSound = None

g_face_color = "gray"
g_win_zoom = 1.0
g_win_size = None

def compute_dispatch(fc, fcd, entry):
    pass

def shader_dispatch(fc, fcd, entry):
    pass

def rmRes_h(lhs):
	return lhs[2]

def rmRes_p(lhs):
	return lhs[0]

def h_copy_t_rmRes(v):
	return [h_copy_f3(v[0]), v[1], v[2]]

def rmRes_h_set(lhs, rhs):
	lhs[2] = rhs
	return rhs

def h_where_t_n_t_rmRes_t_rmRes(b, y, n):
	return [h_where_t_n_t_v_t_v(b, y[0], n[0]), h_where_t_n_t_n_t_n(b, y[1], n[1]), h_where_t_n_t_n_t_n(b, y[2], n[2])]

def rmRes_p_set(lhs, rhs):
	lhs[0] = h_copy_f3(rhs)
	return rhs

def h_where_n_t_rmRes_t_rmRes(b, y, n):
	return [h_where_n_t_v_t_v(b, y[0], n[0]), h_where_n_t_n_t_n(b, y[1], n[1]), h_where_n_t_n_t_n(b, y[2], n[2])]

def rmRes_i(lhs):
	return lhs[1]

def rmRes_i_set(lhs, rhs):
	lhs[1] = rhs
	return rhs

def h_cast_t_rmRes_tp3_t_f3_i_b(rhs):
	ct = len(rhs[0])
	return [h_copy_t_f3(rhs[0]), h_broadcast_i(ct, rhs[1]), h_broadcast_b(ct, rhs[2])]

def glsl_vec3_f(arg):
	return h_f3_n_n_n(arg, arg, arg)

def rot_f(a):
	return h_f2x2_n_n_n_n(h_cos_n(a), h_sin_n(a), h_sub_f(h_sin_n(a)), h_cos_n(a))

def glsl_float_x4_ctor_f_f_f_f(v0, v1, v2, v3):
	__arr_tmp = array_init_an([v0, v1, v2, v3])
	return __arr_tmp

def glsl_vec3_x4_ctor_f3_f3_f3_f3(v0, v1, v2, v3):
	__arr_tmp = array_init_an3([v0, v1, v2, v3])
	return __arr_tmp


def petalDcp_f2_f_arr(uv, w):
	swizzle_set_t_n2_x(uv, h_add_t_f_t_f(h_add_t_f_f(h_abs_t_n(swizzle_t_n2_x(uv)), 0.25), h_mul_f_t_f(0.25, w)))
	return h_sub_t_f_f(h_length_t_v(uv), 0.5)

def rot_f_arr(a):
	return h_t_f2x2_t_n_t_n_t_n_t_n(h_cos_t_n(a), h_sin_t_n(a), h_sub_t_f(h_sin_t_n(a)), h_cos_t_n(a))

def petal_f3_f_arr(p, m):
	global gameTime
	tt = h_sub_f_f(gameTime, h_mul_f_f(h_mul_f_f(h_floor_n(h_mul_f_f(h_div_f_f(gameTime, 6.2831853070000001), 0.5)), 6.2831853070000001), 0.5))
	ouv = h_sub_t_f_f(m, 0.014999999999999999)
	w = m
	a = m
	b = 0.5
	swizzle_set_t_n3_y(p, h_sub_t_f_f(swizzle_t_n3_y(p), 0.45000000000000001))
	swizzle_set_t_n3_z(p, h_sub_t_f_f(swizzle_t_n3_z(p), h_mul_f_f(0.5, 1.0)))
	swizzle_set_t_n3_zy(p, h_matmul_t_f2x2_t_f2(rot_f_arr(h_mul_t_f_f(ouv, 2.0)), swizzle_t_n3_zy(p)))
	pDcp = petalDcp_f2_f_arr(swizzle_t_n3_xy(p), w)
	swizzle_set_t_n3_x(p, h_abs_t_n(swizzle_t_n3_x(p)))
	swizzle_set_t_n3_xz(p, h_matmul_f2x2_t_f2(rot_f(-0.25), swizzle_t_n3_xz(p)))
	c1 = h_sub_t_f_f(h_length_t_v(swizzle_t_n3_yz(p)), 0.5)
	return h_max_t_n_t_n(h_max_t_n_t_n(pDcp, h_sub_t_f_f(h_abs_t_n(c1), 0.01)), swizzle_t_n3_z(p))

def repRot_f2_f_arr(p, aIt):
	return h_matmul_t_f2x2_t_f2(rot_f_arr(h_sub_t_f_t_f(h_sub_t_f_f(h_mul_t_f_t_f(h_sub_t_f(h_div_f_t_f(6.2831853070000001, aIt)), h_floor_t_n(h_mul_t_f_t_f(h_add_t_f_f(h_div_t_f_f(h_atan2_t_n_t_n(swizzle_t_n2_x(p), swizzle_t_n2_y(p)), 6.2831853070000001), 0.5), aIt))), 3.141593), h_div_f_t_f(6.2831853070000001, h_mul_t_f_f(aIt, 2.0)))), p)

def opSmoothUnion_f_f_f_arr1(d1, d2, k):
	h = h_clamp_t_n_n_n(h_add_f_t_f(0.5, h_div_t_f_f(h_mul_f_t_f(0.5, h_sub_t_f_t_f(d2, d1)), k)), 0.0, 1.0)
	return h_sub_t_f_t_f(h_lerp_t_n_t_n_t_n(d2, d1, h), h_mul_t_f_t_f(h_mul_f_t_f(k, h), h_sub_f_t_f(1.0, h)))

def flower_f3_f_f_arr(p, aIt, m):
	swizzle_set_t_n3_xy(p, repRot_f2_f_arr(swizzle_t_n3_xy(p), aIt))
	return petal_f3_f_arr(h_copy_t_f3(p), m)

def opSmoothUnion_f_f_f_arr(d1, d2, k):
	h = h_clamp_t_n_n_n(h_add_f_t_f(0.5, h_div_t_f_f(h_mul_f_t_f(0.5, h_sub_t_f_f(d2, d1)), k)), 0.0, 1.0)
	return h_sub_t_f_t_f(h_lerp_t_n_n_t_n(d2, d1, h), h_mul_t_f_t_f(h_mul_f_t_f(k, h), h_sub_f_t_f(1.0, h)))

def df_f3_i_arr1(_pp, m):
	global gameTime
	swizzle_set_t_n3_y(_pp, h_sub_t_f(swizzle_t_n3_y(_pp)))
	swizzle_set_t_n3_xz(_pp, h_matmul_f2x2_t_f2(rot_f(1.016), swizzle_t_n3_xz(_pp)))
	swizzle_set_t_n3_xy(_pp, h_matmul_f2x2_t_f2(rot_f(-0.64000000000000001), swizzle_t_n3_xy(_pp)))
	dd = 1.0E+10
	ee = 1.0E+10
	p = _pp
	fsz = 0.25
	n = h_f2_n_n(h_cos_n(0.3926991), h_sin_n(0.3926991))
	b = False
	if True:
		g = 0.0
		p = h_where_n_t_v_t_v((b := h_not_n(b)), swizzle_t_n3_xzy(p), swizzle_t_n3_zxy(p))
		r = h_length_t_v(swizzle_t_n3_xy(p))
		pp = h_t_f3_t_n_t_n_t_n(h_sub_t_f_f(h_log_t_n(r), h_mul_f_f(gameTime, h_add_f_f(0.10000000000000001, h_mul_f_f(h_add_f_f(g, 1.0), 0.050999999999999997)))), h_atan2_t_n_t_n(swizzle_t_n3_x(p), swizzle_t_n3_y(p)), h_div_t_f_t_f(swizzle_t_n3_z(p), r))
		e = h_dot_t_v_v(swizzle_t_n3_xy(pp), n)
		f = h_dot_t_v_v(swizzle_t_n3_xy(pp), h_f2_n_n(swizzle_n2_y(n), h_sub_f(swizzle_n2_x(n))))
		if True:
			k = 1.2020999999999999
			e = h_sub_t_f_f(h_sub_t_f_t_f(e, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(e, k)), k)), h_mul_f_f(k, 0.5))
		l = 0.65000000000000002
		f = h_add_t_f_f(f, 1.3)
		i = h_sub_t_f_t_f(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.0), h_mul_t_f_f(h_floor_t_n(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.00)), 3.0))
		f = h_sub_t_f_f(h_sub_t_f_t_f(f, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), l)), h_mul_f_f(l, 0.5))
		d = h_mul_t_f_t_f(h_sub_t_f_t_f(h_length_t_v(h_t_f2_t_n_t_n(e, swizzle_t_n3_z(pp))), h_div_f_t_f(0.014999999999999999, r)), r)
		j = h_equal_t_n_n(i, 0.0)
		dd = opSmoothUnion_f_f_f_arr(dd, d, 0.10000000000000001)
		ff = h_mul_t_f_t_f(h_mul_t_f_f(flower_f3_f_f_arr(h_div_t_vf_f(h_t_f3_t_n_t_n_t_n(e, f, h_add_t_f_f(swizzle_t_n3_z(pp), 0.059999999999999998)), 0.25), h_mul_t_f_t_f(h_smoothstep_n_n_t_n(-1.0, 1.0, h_mul_t_f_t_f(r, r)), h_where_t_n_n_n(j, 5.0, 2.0)), h_smoothstep_n_n_t_n(1.0, -0.0, h_mul_t_f_t_f(r, r))), 0.25), r)
		ee = h_min_n_t_n(ee, ff)
		_vecif_722_exp = h_equal_t_n_t_n(ee, ff)
		if any_ifexp_true_t_n(_vecif_722_exp):
			_vecif_722_m = m
			_vecif_722_m = h_where_t_n_n_n(j, 1, 0)
			m = h_where_t_n_t_n_t_n(_vecif_722_exp, _vecif_722_m, m)
		g = 1.0
		p = h_where_n_t_v_t_v((b := h_not_n(b)), swizzle_t_n3_xzy(p), swizzle_t_n3_zxy(p))
		r = h_length_t_v(swizzle_t_n3_xy(p))
		pp = h_t_f3_t_n_t_n_t_n(h_sub_t_f_f(h_log_t_n(r), h_mul_f_f(gameTime, 0.202)), h_atan2_t_n_t_n(swizzle_t_n3_x(p), swizzle_t_n3_y(p)), h_div_t_f_t_f(swizzle_t_n3_z(p), r))
		e = h_dot_t_v_v(swizzle_t_n3_xy(pp), n)
		f = h_dot_t_v_v(swizzle_t_n3_xy(pp), h_f2_n_n(swizzle_n2_y(n), h_sub_f(swizzle_n2_x(n))))
		if True:
			k = 1.2020999999999999
			e = h_sub_t_f_f(h_sub_t_f_t_f(e, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(e, k)), k)), h_mul_f_f(k, 0.5))
		l = 0.65000000000000002
		f = h_add_t_f_f(f, 1.3)
		i = h_sub_t_f_t_f(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 1.0), h_mul_t_f_f(h_floor_t_n(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.3333333)), 3.0))
		f = h_sub_t_f_f(h_sub_t_f_t_f(f, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), l)), h_mul_f_f(l, 0.5))
		d = h_mul_t_f_t_f(h_sub_t_f_t_f(h_length_t_v(h_t_f2_t_n_t_n(e, swizzle_t_n3_z(pp))), h_div_f_t_f(0.014999999999999999, r)), r)
		j = h_equal_t_n_n(i, 0.0)
		dd = opSmoothUnion_f_f_f_arr1(dd, d, 0.10000000000000001)
		ff = h_mul_t_f_t_f(h_mul_t_f_f(flower_f3_f_f_arr(h_div_t_vf_f(h_t_f3_t_n_t_n_t_n(e, f, h_add_t_f_f(swizzle_t_n3_z(pp), 0.059999999999999998)), 0.25), h_mul_t_f_t_f(h_smoothstep_n_n_t_n(-1.0, 1.0, h_mul_t_f_t_f(r, r)), h_where_t_n_n_n(j, 5.0, 2.0)), h_smoothstep_n_n_t_n(1.0, -0.0, h_mul_t_f_t_f(r, r))), 0.25), r)
		ee = h_min_t_n_t_n(ee, ff)
		_vecif_723_exp = h_equal_t_n_t_n(ee, ff)
		if any_ifexp_true_t_n(_vecif_723_exp):
			_vecif_723_m = m
			_vecif_723_m = h_where_t_n_n_n(j, 1, 0)
			m = h_where_t_n_t_n_t_n(_vecif_723_exp, _vecif_723_m, m)
		g = 2.0
		p = h_where_n_t_v_t_v((b := h_not_n(b)), swizzle_t_n3_xzy(p), swizzle_t_n3_zxy(p))
		r = h_length_t_v(swizzle_t_n3_xy(p))
		pp = h_t_f3_t_n_t_n_t_n(h_sub_t_f_f(h_log_t_n(r), h_mul_f_f(gameTime, 0.253)), h_atan2_t_n_t_n(swizzle_t_n3_x(p), swizzle_t_n3_y(p)), h_div_t_f_t_f(swizzle_t_n3_z(p), r))
		e = h_dot_t_v_v(swizzle_t_n3_xy(pp), n)
		f = h_dot_t_v_v(swizzle_t_n3_xy(pp), h_f2_n_n(swizzle_n2_y(n), h_sub_f(swizzle_n2_x(n))))
		if True:
			k = 1.2020999999999999
			e = h_sub_t_f_f(h_sub_t_f_t_f(e, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(e, k)), k)), h_mul_f_f(k, 0.5))
		l = 0.65000000000000002
		f = h_add_t_f_f(f, 1.3)
		i = h_sub_t_f_t_f(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 2.0), h_mul_t_f_f(h_floor_t_n(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.6666667)), 3.0))
		f = h_sub_t_f_f(h_sub_t_f_t_f(f, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), l)), h_mul_f_f(l, 0.5))
		d = h_mul_t_f_t_f(h_sub_t_f_t_f(h_length_t_v(h_t_f2_t_n_t_n(e, swizzle_t_n3_z(pp))), h_div_f_t_f(0.014999999999999999, r)), r)
		j = h_equal_t_n_n(i, 0.0)
		dd = opSmoothUnion_f_f_f_arr1(dd, d, 0.10000000000000001)
		ff = h_mul_t_f_t_f(h_mul_t_f_f(flower_f3_f_f_arr(h_div_t_vf_f(h_t_f3_t_n_t_n_t_n(e, f, h_add_t_f_f(swizzle_t_n3_z(pp), 0.059999999999999998)), 0.25), h_mul_t_f_t_f(h_smoothstep_n_n_t_n(-1.0, 1.0, h_mul_t_f_t_f(r, r)), h_where_t_n_n_n(j, 5.0, 2.0)), h_smoothstep_n_n_t_n(1.0, -0.0, h_mul_t_f_t_f(r, r))), 0.25), r)
		ee = h_min_t_n_t_n(ee, ff)
		_vecif_724_exp = h_equal_t_n_t_n(ee, ff)
		if any_ifexp_true_t_n(_vecif_724_exp):
			_vecif_724_m = m
			_vecif_724_m = h_where_t_n_n_n(j, 1, 0)
			m = h_where_t_n_t_n_t_n(_vecif_724_exp, _vecif_724_m, m)
	ff = h_min_t_n_t_n(dd, ee)
	_vecif_725_exp = h_equal_t_n_t_n(ff, dd)
	if any_ifexp_true_t_n(_vecif_725_exp):
		_vecif_725_m = m
		_vecif_725_m = h_broadcast_t_i_i(_vecif_725_m, 0)
		m = h_where_t_n_t_n_t_n(_vecif_725_exp, _vecif_725_m, m)
	return h_mul_t_f_f(ff, 0.80000000000000004), m

def df_f3_i_arr(_pp, m):
	global gameTime
	swizzle_set_t_n3_y(_pp, h_sub_t_f(swizzle_t_n3_y(_pp)))
	swizzle_set_t_n3_xz(_pp, h_matmul_f2x2_t_f2(rot_f(1.016), swizzle_t_n3_xz(_pp)))
	swizzle_set_t_n3_xy(_pp, h_matmul_f2x2_t_f2(rot_f(-0.64000000000000001), swizzle_t_n3_xy(_pp)))
	dd = 1.0E+10
	ee = 1.0E+10
	p = _pp
	fsz = 0.25
	n = h_f2_n_n(h_cos_n(0.3926991), h_sin_n(0.3926991))
	b = False
	if True:
		g = 0.0
		p = h_where_n_t_v_t_v((b := h_not_n(b)), swizzle_t_n3_xzy(p), swizzle_t_n3_zxy(p))
		r = h_length_t_v(swizzle_t_n3_xy(p))
		pp = h_t_f3_t_n_t_n_t_n(h_sub_t_f_f(h_log_t_n(r), h_mul_f_f(gameTime, h_add_f_f(0.10000000000000001, h_mul_f_f(h_add_f_f(g, 1.0), 0.050999999999999997)))), h_atan2_t_n_t_n(swizzle_t_n3_x(p), swizzle_t_n3_y(p)), h_div_t_f_t_f(swizzle_t_n3_z(p), r))
		e = h_dot_t_v_v(swizzle_t_n3_xy(pp), n)
		f = h_dot_t_v_v(swizzle_t_n3_xy(pp), h_f2_n_n(swizzle_n2_y(n), h_sub_f(swizzle_n2_x(n))))
		if True:
			k = 1.2020999999999999
			e = h_sub_t_f_f(h_sub_t_f_t_f(e, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(e, k)), k)), h_mul_f_f(k, 0.5))
		l = 0.65000000000000002
		f = h_add_t_f_f(f, 1.3)
		i = h_sub_t_f_t_f(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.0), h_mul_t_f_f(h_floor_t_n(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.00)), 3.0))
		f = h_sub_t_f_f(h_sub_t_f_t_f(f, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), l)), h_mul_f_f(l, 0.5))
		d = h_mul_t_f_t_f(h_sub_t_f_t_f(h_length_t_v(h_t_f2_t_n_t_n(e, swizzle_t_n3_z(pp))), h_div_f_t_f(0.014999999999999999, r)), r)
		j = h_equal_t_n_n(i, 0.0)
		dd = opSmoothUnion_f_f_f_arr(dd, d, 0.10000000000000001)
		ff = h_mul_t_f_t_f(h_mul_t_f_f(flower_f3_f_f_arr(h_div_t_vf_f(h_t_f3_t_n_t_n_t_n(e, f, h_add_t_f_f(swizzle_t_n3_z(pp), 0.059999999999999998)), 0.25), h_mul_t_f_t_f(h_smoothstep_n_n_t_n(-1.0, 1.0, h_mul_t_f_t_f(r, r)), h_where_t_n_n_n(j, 5.0, 2.0)), h_smoothstep_n_n_t_n(1.0, -0.0, h_mul_t_f_t_f(r, r))), 0.25), r)
		ee = h_min_n_t_n(ee, ff)
		_vecif_718_exp = h_equal_t_n_t_n(ee, ff)
		if any_ifexp_true_t_n(_vecif_718_exp):
			_vecif_718_m = m
			_vecif_718_m = h_where_t_n_n_n(j, 1, 0)
			m = h_where_t_n_t_n_n(_vecif_718_exp, _vecif_718_m, m)
		else:
			m = h_broadcast_t_i_i(_pp, m)
		g = 1.0
		p = h_where_n_t_v_t_v((b := h_not_n(b)), swizzle_t_n3_xzy(p), swizzle_t_n3_zxy(p))
		r = h_length_t_v(swizzle_t_n3_xy(p))
		pp = h_t_f3_t_n_t_n_t_n(h_sub_t_f_f(h_log_t_n(r), h_mul_f_f(gameTime, 0.202)), h_atan2_t_n_t_n(swizzle_t_n3_x(p), swizzle_t_n3_y(p)), h_div_t_f_t_f(swizzle_t_n3_z(p), r))
		e = h_dot_t_v_v(swizzle_t_n3_xy(pp), n)
		f = h_dot_t_v_v(swizzle_t_n3_xy(pp), h_f2_n_n(swizzle_n2_y(n), h_sub_f(swizzle_n2_x(n))))
		if True:
			k = 1.2020999999999999
			e = h_sub_t_f_f(h_sub_t_f_t_f(e, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(e, k)), k)), h_mul_f_f(k, 0.5))
		l = 0.65000000000000002
		f = h_add_t_f_f(f, 1.3)
		i = h_sub_t_f_t_f(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 1.0), h_mul_t_f_f(h_floor_t_n(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.3333333)), 3.0))
		f = h_sub_t_f_f(h_sub_t_f_t_f(f, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), l)), h_mul_f_f(l, 0.5))
		d = h_mul_t_f_t_f(h_sub_t_f_t_f(h_length_t_v(h_t_f2_t_n_t_n(e, swizzle_t_n3_z(pp))), h_div_f_t_f(0.014999999999999999, r)), r)
		j = h_equal_t_n_n(i, 0.0)
		dd = opSmoothUnion_f_f_f_arr1(dd, d, 0.10000000000000001)
		ff = h_mul_t_f_t_f(h_mul_t_f_f(flower_f3_f_f_arr(h_div_t_vf_f(h_t_f3_t_n_t_n_t_n(e, f, h_add_t_f_f(swizzle_t_n3_z(pp), 0.059999999999999998)), 0.25), h_mul_t_f_t_f(h_smoothstep_n_n_t_n(-1.0, 1.0, h_mul_t_f_t_f(r, r)), h_where_t_n_n_n(j, 5.0, 2.0)), h_smoothstep_n_n_t_n(1.0, -0.0, h_mul_t_f_t_f(r, r))), 0.25), r)
		ee = h_min_t_n_t_n(ee, ff)
		_vecif_719_exp = h_equal_t_n_t_n(ee, ff)
		if any_ifexp_true_t_n(_vecif_719_exp):
			_vecif_719_m = m
			_vecif_719_m = h_where_t_n_n_n(j, 1, 0)
			m = h_where_t_n_t_n_t_n(_vecif_719_exp, _vecif_719_m, m)
		g = 2.0
		p = h_where_n_t_v_t_v((b := h_not_n(b)), swizzle_t_n3_xzy(p), swizzle_t_n3_zxy(p))
		r = h_length_t_v(swizzle_t_n3_xy(p))
		pp = h_t_f3_t_n_t_n_t_n(h_sub_t_f_f(h_log_t_n(r), h_mul_f_f(gameTime, 0.253)), h_atan2_t_n_t_n(swizzle_t_n3_x(p), swizzle_t_n3_y(p)), h_div_t_f_t_f(swizzle_t_n3_z(p), r))
		e = h_dot_t_v_v(swizzle_t_n3_xy(pp), n)
		f = h_dot_t_v_v(swizzle_t_n3_xy(pp), h_f2_n_n(swizzle_n2_y(n), h_sub_f(swizzle_n2_x(n))))
		if True:
			k = 1.2020999999999999
			e = h_sub_t_f_f(h_sub_t_f_t_f(e, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(e, k)), k)), h_mul_f_f(k, 0.5))
		l = 0.65000000000000002
		f = h_add_t_f_f(f, 1.3)
		i = h_sub_t_f_t_f(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 2.0), h_mul_t_f_f(h_floor_t_n(h_add_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), 0.6666667)), 3.0))
		f = h_sub_t_f_f(h_sub_t_f_t_f(f, h_mul_t_f_f(h_floor_t_n(h_div_t_f_f(f, l)), l)), h_mul_f_f(l, 0.5))
		d = h_mul_t_f_t_f(h_sub_t_f_t_f(h_length_t_v(h_t_f2_t_n_t_n(e, swizzle_t_n3_z(pp))), h_div_f_t_f(0.014999999999999999, r)), r)
		j = h_equal_t_n_n(i, 0.0)
		dd = opSmoothUnion_f_f_f_arr1(dd, d, 0.10000000000000001)
		ff = h_mul_t_f_t_f(h_mul_t_f_f(flower_f3_f_f_arr(h_div_t_vf_f(h_t_f3_t_n_t_n_t_n(e, f, h_add_t_f_f(swizzle_t_n3_z(pp), 0.059999999999999998)), 0.25), h_mul_t_f_t_f(h_smoothstep_n_n_t_n(-1.0, 1.0, h_mul_t_f_t_f(r, r)), h_where_t_n_n_n(j, 5.0, 2.0)), h_smoothstep_n_n_t_n(1.0, -0.0, h_mul_t_f_t_f(r, r))), 0.25), r)
		ee = h_min_t_n_t_n(ee, ff)
		_vecif_720_exp = h_equal_t_n_t_n(ee, ff)
		if any_ifexp_true_t_n(_vecif_720_exp):
			_vecif_720_m = m
			_vecif_720_m = h_where_t_n_n_n(j, 1, 0)
			m = h_where_t_n_t_n_t_n(_vecif_720_exp, _vecif_720_m, m)
	ff = h_min_t_n_t_n(dd, ee)
	_vecif_721_exp = h_equal_t_n_t_n(ff, dd)
	if any_ifexp_true_t_n(_vecif_721_exp):
		_vecif_721_m = m
		_vecif_721_m = h_broadcast_t_i_i(_vecif_721_m, 0)
		m = h_where_t_n_t_n_t_n(_vecif_721_exp, _vecif_721_m, m)
	return h_mul_t_f_f(ff, 0.80000000000000004), m

def glsl_rmRes_ctor_f3_i_b_arr(_p, _i, _h):
	__stru_tmp = h_cast_t_rmRes_tp3_t_f3_i_b([_p, _i, _h])
	return __stru_tmp

def normal_f3_i_arr(p, m):
	d = tuple_get_retval((_call_ret_714 := df_f3_i_arr1(h_copy_t_f3(p), m), m := tuple_get_outparam(_call_ret_714, 1)))
	u = h_f2_n_n(0.0, 2.0000000000000001E-4)
	return h_normalize_t_v(h_sub_t_vf_t_f(h_t_f3_t_n_t_n_t_n(tuple_get_retval((_call_ret_715 := df_f3_i_arr1(h_add_t_vf_vf(p, swizzle_n2_yxx(u)), m), m := tuple_get_outparam(_call_ret_715, 1))), tuple_get_retval((_call_ret_716 := df_f3_i_arr1(h_add_t_vf_vf(p, swizzle_n2_xyx(u)), m), m := tuple_get_outparam(_call_ret_716, 1))), tuple_get_retval((_call_ret_717 := df_f3_i_arr1(h_add_t_vf_vf(p, swizzle_n2_xxy(u)), m), m := tuple_get_outparam(_call_ret_717, 1)))), d)), m

def rm_f3_f3_i_arr(c, r, m):
	s = glsl_rmRes_ctor_f3_i_b_arr(h_add_vf_t_vf(c, h_mul_t_vf_f(r, 0.0)), 0, False)
	d = 0.0
	i = 0
	_br_flag_188 = False
	if True:
		_vecif_266_exp = h_less_than_n_n(0, 16)
		if any_ifexp_true_n(_vecif_266_exp):
			_vecif_266_s = h_copy_t_rmRes(s)
			_vecif_266_d = d
			_vecif_266_m = m
			_vecif_266__br_flag_188 = _br_flag_188
			_vecif_267_exp_0 = h_not_n(_vecif_266__br_flag_188)
			if any_ifexp_true_n(_vecif_267_exp_0):
				_vecif_267_s = h_copy_t_rmRes(_vecif_266_s)
				_vecif_267_d = _vecif_266_d
				_vecif_267_m = _vecif_266_m
				_vecif_267__br_flag_188 = _vecif_266__br_flag_188
				_vecif_267_d = tuple_get_retval((_call_ret_268 := df_f3_i_arr(h_copy_t_f3(rmRes_p(_vecif_267_s)), _vecif_267_m), _vecif_267_m := tuple_get_outparam(_call_ret_268, 1)))
				_vecif_269_exp = h_less_than_t_n_n(_vecif_267_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_269_exp):
					_vecif_269_s = h_copy_t_rmRes(_vecif_267_s)
					_vecif_269__br_flag_188 = _vecif_267__br_flag_188
					rmRes_h_set(_vecif_269_s, h_broadcast_t_b_b(rmRes_h(_vecif_269_s), True))
					_vecif_269__br_flag_188 = True
					_vecif_267_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_269_exp, _vecif_269_s, _vecif_267_s)
					_vecif_267__br_flag_188 = h_where_t_n_n_n(_vecif_269_exp, _vecif_269__br_flag_188, _vecif_267__br_flag_188)
				else:
					_vecif_267__br_flag_188 = h_broadcast_t_b_b(r, _vecif_267__br_flag_188)
				_vecif_270_exp = h_not_t_n(_vecif_267__br_flag_188)
				if any_ifexp_true_t_n(_vecif_270_exp):
					_vecif_270_s = h_copy_t_rmRes(_vecif_267_s)
					_vecif_270__br_flag_188 = _vecif_267__br_flag_188
					_vecif_271_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_270_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_271_exp):
						_vecif_271__br_flag_188 = _vecif_270__br_flag_188
						_vecif_271__br_flag_188 = h_broadcast_t_b_b(_vecif_271__br_flag_188, True)
						_vecif_270__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_271_exp, _vecif_271__br_flag_188, _vecif_270__br_flag_188)
					_vecif_272_exp = h_not_t_n(_vecif_270__br_flag_188)
					if any_ifexp_true_t_n(_vecif_272_exp):
						_vecif_272_s = h_copy_t_rmRes(_vecif_270_s)
						rmRes_p_set(_vecif_272_s, h_add_t_vf_t_vf(rmRes_p(_vecif_272_s), h_mul_t_f_t_vf(_vecif_267_d, r)))
						_vecif_270_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_272_exp, _vecif_272_s, _vecif_270_s)
					_vecif_267_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_270_exp, _vecif_270_s, _vecif_267_s)
					_vecif_267__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_270_exp, _vecif_270__br_flag_188, _vecif_267__br_flag_188)
				_vecif_266_s = h_where_n_t_rmRes_t_rmRes(_vecif_267_exp_0, _vecif_267_s, _vecif_266_s)
				_vecif_266_d = h_where_n_t_n_n(_vecif_267_exp_0, _vecif_267_d, _vecif_266_d)
				_vecif_266_m = h_where_n_t_n_n(_vecif_267_exp_0, _vecif_267_m, _vecif_266_m)
				_vecif_266__br_flag_188 = h_where_n_t_n_n(_vecif_267_exp_0, _vecif_267__br_flag_188, _vecif_266__br_flag_188)
			else:
				_vecif_266_d = h_broadcast_t_f_f(r, _vecif_266_d)
				_vecif_266_m = h_broadcast_t_i_i(r, _vecif_266_m)
				_vecif_266__br_flag_188 = h_broadcast_t_b_b(r, _vecif_266__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_266_exp, _vecif_266_s, s)
			d = h_where_n_t_n_n(_vecif_266_exp, _vecif_266_d, d)
			m = h_where_n_t_n_n(_vecif_266_exp, _vecif_266_m, m)
			_br_flag_188 = h_where_n_t_n_n(_vecif_266_exp, _vecif_266__br_flag_188, _br_flag_188)
		else:
			d = h_broadcast_t_f_f(r, d)
			m = h_broadcast_t_i_i(r, m)
			_br_flag_188 = h_broadcast_t_b_b(r, _br_flag_188)
		i = h_inc_i(i)
		_vecif_273_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_273_exp):
			_vecif_273_s = h_copy_t_rmRes(s)
			_vecif_273_d = d
			_vecif_273_m = m
			_vecif_273__br_flag_188 = _br_flag_188
			_vecif_274_exp_0 = h_not_t_n(_vecif_273__br_flag_188)
			if any_ifexp_true_t_n(_vecif_274_exp_0):
				_vecif_274_s = h_copy_t_rmRes(_vecif_273_s)
				_vecif_274_d = _vecif_273_d
				_vecif_274_m = _vecif_273_m
				_vecif_274__br_flag_188 = _vecif_273__br_flag_188
				_vecif_274_d = tuple_get_retval((_call_ret_275 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_274_s)), _vecif_274_m), _vecif_274_m := tuple_get_outparam(_call_ret_275, 1)))
				_vecif_276_exp = h_less_than_t_n_n(_vecif_274_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_276_exp):
					_vecif_276_s = h_copy_t_rmRes(_vecif_274_s)
					_vecif_276__br_flag_188 = _vecif_274__br_flag_188
					rmRes_h_set(_vecif_276_s, h_broadcast_t_b_b(rmRes_h(_vecif_276_s), True))
					_vecif_276__br_flag_188 = h_broadcast_t_b_b(_vecif_276__br_flag_188, True)
					_vecif_274_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_276_exp, _vecif_276_s, _vecif_274_s)
					_vecif_274__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_276_exp, _vecif_276__br_flag_188, _vecif_274__br_flag_188)
				_vecif_277_exp = h_not_t_n(_vecif_274__br_flag_188)
				if any_ifexp_true_t_n(_vecif_277_exp):
					_vecif_277_s = h_copy_t_rmRes(_vecif_274_s)
					_vecif_277__br_flag_188 = _vecif_274__br_flag_188
					_vecif_278_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_277_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_278_exp):
						_vecif_278__br_flag_188 = _vecif_277__br_flag_188
						_vecif_278__br_flag_188 = h_broadcast_t_b_b(_vecif_278__br_flag_188, True)
						_vecif_277__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_278_exp, _vecif_278__br_flag_188, _vecif_277__br_flag_188)
					_vecif_279_exp = h_not_t_n(_vecif_277__br_flag_188)
					if any_ifexp_true_t_n(_vecif_279_exp):
						_vecif_279_s = h_copy_t_rmRes(_vecif_277_s)
						rmRes_p_set(_vecif_279_s, h_add_t_vf_t_vf(rmRes_p(_vecif_279_s), h_mul_t_f_t_vf(_vecif_274_d, r)))
						_vecif_277_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_279_exp, _vecif_279_s, _vecif_277_s)
					_vecif_274_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_277_exp, _vecif_277_s, _vecif_274_s)
					_vecif_274__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_277_exp, _vecif_277__br_flag_188, _vecif_274__br_flag_188)
				_vecif_273_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_274_exp_0, _vecif_274_s, _vecif_273_s)
				_vecif_273_d = h_where_t_n_t_n_t_n(_vecif_274_exp_0, _vecif_274_d, _vecif_273_d)
				_vecif_273_m = h_where_t_n_t_n_t_n(_vecif_274_exp_0, _vecif_274_m, _vecif_273_m)
				_vecif_273__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_274_exp_0, _vecif_274__br_flag_188, _vecif_273__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_273_exp, _vecif_273_s, s)
			d = h_where_n_t_n_t_n(_vecif_273_exp, _vecif_273_d, d)
			m = h_where_n_t_n_t_n(_vecif_273_exp, _vecif_273_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_273_exp, _vecif_273__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_280_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_280_exp):
			_vecif_280_s = h_copy_t_rmRes(s)
			_vecif_280_d = d
			_vecif_280_m = m
			_vecif_280__br_flag_188 = _br_flag_188
			_vecif_281_exp_0 = h_not_t_n(_vecif_280__br_flag_188)
			if any_ifexp_true_t_n(_vecif_281_exp_0):
				_vecif_281_s = h_copy_t_rmRes(_vecif_280_s)
				_vecif_281_d = _vecif_280_d
				_vecif_281_m = _vecif_280_m
				_vecif_281__br_flag_188 = _vecif_280__br_flag_188
				_vecif_281_d = tuple_get_retval((_call_ret_282 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_281_s)), _vecif_281_m), _vecif_281_m := tuple_get_outparam(_call_ret_282, 1)))
				_vecif_283_exp = h_less_than_t_n_n(_vecif_281_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_283_exp):
					_vecif_283_s = h_copy_t_rmRes(_vecif_281_s)
					_vecif_283__br_flag_188 = _vecif_281__br_flag_188
					rmRes_h_set(_vecif_283_s, h_broadcast_t_b_b(rmRes_h(_vecif_283_s), True))
					_vecif_283__br_flag_188 = h_broadcast_t_b_b(_vecif_283__br_flag_188, True)
					_vecif_281_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_283_exp, _vecif_283_s, _vecif_281_s)
					_vecif_281__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_283_exp, _vecif_283__br_flag_188, _vecif_281__br_flag_188)
				_vecif_284_exp = h_not_t_n(_vecif_281__br_flag_188)
				if any_ifexp_true_t_n(_vecif_284_exp):
					_vecif_284_s = h_copy_t_rmRes(_vecif_281_s)
					_vecif_284__br_flag_188 = _vecif_281__br_flag_188
					_vecif_285_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_284_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_285_exp):
						_vecif_285__br_flag_188 = _vecif_284__br_flag_188
						_vecif_285__br_flag_188 = h_broadcast_t_b_b(_vecif_285__br_flag_188, True)
						_vecif_284__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_285_exp, _vecif_285__br_flag_188, _vecif_284__br_flag_188)
					_vecif_286_exp = h_not_t_n(_vecif_284__br_flag_188)
					if any_ifexp_true_t_n(_vecif_286_exp):
						_vecif_286_s = h_copy_t_rmRes(_vecif_284_s)
						rmRes_p_set(_vecif_286_s, h_add_t_vf_t_vf(rmRes_p(_vecif_286_s), h_mul_t_f_t_vf(_vecif_281_d, r)))
						_vecif_284_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_286_exp, _vecif_286_s, _vecif_284_s)
					_vecif_281_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_284_exp, _vecif_284_s, _vecif_281_s)
					_vecif_281__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_284_exp, _vecif_284__br_flag_188, _vecif_281__br_flag_188)
				_vecif_280_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_281_exp_0, _vecif_281_s, _vecif_280_s)
				_vecif_280_d = h_where_t_n_t_n_t_n(_vecif_281_exp_0, _vecif_281_d, _vecif_280_d)
				_vecif_280_m = h_where_t_n_t_n_t_n(_vecif_281_exp_0, _vecif_281_m, _vecif_280_m)
				_vecif_280__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_281_exp_0, _vecif_281__br_flag_188, _vecif_280__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_280_exp, _vecif_280_s, s)
			d = h_where_n_t_n_t_n(_vecif_280_exp, _vecif_280_d, d)
			m = h_where_n_t_n_t_n(_vecif_280_exp, _vecif_280_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_280_exp, _vecif_280__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_287_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_287_exp):
			_vecif_287_s = h_copy_t_rmRes(s)
			_vecif_287_d = d
			_vecif_287_m = m
			_vecif_287__br_flag_188 = _br_flag_188
			_vecif_288_exp_0 = h_not_t_n(_vecif_287__br_flag_188)
			if any_ifexp_true_t_n(_vecif_288_exp_0):
				_vecif_288_s = h_copy_t_rmRes(_vecif_287_s)
				_vecif_288_d = _vecif_287_d
				_vecif_288_m = _vecif_287_m
				_vecif_288__br_flag_188 = _vecif_287__br_flag_188
				_vecif_288_d = tuple_get_retval((_call_ret_289 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_288_s)), _vecif_288_m), _vecif_288_m := tuple_get_outparam(_call_ret_289, 1)))
				_vecif_290_exp = h_less_than_t_n_n(_vecif_288_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_290_exp):
					_vecif_290_s = h_copy_t_rmRes(_vecif_288_s)
					_vecif_290__br_flag_188 = _vecif_288__br_flag_188
					rmRes_h_set(_vecif_290_s, h_broadcast_t_b_b(rmRes_h(_vecif_290_s), True))
					_vecif_290__br_flag_188 = h_broadcast_t_b_b(_vecif_290__br_flag_188, True)
					_vecif_288_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_290_exp, _vecif_290_s, _vecif_288_s)
					_vecif_288__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_290_exp, _vecif_290__br_flag_188, _vecif_288__br_flag_188)
				_vecif_291_exp = h_not_t_n(_vecif_288__br_flag_188)
				if any_ifexp_true_t_n(_vecif_291_exp):
					_vecif_291_s = h_copy_t_rmRes(_vecif_288_s)
					_vecif_291__br_flag_188 = _vecif_288__br_flag_188
					_vecif_292_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_291_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_292_exp):
						_vecif_292__br_flag_188 = _vecif_291__br_flag_188
						_vecif_292__br_flag_188 = h_broadcast_t_b_b(_vecif_292__br_flag_188, True)
						_vecif_291__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_292_exp, _vecif_292__br_flag_188, _vecif_291__br_flag_188)
					_vecif_293_exp = h_not_t_n(_vecif_291__br_flag_188)
					if any_ifexp_true_t_n(_vecif_293_exp):
						_vecif_293_s = h_copy_t_rmRes(_vecif_291_s)
						rmRes_p_set(_vecif_293_s, h_add_t_vf_t_vf(rmRes_p(_vecif_293_s), h_mul_t_f_t_vf(_vecif_288_d, r)))
						_vecif_291_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_293_exp, _vecif_293_s, _vecif_291_s)
					_vecif_288_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_291_exp, _vecif_291_s, _vecif_288_s)
					_vecif_288__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_291_exp, _vecif_291__br_flag_188, _vecif_288__br_flag_188)
				_vecif_287_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_288_exp_0, _vecif_288_s, _vecif_287_s)
				_vecif_287_d = h_where_t_n_t_n_t_n(_vecif_288_exp_0, _vecif_288_d, _vecif_287_d)
				_vecif_287_m = h_where_t_n_t_n_t_n(_vecif_288_exp_0, _vecif_288_m, _vecif_287_m)
				_vecif_287__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_288_exp_0, _vecif_288__br_flag_188, _vecif_287__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_287_exp, _vecif_287_s, s)
			d = h_where_n_t_n_t_n(_vecif_287_exp, _vecif_287_d, d)
			m = h_where_n_t_n_t_n(_vecif_287_exp, _vecif_287_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_287_exp, _vecif_287__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_294_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_294_exp):
			_vecif_294_s = h_copy_t_rmRes(s)
			_vecif_294_d = d
			_vecif_294_m = m
			_vecif_294__br_flag_188 = _br_flag_188
			_vecif_295_exp_0 = h_not_t_n(_vecif_294__br_flag_188)
			if any_ifexp_true_t_n(_vecif_295_exp_0):
				_vecif_295_s = h_copy_t_rmRes(_vecif_294_s)
				_vecif_295_d = _vecif_294_d
				_vecif_295_m = _vecif_294_m
				_vecif_295__br_flag_188 = _vecif_294__br_flag_188
				_vecif_295_d = tuple_get_retval((_call_ret_296 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_295_s)), _vecif_295_m), _vecif_295_m := tuple_get_outparam(_call_ret_296, 1)))
				_vecif_297_exp = h_less_than_t_n_n(_vecif_295_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_297_exp):
					_vecif_297_s = h_copy_t_rmRes(_vecif_295_s)
					_vecif_297__br_flag_188 = _vecif_295__br_flag_188
					rmRes_h_set(_vecif_297_s, h_broadcast_t_b_b(rmRes_h(_vecif_297_s), True))
					_vecif_297__br_flag_188 = h_broadcast_t_b_b(_vecif_297__br_flag_188, True)
					_vecif_295_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_297_exp, _vecif_297_s, _vecif_295_s)
					_vecif_295__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_297_exp, _vecif_297__br_flag_188, _vecif_295__br_flag_188)
				_vecif_298_exp = h_not_t_n(_vecif_295__br_flag_188)
				if any_ifexp_true_t_n(_vecif_298_exp):
					_vecif_298_s = h_copy_t_rmRes(_vecif_295_s)
					_vecif_298__br_flag_188 = _vecif_295__br_flag_188
					_vecif_299_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_298_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_299_exp):
						_vecif_299__br_flag_188 = _vecif_298__br_flag_188
						_vecif_299__br_flag_188 = h_broadcast_t_b_b(_vecif_299__br_flag_188, True)
						_vecif_298__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_299_exp, _vecif_299__br_flag_188, _vecif_298__br_flag_188)
					_vecif_300_exp = h_not_t_n(_vecif_298__br_flag_188)
					if any_ifexp_true_t_n(_vecif_300_exp):
						_vecif_300_s = h_copy_t_rmRes(_vecif_298_s)
						rmRes_p_set(_vecif_300_s, h_add_t_vf_t_vf(rmRes_p(_vecif_300_s), h_mul_t_f_t_vf(_vecif_295_d, r)))
						_vecif_298_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_300_exp, _vecif_300_s, _vecif_298_s)
					_vecif_295_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_298_exp, _vecif_298_s, _vecif_295_s)
					_vecif_295__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_298_exp, _vecif_298__br_flag_188, _vecif_295__br_flag_188)
				_vecif_294_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_295_exp_0, _vecif_295_s, _vecif_294_s)
				_vecif_294_d = h_where_t_n_t_n_t_n(_vecif_295_exp_0, _vecif_295_d, _vecif_294_d)
				_vecif_294_m = h_where_t_n_t_n_t_n(_vecif_295_exp_0, _vecif_295_m, _vecif_294_m)
				_vecif_294__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_295_exp_0, _vecif_295__br_flag_188, _vecif_294__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_294_exp, _vecif_294_s, s)
			d = h_where_n_t_n_t_n(_vecif_294_exp, _vecif_294_d, d)
			m = h_where_n_t_n_t_n(_vecif_294_exp, _vecif_294_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_294_exp, _vecif_294__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_301_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_301_exp):
			_vecif_301_s = h_copy_t_rmRes(s)
			_vecif_301_d = d
			_vecif_301_m = m
			_vecif_301__br_flag_188 = _br_flag_188
			_vecif_302_exp_0 = h_not_t_n(_vecif_301__br_flag_188)
			if any_ifexp_true_t_n(_vecif_302_exp_0):
				_vecif_302_s = h_copy_t_rmRes(_vecif_301_s)
				_vecif_302_d = _vecif_301_d
				_vecif_302_m = _vecif_301_m
				_vecif_302__br_flag_188 = _vecif_301__br_flag_188
				_vecif_302_d = tuple_get_retval((_call_ret_303 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_302_s)), _vecif_302_m), _vecif_302_m := tuple_get_outparam(_call_ret_303, 1)))
				_vecif_304_exp = h_less_than_t_n_n(_vecif_302_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_304_exp):
					_vecif_304_s = h_copy_t_rmRes(_vecif_302_s)
					_vecif_304__br_flag_188 = _vecif_302__br_flag_188
					rmRes_h_set(_vecif_304_s, h_broadcast_t_b_b(rmRes_h(_vecif_304_s), True))
					_vecif_304__br_flag_188 = h_broadcast_t_b_b(_vecif_304__br_flag_188, True)
					_vecif_302_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_304_exp, _vecif_304_s, _vecif_302_s)
					_vecif_302__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_304_exp, _vecif_304__br_flag_188, _vecif_302__br_flag_188)
				_vecif_305_exp = h_not_t_n(_vecif_302__br_flag_188)
				if any_ifexp_true_t_n(_vecif_305_exp):
					_vecif_305_s = h_copy_t_rmRes(_vecif_302_s)
					_vecif_305__br_flag_188 = _vecif_302__br_flag_188
					_vecif_306_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_305_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_306_exp):
						_vecif_306__br_flag_188 = _vecif_305__br_flag_188
						_vecif_306__br_flag_188 = h_broadcast_t_b_b(_vecif_306__br_flag_188, True)
						_vecif_305__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_306_exp, _vecif_306__br_flag_188, _vecif_305__br_flag_188)
					_vecif_307_exp = h_not_t_n(_vecif_305__br_flag_188)
					if any_ifexp_true_t_n(_vecif_307_exp):
						_vecif_307_s = h_copy_t_rmRes(_vecif_305_s)
						rmRes_p_set(_vecif_307_s, h_add_t_vf_t_vf(rmRes_p(_vecif_307_s), h_mul_t_f_t_vf(_vecif_302_d, r)))
						_vecif_305_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_307_exp, _vecif_307_s, _vecif_305_s)
					_vecif_302_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_305_exp, _vecif_305_s, _vecif_302_s)
					_vecif_302__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_305_exp, _vecif_305__br_flag_188, _vecif_302__br_flag_188)
				_vecif_301_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_302_exp_0, _vecif_302_s, _vecif_301_s)
				_vecif_301_d = h_where_t_n_t_n_t_n(_vecif_302_exp_0, _vecif_302_d, _vecif_301_d)
				_vecif_301_m = h_where_t_n_t_n_t_n(_vecif_302_exp_0, _vecif_302_m, _vecif_301_m)
				_vecif_301__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_302_exp_0, _vecif_302__br_flag_188, _vecif_301__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_301_exp, _vecif_301_s, s)
			d = h_where_n_t_n_t_n(_vecif_301_exp, _vecif_301_d, d)
			m = h_where_n_t_n_t_n(_vecif_301_exp, _vecif_301_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_301_exp, _vecif_301__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_308_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_308_exp):
			_vecif_308_s = h_copy_t_rmRes(s)
			_vecif_308_d = d
			_vecif_308_m = m
			_vecif_308__br_flag_188 = _br_flag_188
			_vecif_309_exp_0 = h_not_t_n(_vecif_308__br_flag_188)
			if any_ifexp_true_t_n(_vecif_309_exp_0):
				_vecif_309_s = h_copy_t_rmRes(_vecif_308_s)
				_vecif_309_d = _vecif_308_d
				_vecif_309_m = _vecif_308_m
				_vecif_309__br_flag_188 = _vecif_308__br_flag_188
				_vecif_309_d = tuple_get_retval((_call_ret_310 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_309_s)), _vecif_309_m), _vecif_309_m := tuple_get_outparam(_call_ret_310, 1)))
				_vecif_311_exp = h_less_than_t_n_n(_vecif_309_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_311_exp):
					_vecif_311_s = h_copy_t_rmRes(_vecif_309_s)
					_vecif_311__br_flag_188 = _vecif_309__br_flag_188
					rmRes_h_set(_vecif_311_s, h_broadcast_t_b_b(rmRes_h(_vecif_311_s), True))
					_vecif_311__br_flag_188 = h_broadcast_t_b_b(_vecif_311__br_flag_188, True)
					_vecif_309_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_311_exp, _vecif_311_s, _vecif_309_s)
					_vecif_309__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_311_exp, _vecif_311__br_flag_188, _vecif_309__br_flag_188)
				_vecif_312_exp = h_not_t_n(_vecif_309__br_flag_188)
				if any_ifexp_true_t_n(_vecif_312_exp):
					_vecif_312_s = h_copy_t_rmRes(_vecif_309_s)
					_vecif_312__br_flag_188 = _vecif_309__br_flag_188
					_vecif_313_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_312_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_313_exp):
						_vecif_313__br_flag_188 = _vecif_312__br_flag_188
						_vecif_313__br_flag_188 = h_broadcast_t_b_b(_vecif_313__br_flag_188, True)
						_vecif_312__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_313_exp, _vecif_313__br_flag_188, _vecif_312__br_flag_188)
					_vecif_314_exp = h_not_t_n(_vecif_312__br_flag_188)
					if any_ifexp_true_t_n(_vecif_314_exp):
						_vecif_314_s = h_copy_t_rmRes(_vecif_312_s)
						rmRes_p_set(_vecif_314_s, h_add_t_vf_t_vf(rmRes_p(_vecif_314_s), h_mul_t_f_t_vf(_vecif_309_d, r)))
						_vecif_312_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_314_exp, _vecif_314_s, _vecif_312_s)
					_vecif_309_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_312_exp, _vecif_312_s, _vecif_309_s)
					_vecif_309__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_312_exp, _vecif_312__br_flag_188, _vecif_309__br_flag_188)
				_vecif_308_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_309_exp_0, _vecif_309_s, _vecif_308_s)
				_vecif_308_d = h_where_t_n_t_n_t_n(_vecif_309_exp_0, _vecif_309_d, _vecif_308_d)
				_vecif_308_m = h_where_t_n_t_n_t_n(_vecif_309_exp_0, _vecif_309_m, _vecif_308_m)
				_vecif_308__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_309_exp_0, _vecif_309__br_flag_188, _vecif_308__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_308_exp, _vecif_308_s, s)
			d = h_where_n_t_n_t_n(_vecif_308_exp, _vecif_308_d, d)
			m = h_where_n_t_n_t_n(_vecif_308_exp, _vecif_308_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_308_exp, _vecif_308__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_315_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_315_exp):
			_vecif_315_s = h_copy_t_rmRes(s)
			_vecif_315_d = d
			_vecif_315_m = m
			_vecif_315__br_flag_188 = _br_flag_188
			_vecif_316_exp_0 = h_not_t_n(_vecif_315__br_flag_188)
			if any_ifexp_true_t_n(_vecif_316_exp_0):
				_vecif_316_s = h_copy_t_rmRes(_vecif_315_s)
				_vecif_316_d = _vecif_315_d
				_vecif_316_m = _vecif_315_m
				_vecif_316__br_flag_188 = _vecif_315__br_flag_188
				_vecif_316_d = tuple_get_retval((_call_ret_317 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_316_s)), _vecif_316_m), _vecif_316_m := tuple_get_outparam(_call_ret_317, 1)))
				_vecif_318_exp = h_less_than_t_n_n(_vecif_316_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_318_exp):
					_vecif_318_s = h_copy_t_rmRes(_vecif_316_s)
					_vecif_318__br_flag_188 = _vecif_316__br_flag_188
					rmRes_h_set(_vecif_318_s, h_broadcast_t_b_b(rmRes_h(_vecif_318_s), True))
					_vecif_318__br_flag_188 = h_broadcast_t_b_b(_vecif_318__br_flag_188, True)
					_vecif_316_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_318_exp, _vecif_318_s, _vecif_316_s)
					_vecif_316__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_318_exp, _vecif_318__br_flag_188, _vecif_316__br_flag_188)
				_vecif_319_exp = h_not_t_n(_vecif_316__br_flag_188)
				if any_ifexp_true_t_n(_vecif_319_exp):
					_vecif_319_s = h_copy_t_rmRes(_vecif_316_s)
					_vecif_319__br_flag_188 = _vecif_316__br_flag_188
					_vecif_320_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_319_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_320_exp):
						_vecif_320__br_flag_188 = _vecif_319__br_flag_188
						_vecif_320__br_flag_188 = h_broadcast_t_b_b(_vecif_320__br_flag_188, True)
						_vecif_319__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_320_exp, _vecif_320__br_flag_188, _vecif_319__br_flag_188)
					_vecif_321_exp = h_not_t_n(_vecif_319__br_flag_188)
					if any_ifexp_true_t_n(_vecif_321_exp):
						_vecif_321_s = h_copy_t_rmRes(_vecif_319_s)
						rmRes_p_set(_vecif_321_s, h_add_t_vf_t_vf(rmRes_p(_vecif_321_s), h_mul_t_f_t_vf(_vecif_316_d, r)))
						_vecif_319_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_321_exp, _vecif_321_s, _vecif_319_s)
					_vecif_316_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_319_exp, _vecif_319_s, _vecif_316_s)
					_vecif_316__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_319_exp, _vecif_319__br_flag_188, _vecif_316__br_flag_188)
				_vecif_315_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_316_exp_0, _vecif_316_s, _vecif_315_s)
				_vecif_315_d = h_where_t_n_t_n_t_n(_vecif_316_exp_0, _vecif_316_d, _vecif_315_d)
				_vecif_315_m = h_where_t_n_t_n_t_n(_vecif_316_exp_0, _vecif_316_m, _vecif_315_m)
				_vecif_315__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_316_exp_0, _vecif_316__br_flag_188, _vecif_315__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_315_exp, _vecif_315_s, s)
			d = h_where_n_t_n_t_n(_vecif_315_exp, _vecif_315_d, d)
			m = h_where_n_t_n_t_n(_vecif_315_exp, _vecif_315_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_315_exp, _vecif_315__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_322_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_322_exp):
			_vecif_322_s = h_copy_t_rmRes(s)
			_vecif_322_d = d
			_vecif_322_m = m
			_vecif_322__br_flag_188 = _br_flag_188
			_vecif_323_exp_0 = h_not_t_n(_vecif_322__br_flag_188)
			if any_ifexp_true_t_n(_vecif_323_exp_0):
				_vecif_323_s = h_copy_t_rmRes(_vecif_322_s)
				_vecif_323_d = _vecif_322_d
				_vecif_323_m = _vecif_322_m
				_vecif_323__br_flag_188 = _vecif_322__br_flag_188
				_vecif_323_d = tuple_get_retval((_call_ret_324 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_323_s)), _vecif_323_m), _vecif_323_m := tuple_get_outparam(_call_ret_324, 1)))
				_vecif_325_exp = h_less_than_t_n_n(_vecif_323_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_325_exp):
					_vecif_325_s = h_copy_t_rmRes(_vecif_323_s)
					_vecif_325__br_flag_188 = _vecif_323__br_flag_188
					rmRes_h_set(_vecif_325_s, h_broadcast_t_b_b(rmRes_h(_vecif_325_s), True))
					_vecif_325__br_flag_188 = h_broadcast_t_b_b(_vecif_325__br_flag_188, True)
					_vecif_323_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_325_exp, _vecif_325_s, _vecif_323_s)
					_vecif_323__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_325_exp, _vecif_325__br_flag_188, _vecif_323__br_flag_188)
				_vecif_326_exp = h_not_t_n(_vecif_323__br_flag_188)
				if any_ifexp_true_t_n(_vecif_326_exp):
					_vecif_326_s = h_copy_t_rmRes(_vecif_323_s)
					_vecif_326__br_flag_188 = _vecif_323__br_flag_188
					_vecif_327_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_326_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_327_exp):
						_vecif_327__br_flag_188 = _vecif_326__br_flag_188
						_vecif_327__br_flag_188 = h_broadcast_t_b_b(_vecif_327__br_flag_188, True)
						_vecif_326__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_327_exp, _vecif_327__br_flag_188, _vecif_326__br_flag_188)
					_vecif_328_exp = h_not_t_n(_vecif_326__br_flag_188)
					if any_ifexp_true_t_n(_vecif_328_exp):
						_vecif_328_s = h_copy_t_rmRes(_vecif_326_s)
						rmRes_p_set(_vecif_328_s, h_add_t_vf_t_vf(rmRes_p(_vecif_328_s), h_mul_t_f_t_vf(_vecif_323_d, r)))
						_vecif_326_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_328_exp, _vecif_328_s, _vecif_326_s)
					_vecif_323_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_326_exp, _vecif_326_s, _vecif_323_s)
					_vecif_323__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_326_exp, _vecif_326__br_flag_188, _vecif_323__br_flag_188)
				_vecif_322_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_323_exp_0, _vecif_323_s, _vecif_322_s)
				_vecif_322_d = h_where_t_n_t_n_t_n(_vecif_323_exp_0, _vecif_323_d, _vecif_322_d)
				_vecif_322_m = h_where_t_n_t_n_t_n(_vecif_323_exp_0, _vecif_323_m, _vecif_322_m)
				_vecif_322__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_323_exp_0, _vecif_323__br_flag_188, _vecif_322__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_322_exp, _vecif_322_s, s)
			d = h_where_n_t_n_t_n(_vecif_322_exp, _vecif_322_d, d)
			m = h_where_n_t_n_t_n(_vecif_322_exp, _vecif_322_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_322_exp, _vecif_322__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_329_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_329_exp):
			_vecif_329_s = h_copy_t_rmRes(s)
			_vecif_329_d = d
			_vecif_329_m = m
			_vecif_329__br_flag_188 = _br_flag_188
			_vecif_330_exp_0 = h_not_t_n(_vecif_329__br_flag_188)
			if any_ifexp_true_t_n(_vecif_330_exp_0):
				_vecif_330_s = h_copy_t_rmRes(_vecif_329_s)
				_vecif_330_d = _vecif_329_d
				_vecif_330_m = _vecif_329_m
				_vecif_330__br_flag_188 = _vecif_329__br_flag_188
				_vecif_330_d = tuple_get_retval((_call_ret_331 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_330_s)), _vecif_330_m), _vecif_330_m := tuple_get_outparam(_call_ret_331, 1)))
				_vecif_332_exp = h_less_than_t_n_n(_vecif_330_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_332_exp):
					_vecif_332_s = h_copy_t_rmRes(_vecif_330_s)
					_vecif_332__br_flag_188 = _vecif_330__br_flag_188
					rmRes_h_set(_vecif_332_s, h_broadcast_t_b_b(rmRes_h(_vecif_332_s), True))
					_vecif_332__br_flag_188 = h_broadcast_t_b_b(_vecif_332__br_flag_188, True)
					_vecif_330_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_332_exp, _vecif_332_s, _vecif_330_s)
					_vecif_330__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_332_exp, _vecif_332__br_flag_188, _vecif_330__br_flag_188)
				_vecif_333_exp = h_not_t_n(_vecif_330__br_flag_188)
				if any_ifexp_true_t_n(_vecif_333_exp):
					_vecif_333_s = h_copy_t_rmRes(_vecif_330_s)
					_vecif_333__br_flag_188 = _vecif_330__br_flag_188
					_vecif_334_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_333_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_334_exp):
						_vecif_334__br_flag_188 = _vecif_333__br_flag_188
						_vecif_334__br_flag_188 = h_broadcast_t_b_b(_vecif_334__br_flag_188, True)
						_vecif_333__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_334_exp, _vecif_334__br_flag_188, _vecif_333__br_flag_188)
					_vecif_335_exp = h_not_t_n(_vecif_333__br_flag_188)
					if any_ifexp_true_t_n(_vecif_335_exp):
						_vecif_335_s = h_copy_t_rmRes(_vecif_333_s)
						rmRes_p_set(_vecif_335_s, h_add_t_vf_t_vf(rmRes_p(_vecif_335_s), h_mul_t_f_t_vf(_vecif_330_d, r)))
						_vecif_333_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_335_exp, _vecif_335_s, _vecif_333_s)
					_vecif_330_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_333_exp, _vecif_333_s, _vecif_330_s)
					_vecif_330__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_333_exp, _vecif_333__br_flag_188, _vecif_330__br_flag_188)
				_vecif_329_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_330_exp_0, _vecif_330_s, _vecif_329_s)
				_vecif_329_d = h_where_t_n_t_n_t_n(_vecif_330_exp_0, _vecif_330_d, _vecif_329_d)
				_vecif_329_m = h_where_t_n_t_n_t_n(_vecif_330_exp_0, _vecif_330_m, _vecif_329_m)
				_vecif_329__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_330_exp_0, _vecif_330__br_flag_188, _vecif_329__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_329_exp, _vecif_329_s, s)
			d = h_where_n_t_n_t_n(_vecif_329_exp, _vecif_329_d, d)
			m = h_where_n_t_n_t_n(_vecif_329_exp, _vecif_329_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_329_exp, _vecif_329__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_336_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_336_exp):
			_vecif_336_s = h_copy_t_rmRes(s)
			_vecif_336_d = d
			_vecif_336_m = m
			_vecif_336__br_flag_188 = _br_flag_188
			_vecif_337_exp_0 = h_not_t_n(_vecif_336__br_flag_188)
			if any_ifexp_true_t_n(_vecif_337_exp_0):
				_vecif_337_s = h_copy_t_rmRes(_vecif_336_s)
				_vecif_337_d = _vecif_336_d
				_vecif_337_m = _vecif_336_m
				_vecif_337__br_flag_188 = _vecif_336__br_flag_188
				_vecif_337_d = tuple_get_retval((_call_ret_338 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_337_s)), _vecif_337_m), _vecif_337_m := tuple_get_outparam(_call_ret_338, 1)))
				_vecif_339_exp = h_less_than_t_n_n(_vecif_337_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_339_exp):
					_vecif_339_s = h_copy_t_rmRes(_vecif_337_s)
					_vecif_339__br_flag_188 = _vecif_337__br_flag_188
					rmRes_h_set(_vecif_339_s, h_broadcast_t_b_b(rmRes_h(_vecif_339_s), True))
					_vecif_339__br_flag_188 = h_broadcast_t_b_b(_vecif_339__br_flag_188, True)
					_vecif_337_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_339_exp, _vecif_339_s, _vecif_337_s)
					_vecif_337__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_339_exp, _vecif_339__br_flag_188, _vecif_337__br_flag_188)
				_vecif_340_exp = h_not_t_n(_vecif_337__br_flag_188)
				if any_ifexp_true_t_n(_vecif_340_exp):
					_vecif_340_s = h_copy_t_rmRes(_vecif_337_s)
					_vecif_340__br_flag_188 = _vecif_337__br_flag_188
					_vecif_341_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_340_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_341_exp):
						_vecif_341__br_flag_188 = _vecif_340__br_flag_188
						_vecif_341__br_flag_188 = h_broadcast_t_b_b(_vecif_341__br_flag_188, True)
						_vecif_340__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_341_exp, _vecif_341__br_flag_188, _vecif_340__br_flag_188)
					_vecif_342_exp = h_not_t_n(_vecif_340__br_flag_188)
					if any_ifexp_true_t_n(_vecif_342_exp):
						_vecif_342_s = h_copy_t_rmRes(_vecif_340_s)
						rmRes_p_set(_vecif_342_s, h_add_t_vf_t_vf(rmRes_p(_vecif_342_s), h_mul_t_f_t_vf(_vecif_337_d, r)))
						_vecif_340_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_342_exp, _vecif_342_s, _vecif_340_s)
					_vecif_337_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_340_exp, _vecif_340_s, _vecif_337_s)
					_vecif_337__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_340_exp, _vecif_340__br_flag_188, _vecif_337__br_flag_188)
				_vecif_336_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_337_exp_0, _vecif_337_s, _vecif_336_s)
				_vecif_336_d = h_where_t_n_t_n_t_n(_vecif_337_exp_0, _vecif_337_d, _vecif_336_d)
				_vecif_336_m = h_where_t_n_t_n_t_n(_vecif_337_exp_0, _vecif_337_m, _vecif_336_m)
				_vecif_336__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_337_exp_0, _vecif_337__br_flag_188, _vecif_336__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_336_exp, _vecif_336_s, s)
			d = h_where_n_t_n_t_n(_vecif_336_exp, _vecif_336_d, d)
			m = h_where_n_t_n_t_n(_vecif_336_exp, _vecif_336_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_336_exp, _vecif_336__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_343_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_343_exp):
			_vecif_343_s = h_copy_t_rmRes(s)
			_vecif_343_d = d
			_vecif_343_m = m
			_vecif_343__br_flag_188 = _br_flag_188
			_vecif_344_exp_0 = h_not_t_n(_vecif_343__br_flag_188)
			if any_ifexp_true_t_n(_vecif_344_exp_0):
				_vecif_344_s = h_copy_t_rmRes(_vecif_343_s)
				_vecif_344_d = _vecif_343_d
				_vecif_344_m = _vecif_343_m
				_vecif_344__br_flag_188 = _vecif_343__br_flag_188
				_vecif_344_d = tuple_get_retval((_call_ret_345 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_344_s)), _vecif_344_m), _vecif_344_m := tuple_get_outparam(_call_ret_345, 1)))
				_vecif_346_exp = h_less_than_t_n_n(_vecif_344_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_346_exp):
					_vecif_346_s = h_copy_t_rmRes(_vecif_344_s)
					_vecif_346__br_flag_188 = _vecif_344__br_flag_188
					rmRes_h_set(_vecif_346_s, h_broadcast_t_b_b(rmRes_h(_vecif_346_s), True))
					_vecif_346__br_flag_188 = h_broadcast_t_b_b(_vecif_346__br_flag_188, True)
					_vecif_344_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_346_exp, _vecif_346_s, _vecif_344_s)
					_vecif_344__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_346_exp, _vecif_346__br_flag_188, _vecif_344__br_flag_188)
				_vecif_347_exp = h_not_t_n(_vecif_344__br_flag_188)
				if any_ifexp_true_t_n(_vecif_347_exp):
					_vecif_347_s = h_copy_t_rmRes(_vecif_344_s)
					_vecif_347__br_flag_188 = _vecif_344__br_flag_188
					_vecif_348_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_347_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_348_exp):
						_vecif_348__br_flag_188 = _vecif_347__br_flag_188
						_vecif_348__br_flag_188 = h_broadcast_t_b_b(_vecif_348__br_flag_188, True)
						_vecif_347__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_348_exp, _vecif_348__br_flag_188, _vecif_347__br_flag_188)
					_vecif_349_exp = h_not_t_n(_vecif_347__br_flag_188)
					if any_ifexp_true_t_n(_vecif_349_exp):
						_vecif_349_s = h_copy_t_rmRes(_vecif_347_s)
						rmRes_p_set(_vecif_349_s, h_add_t_vf_t_vf(rmRes_p(_vecif_349_s), h_mul_t_f_t_vf(_vecif_344_d, r)))
						_vecif_347_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_349_exp, _vecif_349_s, _vecif_347_s)
					_vecif_344_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_347_exp, _vecif_347_s, _vecif_344_s)
					_vecif_344__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_347_exp, _vecif_347__br_flag_188, _vecif_344__br_flag_188)
				_vecif_343_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_344_exp_0, _vecif_344_s, _vecif_343_s)
				_vecif_343_d = h_where_t_n_t_n_t_n(_vecif_344_exp_0, _vecif_344_d, _vecif_343_d)
				_vecif_343_m = h_where_t_n_t_n_t_n(_vecif_344_exp_0, _vecif_344_m, _vecif_343_m)
				_vecif_343__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_344_exp_0, _vecif_344__br_flag_188, _vecif_343__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_343_exp, _vecif_343_s, s)
			d = h_where_n_t_n_t_n(_vecif_343_exp, _vecif_343_d, d)
			m = h_where_n_t_n_t_n(_vecif_343_exp, _vecif_343_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_343_exp, _vecif_343__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_350_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_350_exp):
			_vecif_350_s = h_copy_t_rmRes(s)
			_vecif_350_d = d
			_vecif_350_m = m
			_vecif_350__br_flag_188 = _br_flag_188
			_vecif_351_exp_0 = h_not_t_n(_vecif_350__br_flag_188)
			if any_ifexp_true_t_n(_vecif_351_exp_0):
				_vecif_351_s = h_copy_t_rmRes(_vecif_350_s)
				_vecif_351_d = _vecif_350_d
				_vecif_351_m = _vecif_350_m
				_vecif_351__br_flag_188 = _vecif_350__br_flag_188
				_vecif_351_d = tuple_get_retval((_call_ret_352 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_351_s)), _vecif_351_m), _vecif_351_m := tuple_get_outparam(_call_ret_352, 1)))
				_vecif_353_exp = h_less_than_t_n_n(_vecif_351_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_353_exp):
					_vecif_353_s = h_copy_t_rmRes(_vecif_351_s)
					_vecif_353__br_flag_188 = _vecif_351__br_flag_188
					rmRes_h_set(_vecif_353_s, h_broadcast_t_b_b(rmRes_h(_vecif_353_s), True))
					_vecif_353__br_flag_188 = h_broadcast_t_b_b(_vecif_353__br_flag_188, True)
					_vecif_351_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_353_exp, _vecif_353_s, _vecif_351_s)
					_vecif_351__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_353_exp, _vecif_353__br_flag_188, _vecif_351__br_flag_188)
				_vecif_354_exp = h_not_t_n(_vecif_351__br_flag_188)
				if any_ifexp_true_t_n(_vecif_354_exp):
					_vecif_354_s = h_copy_t_rmRes(_vecif_351_s)
					_vecif_354__br_flag_188 = _vecif_351__br_flag_188
					_vecif_355_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_354_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_355_exp):
						_vecif_355__br_flag_188 = _vecif_354__br_flag_188
						_vecif_355__br_flag_188 = h_broadcast_t_b_b(_vecif_355__br_flag_188, True)
						_vecif_354__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_355_exp, _vecif_355__br_flag_188, _vecif_354__br_flag_188)
					_vecif_356_exp = h_not_t_n(_vecif_354__br_flag_188)
					if any_ifexp_true_t_n(_vecif_356_exp):
						_vecif_356_s = h_copy_t_rmRes(_vecif_354_s)
						rmRes_p_set(_vecif_356_s, h_add_t_vf_t_vf(rmRes_p(_vecif_356_s), h_mul_t_f_t_vf(_vecif_351_d, r)))
						_vecif_354_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_356_exp, _vecif_356_s, _vecif_354_s)
					_vecif_351_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_354_exp, _vecif_354_s, _vecif_351_s)
					_vecif_351__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_354_exp, _vecif_354__br_flag_188, _vecif_351__br_flag_188)
				_vecif_350_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_351_exp_0, _vecif_351_s, _vecif_350_s)
				_vecif_350_d = h_where_t_n_t_n_t_n(_vecif_351_exp_0, _vecif_351_d, _vecif_350_d)
				_vecif_350_m = h_where_t_n_t_n_t_n(_vecif_351_exp_0, _vecif_351_m, _vecif_350_m)
				_vecif_350__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_351_exp_0, _vecif_351__br_flag_188, _vecif_350__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_350_exp, _vecif_350_s, s)
			d = h_where_n_t_n_t_n(_vecif_350_exp, _vecif_350_d, d)
			m = h_where_n_t_n_t_n(_vecif_350_exp, _vecif_350_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_350_exp, _vecif_350__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_357_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_357_exp):
			_vecif_357_s = h_copy_t_rmRes(s)
			_vecif_357_d = d
			_vecif_357_m = m
			_vecif_357__br_flag_188 = _br_flag_188
			_vecif_358_exp_0 = h_not_t_n(_vecif_357__br_flag_188)
			if any_ifexp_true_t_n(_vecif_358_exp_0):
				_vecif_358_s = h_copy_t_rmRes(_vecif_357_s)
				_vecif_358_d = _vecif_357_d
				_vecif_358_m = _vecif_357_m
				_vecif_358__br_flag_188 = _vecif_357__br_flag_188
				_vecif_358_d = tuple_get_retval((_call_ret_359 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_358_s)), _vecif_358_m), _vecif_358_m := tuple_get_outparam(_call_ret_359, 1)))
				_vecif_360_exp = h_less_than_t_n_n(_vecif_358_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_360_exp):
					_vecif_360_s = h_copy_t_rmRes(_vecif_358_s)
					_vecif_360__br_flag_188 = _vecif_358__br_flag_188
					rmRes_h_set(_vecif_360_s, h_broadcast_t_b_b(rmRes_h(_vecif_360_s), True))
					_vecif_360__br_flag_188 = h_broadcast_t_b_b(_vecif_360__br_flag_188, True)
					_vecif_358_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_360_exp, _vecif_360_s, _vecif_358_s)
					_vecif_358__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_360_exp, _vecif_360__br_flag_188, _vecif_358__br_flag_188)
				_vecif_361_exp = h_not_t_n(_vecif_358__br_flag_188)
				if any_ifexp_true_t_n(_vecif_361_exp):
					_vecif_361_s = h_copy_t_rmRes(_vecif_358_s)
					_vecif_361__br_flag_188 = _vecif_358__br_flag_188
					_vecif_362_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_361_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_362_exp):
						_vecif_362__br_flag_188 = _vecif_361__br_flag_188
						_vecif_362__br_flag_188 = h_broadcast_t_b_b(_vecif_362__br_flag_188, True)
						_vecif_361__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_362_exp, _vecif_362__br_flag_188, _vecif_361__br_flag_188)
					_vecif_363_exp = h_not_t_n(_vecif_361__br_flag_188)
					if any_ifexp_true_t_n(_vecif_363_exp):
						_vecif_363_s = h_copy_t_rmRes(_vecif_361_s)
						rmRes_p_set(_vecif_363_s, h_add_t_vf_t_vf(rmRes_p(_vecif_363_s), h_mul_t_f_t_vf(_vecif_358_d, r)))
						_vecif_361_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_363_exp, _vecif_363_s, _vecif_361_s)
					_vecif_358_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_361_exp, _vecif_361_s, _vecif_358_s)
					_vecif_358__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_361_exp, _vecif_361__br_flag_188, _vecif_358__br_flag_188)
				_vecif_357_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_358_exp_0, _vecif_358_s, _vecif_357_s)
				_vecif_357_d = h_where_t_n_t_n_t_n(_vecif_358_exp_0, _vecif_358_d, _vecif_357_d)
				_vecif_357_m = h_where_t_n_t_n_t_n(_vecif_358_exp_0, _vecif_358_m, _vecif_357_m)
				_vecif_357__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_358_exp_0, _vecif_358__br_flag_188, _vecif_357__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_357_exp, _vecif_357_s, s)
			d = h_where_n_t_n_t_n(_vecif_357_exp, _vecif_357_d, d)
			m = h_where_n_t_n_t_n(_vecif_357_exp, _vecif_357_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_357_exp, _vecif_357__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_364_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_364_exp):
			_vecif_364_s = h_copy_t_rmRes(s)
			_vecif_364_d = d
			_vecif_364_m = m
			_vecif_364__br_flag_188 = _br_flag_188
			_vecif_365_exp_0 = h_not_t_n(_vecif_364__br_flag_188)
			if any_ifexp_true_t_n(_vecif_365_exp_0):
				_vecif_365_s = h_copy_t_rmRes(_vecif_364_s)
				_vecif_365_d = _vecif_364_d
				_vecif_365_m = _vecif_364_m
				_vecif_365__br_flag_188 = _vecif_364__br_flag_188
				_vecif_365_d = tuple_get_retval((_call_ret_366 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_365_s)), _vecif_365_m), _vecif_365_m := tuple_get_outparam(_call_ret_366, 1)))
				_vecif_367_exp = h_less_than_t_n_n(_vecif_365_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_367_exp):
					_vecif_367_s = h_copy_t_rmRes(_vecif_365_s)
					_vecif_367__br_flag_188 = _vecif_365__br_flag_188
					rmRes_h_set(_vecif_367_s, h_broadcast_t_b_b(rmRes_h(_vecif_367_s), True))
					_vecif_367__br_flag_188 = h_broadcast_t_b_b(_vecif_367__br_flag_188, True)
					_vecif_365_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_367_exp, _vecif_367_s, _vecif_365_s)
					_vecif_365__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_367_exp, _vecif_367__br_flag_188, _vecif_365__br_flag_188)
				_vecif_368_exp = h_not_t_n(_vecif_365__br_flag_188)
				if any_ifexp_true_t_n(_vecif_368_exp):
					_vecif_368_s = h_copy_t_rmRes(_vecif_365_s)
					_vecif_368__br_flag_188 = _vecif_365__br_flag_188
					_vecif_369_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_368_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_369_exp):
						_vecif_369__br_flag_188 = _vecif_368__br_flag_188
						_vecif_369__br_flag_188 = h_broadcast_t_b_b(_vecif_369__br_flag_188, True)
						_vecif_368__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_369_exp, _vecif_369__br_flag_188, _vecif_368__br_flag_188)
					_vecif_370_exp = h_not_t_n(_vecif_368__br_flag_188)
					if any_ifexp_true_t_n(_vecif_370_exp):
						_vecif_370_s = h_copy_t_rmRes(_vecif_368_s)
						rmRes_p_set(_vecif_370_s, h_add_t_vf_t_vf(rmRes_p(_vecif_370_s), h_mul_t_f_t_vf(_vecif_365_d, r)))
						_vecif_368_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_370_exp, _vecif_370_s, _vecif_368_s)
					_vecif_365_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_368_exp, _vecif_368_s, _vecif_365_s)
					_vecif_365__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_368_exp, _vecif_368__br_flag_188, _vecif_365__br_flag_188)
				_vecif_364_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_365_exp_0, _vecif_365_s, _vecif_364_s)
				_vecif_364_d = h_where_t_n_t_n_t_n(_vecif_365_exp_0, _vecif_365_d, _vecif_364_d)
				_vecif_364_m = h_where_t_n_t_n_t_n(_vecif_365_exp_0, _vecif_365_m, _vecif_364_m)
				_vecif_364__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_365_exp_0, _vecif_365__br_flag_188, _vecif_364__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_364_exp, _vecif_364_s, s)
			d = h_where_n_t_n_t_n(_vecif_364_exp, _vecif_364_d, d)
			m = h_where_n_t_n_t_n(_vecif_364_exp, _vecif_364_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_364_exp, _vecif_364__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_371_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_371_exp):
			_vecif_371_s = h_copy_t_rmRes(s)
			_vecif_371_d = d
			_vecif_371_m = m
			_vecif_371__br_flag_188 = _br_flag_188
			_vecif_372_exp_0 = h_not_t_n(_vecif_371__br_flag_188)
			if any_ifexp_true_t_n(_vecif_372_exp_0):
				_vecif_372_s = h_copy_t_rmRes(_vecif_371_s)
				_vecif_372_d = _vecif_371_d
				_vecif_372_m = _vecif_371_m
				_vecif_372__br_flag_188 = _vecif_371__br_flag_188
				_vecif_372_d = tuple_get_retval((_call_ret_373 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_372_s)), _vecif_372_m), _vecif_372_m := tuple_get_outparam(_call_ret_373, 1)))
				_vecif_374_exp = h_less_than_t_n_n(_vecif_372_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_374_exp):
					_vecif_374_s = h_copy_t_rmRes(_vecif_372_s)
					_vecif_374__br_flag_188 = _vecif_372__br_flag_188
					rmRes_h_set(_vecif_374_s, h_broadcast_t_b_b(rmRes_h(_vecif_374_s), True))
					_vecif_374__br_flag_188 = h_broadcast_t_b_b(_vecif_374__br_flag_188, True)
					_vecif_372_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_374_exp, _vecif_374_s, _vecif_372_s)
					_vecif_372__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_374_exp, _vecif_374__br_flag_188, _vecif_372__br_flag_188)
				_vecif_375_exp = h_not_t_n(_vecif_372__br_flag_188)
				if any_ifexp_true_t_n(_vecif_375_exp):
					_vecif_375_s = h_copy_t_rmRes(_vecif_372_s)
					_vecif_375__br_flag_188 = _vecif_372__br_flag_188
					_vecif_376_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_375_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_376_exp):
						_vecif_376__br_flag_188 = _vecif_375__br_flag_188
						_vecif_376__br_flag_188 = h_broadcast_t_b_b(_vecif_376__br_flag_188, True)
						_vecif_375__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_376_exp, _vecif_376__br_flag_188, _vecif_375__br_flag_188)
					_vecif_377_exp = h_not_t_n(_vecif_375__br_flag_188)
					if any_ifexp_true_t_n(_vecif_377_exp):
						_vecif_377_s = h_copy_t_rmRes(_vecif_375_s)
						rmRes_p_set(_vecif_377_s, h_add_t_vf_t_vf(rmRes_p(_vecif_377_s), h_mul_t_f_t_vf(_vecif_372_d, r)))
						_vecif_375_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_377_exp, _vecif_377_s, _vecif_375_s)
					_vecif_372_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_375_exp, _vecif_375_s, _vecif_372_s)
					_vecif_372__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_375_exp, _vecif_375__br_flag_188, _vecif_372__br_flag_188)
				_vecif_371_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_372_exp_0, _vecif_372_s, _vecif_371_s)
				_vecif_371_d = h_where_t_n_t_n_t_n(_vecif_372_exp_0, _vecif_372_d, _vecif_371_d)
				_vecif_371_m = h_where_t_n_t_n_t_n(_vecif_372_exp_0, _vecif_372_m, _vecif_371_m)
				_vecif_371__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_372_exp_0, _vecif_372__br_flag_188, _vecif_371__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_371_exp, _vecif_371_s, s)
			d = h_where_n_t_n_t_n(_vecif_371_exp, _vecif_371_d, d)
			m = h_where_n_t_n_t_n(_vecif_371_exp, _vecif_371_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_371_exp, _vecif_371__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_378_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_378_exp):
			_vecif_378_s = h_copy_t_rmRes(s)
			_vecif_378_d = d
			_vecif_378_m = m
			_vecif_378__br_flag_188 = _br_flag_188
			_vecif_379_exp_0 = h_not_t_n(_vecif_378__br_flag_188)
			if any_ifexp_true_t_n(_vecif_379_exp_0):
				_vecif_379_s = h_copy_t_rmRes(_vecif_378_s)
				_vecif_379_d = _vecif_378_d
				_vecif_379_m = _vecif_378_m
				_vecif_379__br_flag_188 = _vecif_378__br_flag_188
				_vecif_379_d = tuple_get_retval((_call_ret_380 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_379_s)), _vecif_379_m), _vecif_379_m := tuple_get_outparam(_call_ret_380, 1)))
				_vecif_381_exp = h_less_than_t_n_n(_vecif_379_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_381_exp):
					_vecif_381_s = h_copy_t_rmRes(_vecif_379_s)
					_vecif_381__br_flag_188 = _vecif_379__br_flag_188
					rmRes_h_set(_vecif_381_s, h_broadcast_t_b_b(rmRes_h(_vecif_381_s), True))
					_vecif_381__br_flag_188 = h_broadcast_t_b_b(_vecif_381__br_flag_188, True)
					_vecif_379_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_381_exp, _vecif_381_s, _vecif_379_s)
					_vecif_379__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_381_exp, _vecif_381__br_flag_188, _vecif_379__br_flag_188)
				_vecif_382_exp = h_not_t_n(_vecif_379__br_flag_188)
				if any_ifexp_true_t_n(_vecif_382_exp):
					_vecif_382_s = h_copy_t_rmRes(_vecif_379_s)
					_vecif_382__br_flag_188 = _vecif_379__br_flag_188
					_vecif_383_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_382_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_383_exp):
						_vecif_383__br_flag_188 = _vecif_382__br_flag_188
						_vecif_383__br_flag_188 = h_broadcast_t_b_b(_vecif_383__br_flag_188, True)
						_vecif_382__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_383_exp, _vecif_383__br_flag_188, _vecif_382__br_flag_188)
					_vecif_384_exp = h_not_t_n(_vecif_382__br_flag_188)
					if any_ifexp_true_t_n(_vecif_384_exp):
						_vecif_384_s = h_copy_t_rmRes(_vecif_382_s)
						rmRes_p_set(_vecif_384_s, h_add_t_vf_t_vf(rmRes_p(_vecif_384_s), h_mul_t_f_t_vf(_vecif_379_d, r)))
						_vecif_382_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_384_exp, _vecif_384_s, _vecif_382_s)
					_vecif_379_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_382_exp, _vecif_382_s, _vecif_379_s)
					_vecif_379__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_382_exp, _vecif_382__br_flag_188, _vecif_379__br_flag_188)
				_vecif_378_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_379_exp_0, _vecif_379_s, _vecif_378_s)
				_vecif_378_d = h_where_t_n_t_n_t_n(_vecif_379_exp_0, _vecif_379_d, _vecif_378_d)
				_vecif_378_m = h_where_t_n_t_n_t_n(_vecif_379_exp_0, _vecif_379_m, _vecif_378_m)
				_vecif_378__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_379_exp_0, _vecif_379__br_flag_188, _vecif_378__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_378_exp, _vecif_378_s, s)
			d = h_where_n_t_n_t_n(_vecif_378_exp, _vecif_378_d, d)
			m = h_where_n_t_n_t_n(_vecif_378_exp, _vecif_378_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_378_exp, _vecif_378__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_385_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_385_exp):
			_vecif_385_s = h_copy_t_rmRes(s)
			_vecif_385_d = d
			_vecif_385_m = m
			_vecif_385__br_flag_188 = _br_flag_188
			_vecif_386_exp_0 = h_not_t_n(_vecif_385__br_flag_188)
			if any_ifexp_true_t_n(_vecif_386_exp_0):
				_vecif_386_s = h_copy_t_rmRes(_vecif_385_s)
				_vecif_386_d = _vecif_385_d
				_vecif_386_m = _vecif_385_m
				_vecif_386__br_flag_188 = _vecif_385__br_flag_188
				_vecif_386_d = tuple_get_retval((_call_ret_387 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_386_s)), _vecif_386_m), _vecif_386_m := tuple_get_outparam(_call_ret_387, 1)))
				_vecif_388_exp = h_less_than_t_n_n(_vecif_386_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_388_exp):
					_vecif_388_s = h_copy_t_rmRes(_vecif_386_s)
					_vecif_388__br_flag_188 = _vecif_386__br_flag_188
					rmRes_h_set(_vecif_388_s, h_broadcast_t_b_b(rmRes_h(_vecif_388_s), True))
					_vecif_388__br_flag_188 = h_broadcast_t_b_b(_vecif_388__br_flag_188, True)
					_vecif_386_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_388_exp, _vecif_388_s, _vecif_386_s)
					_vecif_386__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_388_exp, _vecif_388__br_flag_188, _vecif_386__br_flag_188)
				_vecif_389_exp = h_not_t_n(_vecif_386__br_flag_188)
				if any_ifexp_true_t_n(_vecif_389_exp):
					_vecif_389_s = h_copy_t_rmRes(_vecif_386_s)
					_vecif_389__br_flag_188 = _vecif_386__br_flag_188
					_vecif_390_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_389_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_390_exp):
						_vecif_390__br_flag_188 = _vecif_389__br_flag_188
						_vecif_390__br_flag_188 = h_broadcast_t_b_b(_vecif_390__br_flag_188, True)
						_vecif_389__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_390_exp, _vecif_390__br_flag_188, _vecif_389__br_flag_188)
					_vecif_391_exp = h_not_t_n(_vecif_389__br_flag_188)
					if any_ifexp_true_t_n(_vecif_391_exp):
						_vecif_391_s = h_copy_t_rmRes(_vecif_389_s)
						rmRes_p_set(_vecif_391_s, h_add_t_vf_t_vf(rmRes_p(_vecif_391_s), h_mul_t_f_t_vf(_vecif_386_d, r)))
						_vecif_389_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_391_exp, _vecif_391_s, _vecif_389_s)
					_vecif_386_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_389_exp, _vecif_389_s, _vecif_386_s)
					_vecif_386__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_389_exp, _vecif_389__br_flag_188, _vecif_386__br_flag_188)
				_vecif_385_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_386_exp_0, _vecif_386_s, _vecif_385_s)
				_vecif_385_d = h_where_t_n_t_n_t_n(_vecif_386_exp_0, _vecif_386_d, _vecif_385_d)
				_vecif_385_m = h_where_t_n_t_n_t_n(_vecif_386_exp_0, _vecif_386_m, _vecif_385_m)
				_vecif_385__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_386_exp_0, _vecif_386__br_flag_188, _vecif_385__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_385_exp, _vecif_385_s, s)
			d = h_where_n_t_n_t_n(_vecif_385_exp, _vecif_385_d, d)
			m = h_where_n_t_n_t_n(_vecif_385_exp, _vecif_385_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_385_exp, _vecif_385__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_392_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_392_exp):
			_vecif_392_s = h_copy_t_rmRes(s)
			_vecif_392_d = d
			_vecif_392_m = m
			_vecif_392__br_flag_188 = _br_flag_188
			_vecif_393_exp_0 = h_not_t_n(_vecif_392__br_flag_188)
			if any_ifexp_true_t_n(_vecif_393_exp_0):
				_vecif_393_s = h_copy_t_rmRes(_vecif_392_s)
				_vecif_393_d = _vecif_392_d
				_vecif_393_m = _vecif_392_m
				_vecif_393__br_flag_188 = _vecif_392__br_flag_188
				_vecif_393_d = tuple_get_retval((_call_ret_394 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_393_s)), _vecif_393_m), _vecif_393_m := tuple_get_outparam(_call_ret_394, 1)))
				_vecif_395_exp = h_less_than_t_n_n(_vecif_393_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_395_exp):
					_vecif_395_s = h_copy_t_rmRes(_vecif_393_s)
					_vecif_395__br_flag_188 = _vecif_393__br_flag_188
					rmRes_h_set(_vecif_395_s, h_broadcast_t_b_b(rmRes_h(_vecif_395_s), True))
					_vecif_395__br_flag_188 = h_broadcast_t_b_b(_vecif_395__br_flag_188, True)
					_vecif_393_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_395_exp, _vecif_395_s, _vecif_393_s)
					_vecif_393__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_395_exp, _vecif_395__br_flag_188, _vecif_393__br_flag_188)
				_vecif_396_exp = h_not_t_n(_vecif_393__br_flag_188)
				if any_ifexp_true_t_n(_vecif_396_exp):
					_vecif_396_s = h_copy_t_rmRes(_vecif_393_s)
					_vecif_396__br_flag_188 = _vecif_393__br_flag_188
					_vecif_397_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_396_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_397_exp):
						_vecif_397__br_flag_188 = _vecif_396__br_flag_188
						_vecif_397__br_flag_188 = h_broadcast_t_b_b(_vecif_397__br_flag_188, True)
						_vecif_396__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_397_exp, _vecif_397__br_flag_188, _vecif_396__br_flag_188)
					_vecif_398_exp = h_not_t_n(_vecif_396__br_flag_188)
					if any_ifexp_true_t_n(_vecif_398_exp):
						_vecif_398_s = h_copy_t_rmRes(_vecif_396_s)
						rmRes_p_set(_vecif_398_s, h_add_t_vf_t_vf(rmRes_p(_vecif_398_s), h_mul_t_f_t_vf(_vecif_393_d, r)))
						_vecif_396_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_398_exp, _vecif_398_s, _vecif_396_s)
					_vecif_393_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_396_exp, _vecif_396_s, _vecif_393_s)
					_vecif_393__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_396_exp, _vecif_396__br_flag_188, _vecif_393__br_flag_188)
				_vecif_392_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_393_exp_0, _vecif_393_s, _vecif_392_s)
				_vecif_392_d = h_where_t_n_t_n_t_n(_vecif_393_exp_0, _vecif_393_d, _vecif_392_d)
				_vecif_392_m = h_where_t_n_t_n_t_n(_vecif_393_exp_0, _vecif_393_m, _vecif_392_m)
				_vecif_392__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_393_exp_0, _vecif_393__br_flag_188, _vecif_392__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_392_exp, _vecif_392_s, s)
			d = h_where_n_t_n_t_n(_vecif_392_exp, _vecif_392_d, d)
			m = h_where_n_t_n_t_n(_vecif_392_exp, _vecif_392_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_392_exp, _vecif_392__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_399_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_399_exp):
			_vecif_399_s = h_copy_t_rmRes(s)
			_vecif_399_d = d
			_vecif_399_m = m
			_vecif_399__br_flag_188 = _br_flag_188
			_vecif_400_exp_0 = h_not_t_n(_vecif_399__br_flag_188)
			if any_ifexp_true_t_n(_vecif_400_exp_0):
				_vecif_400_s = h_copy_t_rmRes(_vecif_399_s)
				_vecif_400_d = _vecif_399_d
				_vecif_400_m = _vecif_399_m
				_vecif_400__br_flag_188 = _vecif_399__br_flag_188
				_vecif_400_d = tuple_get_retval((_call_ret_401 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_400_s)), _vecif_400_m), _vecif_400_m := tuple_get_outparam(_call_ret_401, 1)))
				_vecif_402_exp = h_less_than_t_n_n(_vecif_400_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_402_exp):
					_vecif_402_s = h_copy_t_rmRes(_vecif_400_s)
					_vecif_402__br_flag_188 = _vecif_400__br_flag_188
					rmRes_h_set(_vecif_402_s, h_broadcast_t_b_b(rmRes_h(_vecif_402_s), True))
					_vecif_402__br_flag_188 = h_broadcast_t_b_b(_vecif_402__br_flag_188, True)
					_vecif_400_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_402_exp, _vecif_402_s, _vecif_400_s)
					_vecif_400__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_402_exp, _vecif_402__br_flag_188, _vecif_400__br_flag_188)
				_vecif_403_exp = h_not_t_n(_vecif_400__br_flag_188)
				if any_ifexp_true_t_n(_vecif_403_exp):
					_vecif_403_s = h_copy_t_rmRes(_vecif_400_s)
					_vecif_403__br_flag_188 = _vecif_400__br_flag_188
					_vecif_404_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_403_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_404_exp):
						_vecif_404__br_flag_188 = _vecif_403__br_flag_188
						_vecif_404__br_flag_188 = h_broadcast_t_b_b(_vecif_404__br_flag_188, True)
						_vecif_403__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_404_exp, _vecif_404__br_flag_188, _vecif_403__br_flag_188)
					_vecif_405_exp = h_not_t_n(_vecif_403__br_flag_188)
					if any_ifexp_true_t_n(_vecif_405_exp):
						_vecif_405_s = h_copy_t_rmRes(_vecif_403_s)
						rmRes_p_set(_vecif_405_s, h_add_t_vf_t_vf(rmRes_p(_vecif_405_s), h_mul_t_f_t_vf(_vecif_400_d, r)))
						_vecif_403_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_405_exp, _vecif_405_s, _vecif_403_s)
					_vecif_400_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_403_exp, _vecif_403_s, _vecif_400_s)
					_vecif_400__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_403_exp, _vecif_403__br_flag_188, _vecif_400__br_flag_188)
				_vecif_399_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_400_exp_0, _vecif_400_s, _vecif_399_s)
				_vecif_399_d = h_where_t_n_t_n_t_n(_vecif_400_exp_0, _vecif_400_d, _vecif_399_d)
				_vecif_399_m = h_where_t_n_t_n_t_n(_vecif_400_exp_0, _vecif_400_m, _vecif_399_m)
				_vecif_399__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_400_exp_0, _vecif_400__br_flag_188, _vecif_399__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_399_exp, _vecif_399_s, s)
			d = h_where_n_t_n_t_n(_vecif_399_exp, _vecif_399_d, d)
			m = h_where_n_t_n_t_n(_vecif_399_exp, _vecif_399_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_399_exp, _vecif_399__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_406_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_406_exp):
			_vecif_406_s = h_copy_t_rmRes(s)
			_vecif_406_d = d
			_vecif_406_m = m
			_vecif_406__br_flag_188 = _br_flag_188
			_vecif_407_exp_0 = h_not_t_n(_vecif_406__br_flag_188)
			if any_ifexp_true_t_n(_vecif_407_exp_0):
				_vecif_407_s = h_copy_t_rmRes(_vecif_406_s)
				_vecif_407_d = _vecif_406_d
				_vecif_407_m = _vecif_406_m
				_vecif_407__br_flag_188 = _vecif_406__br_flag_188
				_vecif_407_d = tuple_get_retval((_call_ret_408 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_407_s)), _vecif_407_m), _vecif_407_m := tuple_get_outparam(_call_ret_408, 1)))
				_vecif_409_exp = h_less_than_t_n_n(_vecif_407_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_409_exp):
					_vecif_409_s = h_copy_t_rmRes(_vecif_407_s)
					_vecif_409__br_flag_188 = _vecif_407__br_flag_188
					rmRes_h_set(_vecif_409_s, h_broadcast_t_b_b(rmRes_h(_vecif_409_s), True))
					_vecif_409__br_flag_188 = h_broadcast_t_b_b(_vecif_409__br_flag_188, True)
					_vecif_407_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_409_exp, _vecif_409_s, _vecif_407_s)
					_vecif_407__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_409_exp, _vecif_409__br_flag_188, _vecif_407__br_flag_188)
				_vecif_410_exp = h_not_t_n(_vecif_407__br_flag_188)
				if any_ifexp_true_t_n(_vecif_410_exp):
					_vecif_410_s = h_copy_t_rmRes(_vecif_407_s)
					_vecif_410__br_flag_188 = _vecif_407__br_flag_188
					_vecif_411_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_410_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_411_exp):
						_vecif_411__br_flag_188 = _vecif_410__br_flag_188
						_vecif_411__br_flag_188 = h_broadcast_t_b_b(_vecif_411__br_flag_188, True)
						_vecif_410__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_411_exp, _vecif_411__br_flag_188, _vecif_410__br_flag_188)
					_vecif_412_exp = h_not_t_n(_vecif_410__br_flag_188)
					if any_ifexp_true_t_n(_vecif_412_exp):
						_vecif_412_s = h_copy_t_rmRes(_vecif_410_s)
						rmRes_p_set(_vecif_412_s, h_add_t_vf_t_vf(rmRes_p(_vecif_412_s), h_mul_t_f_t_vf(_vecif_407_d, r)))
						_vecif_410_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_412_exp, _vecif_412_s, _vecif_410_s)
					_vecif_407_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_410_exp, _vecif_410_s, _vecif_407_s)
					_vecif_407__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_410_exp, _vecif_410__br_flag_188, _vecif_407__br_flag_188)
				_vecif_406_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_407_exp_0, _vecif_407_s, _vecif_406_s)
				_vecif_406_d = h_where_t_n_t_n_t_n(_vecif_407_exp_0, _vecif_407_d, _vecif_406_d)
				_vecif_406_m = h_where_t_n_t_n_t_n(_vecif_407_exp_0, _vecif_407_m, _vecif_406_m)
				_vecif_406__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_407_exp_0, _vecif_407__br_flag_188, _vecif_406__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_406_exp, _vecif_406_s, s)
			d = h_where_n_t_n_t_n(_vecif_406_exp, _vecif_406_d, d)
			m = h_where_n_t_n_t_n(_vecif_406_exp, _vecif_406_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_406_exp, _vecif_406__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_413_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_413_exp):
			_vecif_413_s = h_copy_t_rmRes(s)
			_vecif_413_d = d
			_vecif_413_m = m
			_vecif_413__br_flag_188 = _br_flag_188
			_vecif_414_exp_0 = h_not_t_n(_vecif_413__br_flag_188)
			if any_ifexp_true_t_n(_vecif_414_exp_0):
				_vecif_414_s = h_copy_t_rmRes(_vecif_413_s)
				_vecif_414_d = _vecif_413_d
				_vecif_414_m = _vecif_413_m
				_vecif_414__br_flag_188 = _vecif_413__br_flag_188
				_vecif_414_d = tuple_get_retval((_call_ret_415 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_414_s)), _vecif_414_m), _vecif_414_m := tuple_get_outparam(_call_ret_415, 1)))
				_vecif_416_exp = h_less_than_t_n_n(_vecif_414_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_416_exp):
					_vecif_416_s = h_copy_t_rmRes(_vecif_414_s)
					_vecif_416__br_flag_188 = _vecif_414__br_flag_188
					rmRes_h_set(_vecif_416_s, h_broadcast_t_b_b(rmRes_h(_vecif_416_s), True))
					_vecif_416__br_flag_188 = h_broadcast_t_b_b(_vecif_416__br_flag_188, True)
					_vecif_414_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_416_exp, _vecif_416_s, _vecif_414_s)
					_vecif_414__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_416_exp, _vecif_416__br_flag_188, _vecif_414__br_flag_188)
				_vecif_417_exp = h_not_t_n(_vecif_414__br_flag_188)
				if any_ifexp_true_t_n(_vecif_417_exp):
					_vecif_417_s = h_copy_t_rmRes(_vecif_414_s)
					_vecif_417__br_flag_188 = _vecif_414__br_flag_188
					_vecif_418_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_417_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_418_exp):
						_vecif_418__br_flag_188 = _vecif_417__br_flag_188
						_vecif_418__br_flag_188 = h_broadcast_t_b_b(_vecif_418__br_flag_188, True)
						_vecif_417__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_418_exp, _vecif_418__br_flag_188, _vecif_417__br_flag_188)
					_vecif_419_exp = h_not_t_n(_vecif_417__br_flag_188)
					if any_ifexp_true_t_n(_vecif_419_exp):
						_vecif_419_s = h_copy_t_rmRes(_vecif_417_s)
						rmRes_p_set(_vecif_419_s, h_add_t_vf_t_vf(rmRes_p(_vecif_419_s), h_mul_t_f_t_vf(_vecif_414_d, r)))
						_vecif_417_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_419_exp, _vecif_419_s, _vecif_417_s)
					_vecif_414_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_417_exp, _vecif_417_s, _vecif_414_s)
					_vecif_414__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_417_exp, _vecif_417__br_flag_188, _vecif_414__br_flag_188)
				_vecif_413_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_414_exp_0, _vecif_414_s, _vecif_413_s)
				_vecif_413_d = h_where_t_n_t_n_t_n(_vecif_414_exp_0, _vecif_414_d, _vecif_413_d)
				_vecif_413_m = h_where_t_n_t_n_t_n(_vecif_414_exp_0, _vecif_414_m, _vecif_413_m)
				_vecif_413__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_414_exp_0, _vecif_414__br_flag_188, _vecif_413__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_413_exp, _vecif_413_s, s)
			d = h_where_n_t_n_t_n(_vecif_413_exp, _vecif_413_d, d)
			m = h_where_n_t_n_t_n(_vecif_413_exp, _vecif_413_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_413_exp, _vecif_413__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_420_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_420_exp):
			_vecif_420_s = h_copy_t_rmRes(s)
			_vecif_420_d = d
			_vecif_420_m = m
			_vecif_420__br_flag_188 = _br_flag_188
			_vecif_421_exp_0 = h_not_t_n(_vecif_420__br_flag_188)
			if any_ifexp_true_t_n(_vecif_421_exp_0):
				_vecif_421_s = h_copy_t_rmRes(_vecif_420_s)
				_vecif_421_d = _vecif_420_d
				_vecif_421_m = _vecif_420_m
				_vecif_421__br_flag_188 = _vecif_420__br_flag_188
				_vecif_421_d = tuple_get_retval((_call_ret_422 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_421_s)), _vecif_421_m), _vecif_421_m := tuple_get_outparam(_call_ret_422, 1)))
				_vecif_423_exp = h_less_than_t_n_n(_vecif_421_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_423_exp):
					_vecif_423_s = h_copy_t_rmRes(_vecif_421_s)
					_vecif_423__br_flag_188 = _vecif_421__br_flag_188
					rmRes_h_set(_vecif_423_s, h_broadcast_t_b_b(rmRes_h(_vecif_423_s), True))
					_vecif_423__br_flag_188 = h_broadcast_t_b_b(_vecif_423__br_flag_188, True)
					_vecif_421_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_423_exp, _vecif_423_s, _vecif_421_s)
					_vecif_421__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_423_exp, _vecif_423__br_flag_188, _vecif_421__br_flag_188)
				_vecif_424_exp = h_not_t_n(_vecif_421__br_flag_188)
				if any_ifexp_true_t_n(_vecif_424_exp):
					_vecif_424_s = h_copy_t_rmRes(_vecif_421_s)
					_vecif_424__br_flag_188 = _vecif_421__br_flag_188
					_vecif_425_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_424_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_425_exp):
						_vecif_425__br_flag_188 = _vecif_424__br_flag_188
						_vecif_425__br_flag_188 = h_broadcast_t_b_b(_vecif_425__br_flag_188, True)
						_vecif_424__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_425_exp, _vecif_425__br_flag_188, _vecif_424__br_flag_188)
					_vecif_426_exp = h_not_t_n(_vecif_424__br_flag_188)
					if any_ifexp_true_t_n(_vecif_426_exp):
						_vecif_426_s = h_copy_t_rmRes(_vecif_424_s)
						rmRes_p_set(_vecif_426_s, h_add_t_vf_t_vf(rmRes_p(_vecif_426_s), h_mul_t_f_t_vf(_vecif_421_d, r)))
						_vecif_424_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_426_exp, _vecif_426_s, _vecif_424_s)
					_vecif_421_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_424_exp, _vecif_424_s, _vecif_421_s)
					_vecif_421__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_424_exp, _vecif_424__br_flag_188, _vecif_421__br_flag_188)
				_vecif_420_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_421_exp_0, _vecif_421_s, _vecif_420_s)
				_vecif_420_d = h_where_t_n_t_n_t_n(_vecif_421_exp_0, _vecif_421_d, _vecif_420_d)
				_vecif_420_m = h_where_t_n_t_n_t_n(_vecif_421_exp_0, _vecif_421_m, _vecif_420_m)
				_vecif_420__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_421_exp_0, _vecif_421__br_flag_188, _vecif_420__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_420_exp, _vecif_420_s, s)
			d = h_where_n_t_n_t_n(_vecif_420_exp, _vecif_420_d, d)
			m = h_where_n_t_n_t_n(_vecif_420_exp, _vecif_420_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_420_exp, _vecif_420__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_427_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_427_exp):
			_vecif_427_s = h_copy_t_rmRes(s)
			_vecif_427_d = d
			_vecif_427_m = m
			_vecif_427__br_flag_188 = _br_flag_188
			_vecif_428_exp_0 = h_not_t_n(_vecif_427__br_flag_188)
			if any_ifexp_true_t_n(_vecif_428_exp_0):
				_vecif_428_s = h_copy_t_rmRes(_vecif_427_s)
				_vecif_428_d = _vecif_427_d
				_vecif_428_m = _vecif_427_m
				_vecif_428__br_flag_188 = _vecif_427__br_flag_188
				_vecif_428_d = tuple_get_retval((_call_ret_429 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_428_s)), _vecif_428_m), _vecif_428_m := tuple_get_outparam(_call_ret_429, 1)))
				_vecif_430_exp = h_less_than_t_n_n(_vecif_428_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_430_exp):
					_vecif_430_s = h_copy_t_rmRes(_vecif_428_s)
					_vecif_430__br_flag_188 = _vecif_428__br_flag_188
					rmRes_h_set(_vecif_430_s, h_broadcast_t_b_b(rmRes_h(_vecif_430_s), True))
					_vecif_430__br_flag_188 = h_broadcast_t_b_b(_vecif_430__br_flag_188, True)
					_vecif_428_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_430_exp, _vecif_430_s, _vecif_428_s)
					_vecif_428__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_430_exp, _vecif_430__br_flag_188, _vecif_428__br_flag_188)
				_vecif_431_exp = h_not_t_n(_vecif_428__br_flag_188)
				if any_ifexp_true_t_n(_vecif_431_exp):
					_vecif_431_s = h_copy_t_rmRes(_vecif_428_s)
					_vecif_431__br_flag_188 = _vecif_428__br_flag_188
					_vecif_432_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_431_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_432_exp):
						_vecif_432__br_flag_188 = _vecif_431__br_flag_188
						_vecif_432__br_flag_188 = h_broadcast_t_b_b(_vecif_432__br_flag_188, True)
						_vecif_431__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_432_exp, _vecif_432__br_flag_188, _vecif_431__br_flag_188)
					_vecif_433_exp = h_not_t_n(_vecif_431__br_flag_188)
					if any_ifexp_true_t_n(_vecif_433_exp):
						_vecif_433_s = h_copy_t_rmRes(_vecif_431_s)
						rmRes_p_set(_vecif_433_s, h_add_t_vf_t_vf(rmRes_p(_vecif_433_s), h_mul_t_f_t_vf(_vecif_428_d, r)))
						_vecif_431_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_433_exp, _vecif_433_s, _vecif_431_s)
					_vecif_428_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_431_exp, _vecif_431_s, _vecif_428_s)
					_vecif_428__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_431_exp, _vecif_431__br_flag_188, _vecif_428__br_flag_188)
				_vecif_427_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_428_exp_0, _vecif_428_s, _vecif_427_s)
				_vecif_427_d = h_where_t_n_t_n_t_n(_vecif_428_exp_0, _vecif_428_d, _vecif_427_d)
				_vecif_427_m = h_where_t_n_t_n_t_n(_vecif_428_exp_0, _vecif_428_m, _vecif_427_m)
				_vecif_427__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_428_exp_0, _vecif_428__br_flag_188, _vecif_427__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_427_exp, _vecif_427_s, s)
			d = h_where_n_t_n_t_n(_vecif_427_exp, _vecif_427_d, d)
			m = h_where_n_t_n_t_n(_vecif_427_exp, _vecif_427_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_427_exp, _vecif_427__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_434_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_434_exp):
			_vecif_434_s = h_copy_t_rmRes(s)
			_vecif_434_d = d
			_vecif_434_m = m
			_vecif_434__br_flag_188 = _br_flag_188
			_vecif_435_exp_0 = h_not_t_n(_vecif_434__br_flag_188)
			if any_ifexp_true_t_n(_vecif_435_exp_0):
				_vecif_435_s = h_copy_t_rmRes(_vecif_434_s)
				_vecif_435_d = _vecif_434_d
				_vecif_435_m = _vecif_434_m
				_vecif_435__br_flag_188 = _vecif_434__br_flag_188
				_vecif_435_d = tuple_get_retval((_call_ret_436 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_435_s)), _vecif_435_m), _vecif_435_m := tuple_get_outparam(_call_ret_436, 1)))
				_vecif_437_exp = h_less_than_t_n_n(_vecif_435_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_437_exp):
					_vecif_437_s = h_copy_t_rmRes(_vecif_435_s)
					_vecif_437__br_flag_188 = _vecif_435__br_flag_188
					rmRes_h_set(_vecif_437_s, h_broadcast_t_b_b(rmRes_h(_vecif_437_s), True))
					_vecif_437__br_flag_188 = h_broadcast_t_b_b(_vecif_437__br_flag_188, True)
					_vecif_435_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_437_exp, _vecif_437_s, _vecif_435_s)
					_vecif_435__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_437_exp, _vecif_437__br_flag_188, _vecif_435__br_flag_188)
				_vecif_438_exp = h_not_t_n(_vecif_435__br_flag_188)
				if any_ifexp_true_t_n(_vecif_438_exp):
					_vecif_438_s = h_copy_t_rmRes(_vecif_435_s)
					_vecif_438__br_flag_188 = _vecif_435__br_flag_188
					_vecif_439_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_438_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_439_exp):
						_vecif_439__br_flag_188 = _vecif_438__br_flag_188
						_vecif_439__br_flag_188 = h_broadcast_t_b_b(_vecif_439__br_flag_188, True)
						_vecif_438__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_439_exp, _vecif_439__br_flag_188, _vecif_438__br_flag_188)
					_vecif_440_exp = h_not_t_n(_vecif_438__br_flag_188)
					if any_ifexp_true_t_n(_vecif_440_exp):
						_vecif_440_s = h_copy_t_rmRes(_vecif_438_s)
						rmRes_p_set(_vecif_440_s, h_add_t_vf_t_vf(rmRes_p(_vecif_440_s), h_mul_t_f_t_vf(_vecif_435_d, r)))
						_vecif_438_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_440_exp, _vecif_440_s, _vecif_438_s)
					_vecif_435_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_438_exp, _vecif_438_s, _vecif_435_s)
					_vecif_435__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_438_exp, _vecif_438__br_flag_188, _vecif_435__br_flag_188)
				_vecif_434_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_435_exp_0, _vecif_435_s, _vecif_434_s)
				_vecif_434_d = h_where_t_n_t_n_t_n(_vecif_435_exp_0, _vecif_435_d, _vecif_434_d)
				_vecif_434_m = h_where_t_n_t_n_t_n(_vecif_435_exp_0, _vecif_435_m, _vecif_434_m)
				_vecif_434__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_435_exp_0, _vecif_435__br_flag_188, _vecif_434__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_434_exp, _vecif_434_s, s)
			d = h_where_n_t_n_t_n(_vecif_434_exp, _vecif_434_d, d)
			m = h_where_n_t_n_t_n(_vecif_434_exp, _vecif_434_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_434_exp, _vecif_434__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_441_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_441_exp):
			_vecif_441_s = h_copy_t_rmRes(s)
			_vecif_441_d = d
			_vecif_441_m = m
			_vecif_441__br_flag_188 = _br_flag_188
			_vecif_442_exp_0 = h_not_t_n(_vecif_441__br_flag_188)
			if any_ifexp_true_t_n(_vecif_442_exp_0):
				_vecif_442_s = h_copy_t_rmRes(_vecif_441_s)
				_vecif_442_d = _vecif_441_d
				_vecif_442_m = _vecif_441_m
				_vecif_442__br_flag_188 = _vecif_441__br_flag_188
				_vecif_442_d = tuple_get_retval((_call_ret_443 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_442_s)), _vecif_442_m), _vecif_442_m := tuple_get_outparam(_call_ret_443, 1)))
				_vecif_444_exp = h_less_than_t_n_n(_vecif_442_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_444_exp):
					_vecif_444_s = h_copy_t_rmRes(_vecif_442_s)
					_vecif_444__br_flag_188 = _vecif_442__br_flag_188
					rmRes_h_set(_vecif_444_s, h_broadcast_t_b_b(rmRes_h(_vecif_444_s), True))
					_vecif_444__br_flag_188 = h_broadcast_t_b_b(_vecif_444__br_flag_188, True)
					_vecif_442_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_444_exp, _vecif_444_s, _vecif_442_s)
					_vecif_442__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_444_exp, _vecif_444__br_flag_188, _vecif_442__br_flag_188)
				_vecif_445_exp = h_not_t_n(_vecif_442__br_flag_188)
				if any_ifexp_true_t_n(_vecif_445_exp):
					_vecif_445_s = h_copy_t_rmRes(_vecif_442_s)
					_vecif_445__br_flag_188 = _vecif_442__br_flag_188
					_vecif_446_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_445_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_446_exp):
						_vecif_446__br_flag_188 = _vecif_445__br_flag_188
						_vecif_446__br_flag_188 = h_broadcast_t_b_b(_vecif_446__br_flag_188, True)
						_vecif_445__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_446_exp, _vecif_446__br_flag_188, _vecif_445__br_flag_188)
					_vecif_447_exp = h_not_t_n(_vecif_445__br_flag_188)
					if any_ifexp_true_t_n(_vecif_447_exp):
						_vecif_447_s = h_copy_t_rmRes(_vecif_445_s)
						rmRes_p_set(_vecif_447_s, h_add_t_vf_t_vf(rmRes_p(_vecif_447_s), h_mul_t_f_t_vf(_vecif_442_d, r)))
						_vecif_445_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_447_exp, _vecif_447_s, _vecif_445_s)
					_vecif_442_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_445_exp, _vecif_445_s, _vecif_442_s)
					_vecif_442__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_445_exp, _vecif_445__br_flag_188, _vecif_442__br_flag_188)
				_vecif_441_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_442_exp_0, _vecif_442_s, _vecif_441_s)
				_vecif_441_d = h_where_t_n_t_n_t_n(_vecif_442_exp_0, _vecif_442_d, _vecif_441_d)
				_vecif_441_m = h_where_t_n_t_n_t_n(_vecif_442_exp_0, _vecif_442_m, _vecif_441_m)
				_vecif_441__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_442_exp_0, _vecif_442__br_flag_188, _vecif_441__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_441_exp, _vecif_441_s, s)
			d = h_where_n_t_n_t_n(_vecif_441_exp, _vecif_441_d, d)
			m = h_where_n_t_n_t_n(_vecif_441_exp, _vecif_441_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_441_exp, _vecif_441__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_448_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_448_exp):
			_vecif_448_s = h_copy_t_rmRes(s)
			_vecif_448_d = d
			_vecif_448_m = m
			_vecif_448__br_flag_188 = _br_flag_188
			_vecif_449_exp_0 = h_not_t_n(_vecif_448__br_flag_188)
			if any_ifexp_true_t_n(_vecif_449_exp_0):
				_vecif_449_s = h_copy_t_rmRes(_vecif_448_s)
				_vecif_449_d = _vecif_448_d
				_vecif_449_m = _vecif_448_m
				_vecif_449__br_flag_188 = _vecif_448__br_flag_188
				_vecif_449_d = tuple_get_retval((_call_ret_450 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_449_s)), _vecif_449_m), _vecif_449_m := tuple_get_outparam(_call_ret_450, 1)))
				_vecif_451_exp = h_less_than_t_n_n(_vecif_449_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_451_exp):
					_vecif_451_s = h_copy_t_rmRes(_vecif_449_s)
					_vecif_451__br_flag_188 = _vecif_449__br_flag_188
					rmRes_h_set(_vecif_451_s, h_broadcast_t_b_b(rmRes_h(_vecif_451_s), True))
					_vecif_451__br_flag_188 = h_broadcast_t_b_b(_vecif_451__br_flag_188, True)
					_vecif_449_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_451_exp, _vecif_451_s, _vecif_449_s)
					_vecif_449__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_451_exp, _vecif_451__br_flag_188, _vecif_449__br_flag_188)
				_vecif_452_exp = h_not_t_n(_vecif_449__br_flag_188)
				if any_ifexp_true_t_n(_vecif_452_exp):
					_vecif_452_s = h_copy_t_rmRes(_vecif_449_s)
					_vecif_452__br_flag_188 = _vecif_449__br_flag_188
					_vecif_453_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_452_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_453_exp):
						_vecif_453__br_flag_188 = _vecif_452__br_flag_188
						_vecif_453__br_flag_188 = h_broadcast_t_b_b(_vecif_453__br_flag_188, True)
						_vecif_452__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_453_exp, _vecif_453__br_flag_188, _vecif_452__br_flag_188)
					_vecif_454_exp = h_not_t_n(_vecif_452__br_flag_188)
					if any_ifexp_true_t_n(_vecif_454_exp):
						_vecif_454_s = h_copy_t_rmRes(_vecif_452_s)
						rmRes_p_set(_vecif_454_s, h_add_t_vf_t_vf(rmRes_p(_vecif_454_s), h_mul_t_f_t_vf(_vecif_449_d, r)))
						_vecif_452_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_454_exp, _vecif_454_s, _vecif_452_s)
					_vecif_449_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_452_exp, _vecif_452_s, _vecif_449_s)
					_vecif_449__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_452_exp, _vecif_452__br_flag_188, _vecif_449__br_flag_188)
				_vecif_448_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_449_exp_0, _vecif_449_s, _vecif_448_s)
				_vecif_448_d = h_where_t_n_t_n_t_n(_vecif_449_exp_0, _vecif_449_d, _vecif_448_d)
				_vecif_448_m = h_where_t_n_t_n_t_n(_vecif_449_exp_0, _vecif_449_m, _vecif_448_m)
				_vecif_448__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_449_exp_0, _vecif_449__br_flag_188, _vecif_448__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_448_exp, _vecif_448_s, s)
			d = h_where_n_t_n_t_n(_vecif_448_exp, _vecif_448_d, d)
			m = h_where_n_t_n_t_n(_vecif_448_exp, _vecif_448_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_448_exp, _vecif_448__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_455_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_455_exp):
			_vecif_455_s = h_copy_t_rmRes(s)
			_vecif_455_d = d
			_vecif_455_m = m
			_vecif_455__br_flag_188 = _br_flag_188
			_vecif_456_exp_0 = h_not_t_n(_vecif_455__br_flag_188)
			if any_ifexp_true_t_n(_vecif_456_exp_0):
				_vecif_456_s = h_copy_t_rmRes(_vecif_455_s)
				_vecif_456_d = _vecif_455_d
				_vecif_456_m = _vecif_455_m
				_vecif_456__br_flag_188 = _vecif_455__br_flag_188
				_vecif_456_d = tuple_get_retval((_call_ret_457 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_456_s)), _vecif_456_m), _vecif_456_m := tuple_get_outparam(_call_ret_457, 1)))
				_vecif_458_exp = h_less_than_t_n_n(_vecif_456_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_458_exp):
					_vecif_458_s = h_copy_t_rmRes(_vecif_456_s)
					_vecif_458__br_flag_188 = _vecif_456__br_flag_188
					rmRes_h_set(_vecif_458_s, h_broadcast_t_b_b(rmRes_h(_vecif_458_s), True))
					_vecif_458__br_flag_188 = h_broadcast_t_b_b(_vecif_458__br_flag_188, True)
					_vecif_456_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_458_exp, _vecif_458_s, _vecif_456_s)
					_vecif_456__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_458_exp, _vecif_458__br_flag_188, _vecif_456__br_flag_188)
				_vecif_459_exp = h_not_t_n(_vecif_456__br_flag_188)
				if any_ifexp_true_t_n(_vecif_459_exp):
					_vecif_459_s = h_copy_t_rmRes(_vecif_456_s)
					_vecif_459__br_flag_188 = _vecif_456__br_flag_188
					_vecif_460_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_459_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_460_exp):
						_vecif_460__br_flag_188 = _vecif_459__br_flag_188
						_vecif_460__br_flag_188 = h_broadcast_t_b_b(_vecif_460__br_flag_188, True)
						_vecif_459__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_460_exp, _vecif_460__br_flag_188, _vecif_459__br_flag_188)
					_vecif_461_exp = h_not_t_n(_vecif_459__br_flag_188)
					if any_ifexp_true_t_n(_vecif_461_exp):
						_vecif_461_s = h_copy_t_rmRes(_vecif_459_s)
						rmRes_p_set(_vecif_461_s, h_add_t_vf_t_vf(rmRes_p(_vecif_461_s), h_mul_t_f_t_vf(_vecif_456_d, r)))
						_vecif_459_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_461_exp, _vecif_461_s, _vecif_459_s)
					_vecif_456_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_459_exp, _vecif_459_s, _vecif_456_s)
					_vecif_456__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_459_exp, _vecif_459__br_flag_188, _vecif_456__br_flag_188)
				_vecif_455_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_456_exp_0, _vecif_456_s, _vecif_455_s)
				_vecif_455_d = h_where_t_n_t_n_t_n(_vecif_456_exp_0, _vecif_456_d, _vecif_455_d)
				_vecif_455_m = h_where_t_n_t_n_t_n(_vecif_456_exp_0, _vecif_456_m, _vecif_455_m)
				_vecif_455__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_456_exp_0, _vecif_456__br_flag_188, _vecif_455__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_455_exp, _vecif_455_s, s)
			d = h_where_n_t_n_t_n(_vecif_455_exp, _vecif_455_d, d)
			m = h_where_n_t_n_t_n(_vecif_455_exp, _vecif_455_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_455_exp, _vecif_455__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_462_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_462_exp):
			_vecif_462_s = h_copy_t_rmRes(s)
			_vecif_462_d = d
			_vecif_462_m = m
			_vecif_462__br_flag_188 = _br_flag_188
			_vecif_463_exp_0 = h_not_t_n(_vecif_462__br_flag_188)
			if any_ifexp_true_t_n(_vecif_463_exp_0):
				_vecif_463_s = h_copy_t_rmRes(_vecif_462_s)
				_vecif_463_d = _vecif_462_d
				_vecif_463_m = _vecif_462_m
				_vecif_463__br_flag_188 = _vecif_462__br_flag_188
				_vecif_463_d = tuple_get_retval((_call_ret_464 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_463_s)), _vecif_463_m), _vecif_463_m := tuple_get_outparam(_call_ret_464, 1)))
				_vecif_465_exp = h_less_than_t_n_n(_vecif_463_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_465_exp):
					_vecif_465_s = h_copy_t_rmRes(_vecif_463_s)
					_vecif_465__br_flag_188 = _vecif_463__br_flag_188
					rmRes_h_set(_vecif_465_s, h_broadcast_t_b_b(rmRes_h(_vecif_465_s), True))
					_vecif_465__br_flag_188 = h_broadcast_t_b_b(_vecif_465__br_flag_188, True)
					_vecif_463_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_465_exp, _vecif_465_s, _vecif_463_s)
					_vecif_463__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_465_exp, _vecif_465__br_flag_188, _vecif_463__br_flag_188)
				_vecif_466_exp = h_not_t_n(_vecif_463__br_flag_188)
				if any_ifexp_true_t_n(_vecif_466_exp):
					_vecif_466_s = h_copy_t_rmRes(_vecif_463_s)
					_vecif_466__br_flag_188 = _vecif_463__br_flag_188
					_vecif_467_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_466_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_467_exp):
						_vecif_467__br_flag_188 = _vecif_466__br_flag_188
						_vecif_467__br_flag_188 = h_broadcast_t_b_b(_vecif_467__br_flag_188, True)
						_vecif_466__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_467_exp, _vecif_467__br_flag_188, _vecif_466__br_flag_188)
					_vecif_468_exp = h_not_t_n(_vecif_466__br_flag_188)
					if any_ifexp_true_t_n(_vecif_468_exp):
						_vecif_468_s = h_copy_t_rmRes(_vecif_466_s)
						rmRes_p_set(_vecif_468_s, h_add_t_vf_t_vf(rmRes_p(_vecif_468_s), h_mul_t_f_t_vf(_vecif_463_d, r)))
						_vecif_466_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_468_exp, _vecif_468_s, _vecif_466_s)
					_vecif_463_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_466_exp, _vecif_466_s, _vecif_463_s)
					_vecif_463__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_466_exp, _vecif_466__br_flag_188, _vecif_463__br_flag_188)
				_vecif_462_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_463_exp_0, _vecif_463_s, _vecif_462_s)
				_vecif_462_d = h_where_t_n_t_n_t_n(_vecif_463_exp_0, _vecif_463_d, _vecif_462_d)
				_vecif_462_m = h_where_t_n_t_n_t_n(_vecif_463_exp_0, _vecif_463_m, _vecif_462_m)
				_vecif_462__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_463_exp_0, _vecif_463__br_flag_188, _vecif_462__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_462_exp, _vecif_462_s, s)
			d = h_where_n_t_n_t_n(_vecif_462_exp, _vecif_462_d, d)
			m = h_where_n_t_n_t_n(_vecif_462_exp, _vecif_462_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_462_exp, _vecif_462__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_469_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_469_exp):
			_vecif_469_s = h_copy_t_rmRes(s)
			_vecif_469_d = d
			_vecif_469_m = m
			_vecif_469__br_flag_188 = _br_flag_188
			_vecif_470_exp_0 = h_not_t_n(_vecif_469__br_flag_188)
			if any_ifexp_true_t_n(_vecif_470_exp_0):
				_vecif_470_s = h_copy_t_rmRes(_vecif_469_s)
				_vecif_470_d = _vecif_469_d
				_vecif_470_m = _vecif_469_m
				_vecif_470__br_flag_188 = _vecif_469__br_flag_188
				_vecif_470_d = tuple_get_retval((_call_ret_471 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_470_s)), _vecif_470_m), _vecif_470_m := tuple_get_outparam(_call_ret_471, 1)))
				_vecif_472_exp = h_less_than_t_n_n(_vecif_470_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_472_exp):
					_vecif_472_s = h_copy_t_rmRes(_vecif_470_s)
					_vecif_472__br_flag_188 = _vecif_470__br_flag_188
					rmRes_h_set(_vecif_472_s, h_broadcast_t_b_b(rmRes_h(_vecif_472_s), True))
					_vecif_472__br_flag_188 = h_broadcast_t_b_b(_vecif_472__br_flag_188, True)
					_vecif_470_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_472_exp, _vecif_472_s, _vecif_470_s)
					_vecif_470__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_472_exp, _vecif_472__br_flag_188, _vecif_470__br_flag_188)
				_vecif_473_exp = h_not_t_n(_vecif_470__br_flag_188)
				if any_ifexp_true_t_n(_vecif_473_exp):
					_vecif_473_s = h_copy_t_rmRes(_vecif_470_s)
					_vecif_473__br_flag_188 = _vecif_470__br_flag_188
					_vecif_474_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_473_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_474_exp):
						_vecif_474__br_flag_188 = _vecif_473__br_flag_188
						_vecif_474__br_flag_188 = h_broadcast_t_b_b(_vecif_474__br_flag_188, True)
						_vecif_473__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_474_exp, _vecif_474__br_flag_188, _vecif_473__br_flag_188)
					_vecif_475_exp = h_not_t_n(_vecif_473__br_flag_188)
					if any_ifexp_true_t_n(_vecif_475_exp):
						_vecif_475_s = h_copy_t_rmRes(_vecif_473_s)
						rmRes_p_set(_vecif_475_s, h_add_t_vf_t_vf(rmRes_p(_vecif_475_s), h_mul_t_f_t_vf(_vecif_470_d, r)))
						_vecif_473_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_475_exp, _vecif_475_s, _vecif_473_s)
					_vecif_470_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_473_exp, _vecif_473_s, _vecif_470_s)
					_vecif_470__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_473_exp, _vecif_473__br_flag_188, _vecif_470__br_flag_188)
				_vecif_469_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_470_exp_0, _vecif_470_s, _vecif_469_s)
				_vecif_469_d = h_where_t_n_t_n_t_n(_vecif_470_exp_0, _vecif_470_d, _vecif_469_d)
				_vecif_469_m = h_where_t_n_t_n_t_n(_vecif_470_exp_0, _vecif_470_m, _vecif_469_m)
				_vecif_469__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_470_exp_0, _vecif_470__br_flag_188, _vecif_469__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_469_exp, _vecif_469_s, s)
			d = h_where_n_t_n_t_n(_vecif_469_exp, _vecif_469_d, d)
			m = h_where_n_t_n_t_n(_vecif_469_exp, _vecif_469_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_469_exp, _vecif_469__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_476_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_476_exp):
			_vecif_476_s = h_copy_t_rmRes(s)
			_vecif_476_d = d
			_vecif_476_m = m
			_vecif_476__br_flag_188 = _br_flag_188
			_vecif_477_exp_0 = h_not_t_n(_vecif_476__br_flag_188)
			if any_ifexp_true_t_n(_vecif_477_exp_0):
				_vecif_477_s = h_copy_t_rmRes(_vecif_476_s)
				_vecif_477_d = _vecif_476_d
				_vecif_477_m = _vecif_476_m
				_vecif_477__br_flag_188 = _vecif_476__br_flag_188
				_vecif_477_d = tuple_get_retval((_call_ret_478 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_477_s)), _vecif_477_m), _vecif_477_m := tuple_get_outparam(_call_ret_478, 1)))
				_vecif_479_exp = h_less_than_t_n_n(_vecif_477_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_479_exp):
					_vecif_479_s = h_copy_t_rmRes(_vecif_477_s)
					_vecif_479__br_flag_188 = _vecif_477__br_flag_188
					rmRes_h_set(_vecif_479_s, h_broadcast_t_b_b(rmRes_h(_vecif_479_s), True))
					_vecif_479__br_flag_188 = h_broadcast_t_b_b(_vecif_479__br_flag_188, True)
					_vecif_477_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_479_exp, _vecif_479_s, _vecif_477_s)
					_vecif_477__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_479_exp, _vecif_479__br_flag_188, _vecif_477__br_flag_188)
				_vecif_480_exp = h_not_t_n(_vecif_477__br_flag_188)
				if any_ifexp_true_t_n(_vecif_480_exp):
					_vecif_480_s = h_copy_t_rmRes(_vecif_477_s)
					_vecif_480__br_flag_188 = _vecif_477__br_flag_188
					_vecif_481_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_480_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_481_exp):
						_vecif_481__br_flag_188 = _vecif_480__br_flag_188
						_vecif_481__br_flag_188 = h_broadcast_t_b_b(_vecif_481__br_flag_188, True)
						_vecif_480__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_481_exp, _vecif_481__br_flag_188, _vecif_480__br_flag_188)
					_vecif_482_exp = h_not_t_n(_vecif_480__br_flag_188)
					if any_ifexp_true_t_n(_vecif_482_exp):
						_vecif_482_s = h_copy_t_rmRes(_vecif_480_s)
						rmRes_p_set(_vecif_482_s, h_add_t_vf_t_vf(rmRes_p(_vecif_482_s), h_mul_t_f_t_vf(_vecif_477_d, r)))
						_vecif_480_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_482_exp, _vecif_482_s, _vecif_480_s)
					_vecif_477_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_480_exp, _vecif_480_s, _vecif_477_s)
					_vecif_477__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_480_exp, _vecif_480__br_flag_188, _vecif_477__br_flag_188)
				_vecif_476_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_477_exp_0, _vecif_477_s, _vecif_476_s)
				_vecif_476_d = h_where_t_n_t_n_t_n(_vecif_477_exp_0, _vecif_477_d, _vecif_476_d)
				_vecif_476_m = h_where_t_n_t_n_t_n(_vecif_477_exp_0, _vecif_477_m, _vecif_476_m)
				_vecif_476__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_477_exp_0, _vecif_477__br_flag_188, _vecif_476__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_476_exp, _vecif_476_s, s)
			d = h_where_n_t_n_t_n(_vecif_476_exp, _vecif_476_d, d)
			m = h_where_n_t_n_t_n(_vecif_476_exp, _vecif_476_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_476_exp, _vecif_476__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_483_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_483_exp):
			_vecif_483_s = h_copy_t_rmRes(s)
			_vecif_483_d = d
			_vecif_483_m = m
			_vecif_483__br_flag_188 = _br_flag_188
			_vecif_484_exp_0 = h_not_t_n(_vecif_483__br_flag_188)
			if any_ifexp_true_t_n(_vecif_484_exp_0):
				_vecif_484_s = h_copy_t_rmRes(_vecif_483_s)
				_vecif_484_d = _vecif_483_d
				_vecif_484_m = _vecif_483_m
				_vecif_484__br_flag_188 = _vecif_483__br_flag_188
				_vecif_484_d = tuple_get_retval((_call_ret_485 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_484_s)), _vecif_484_m), _vecif_484_m := tuple_get_outparam(_call_ret_485, 1)))
				_vecif_486_exp = h_less_than_t_n_n(_vecif_484_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_486_exp):
					_vecif_486_s = h_copy_t_rmRes(_vecif_484_s)
					_vecif_486__br_flag_188 = _vecif_484__br_flag_188
					rmRes_h_set(_vecif_486_s, h_broadcast_t_b_b(rmRes_h(_vecif_486_s), True))
					_vecif_486__br_flag_188 = h_broadcast_t_b_b(_vecif_486__br_flag_188, True)
					_vecif_484_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_486_exp, _vecif_486_s, _vecif_484_s)
					_vecif_484__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_486_exp, _vecif_486__br_flag_188, _vecif_484__br_flag_188)
				_vecif_487_exp = h_not_t_n(_vecif_484__br_flag_188)
				if any_ifexp_true_t_n(_vecif_487_exp):
					_vecif_487_s = h_copy_t_rmRes(_vecif_484_s)
					_vecif_487__br_flag_188 = _vecif_484__br_flag_188
					_vecif_488_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_487_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_488_exp):
						_vecif_488__br_flag_188 = _vecif_487__br_flag_188
						_vecif_488__br_flag_188 = h_broadcast_t_b_b(_vecif_488__br_flag_188, True)
						_vecif_487__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_488_exp, _vecif_488__br_flag_188, _vecif_487__br_flag_188)
					_vecif_489_exp = h_not_t_n(_vecif_487__br_flag_188)
					if any_ifexp_true_t_n(_vecif_489_exp):
						_vecif_489_s = h_copy_t_rmRes(_vecif_487_s)
						rmRes_p_set(_vecif_489_s, h_add_t_vf_t_vf(rmRes_p(_vecif_489_s), h_mul_t_f_t_vf(_vecif_484_d, r)))
						_vecif_487_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_489_exp, _vecif_489_s, _vecif_487_s)
					_vecif_484_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_487_exp, _vecif_487_s, _vecif_484_s)
					_vecif_484__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_487_exp, _vecif_487__br_flag_188, _vecif_484__br_flag_188)
				_vecif_483_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_484_exp_0, _vecif_484_s, _vecif_483_s)
				_vecif_483_d = h_where_t_n_t_n_t_n(_vecif_484_exp_0, _vecif_484_d, _vecif_483_d)
				_vecif_483_m = h_where_t_n_t_n_t_n(_vecif_484_exp_0, _vecif_484_m, _vecif_483_m)
				_vecif_483__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_484_exp_0, _vecif_484__br_flag_188, _vecif_483__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_483_exp, _vecif_483_s, s)
			d = h_where_n_t_n_t_n(_vecif_483_exp, _vecif_483_d, d)
			m = h_where_n_t_n_t_n(_vecif_483_exp, _vecif_483_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_483_exp, _vecif_483__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_490_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_490_exp):
			_vecif_490_s = h_copy_t_rmRes(s)
			_vecif_490_d = d
			_vecif_490_m = m
			_vecif_490__br_flag_188 = _br_flag_188
			_vecif_491_exp_0 = h_not_t_n(_vecif_490__br_flag_188)
			if any_ifexp_true_t_n(_vecif_491_exp_0):
				_vecif_491_s = h_copy_t_rmRes(_vecif_490_s)
				_vecif_491_d = _vecif_490_d
				_vecif_491_m = _vecif_490_m
				_vecif_491__br_flag_188 = _vecif_490__br_flag_188
				_vecif_491_d = tuple_get_retval((_call_ret_492 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_491_s)), _vecif_491_m), _vecif_491_m := tuple_get_outparam(_call_ret_492, 1)))
				_vecif_493_exp = h_less_than_t_n_n(_vecif_491_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_493_exp):
					_vecif_493_s = h_copy_t_rmRes(_vecif_491_s)
					_vecif_493__br_flag_188 = _vecif_491__br_flag_188
					rmRes_h_set(_vecif_493_s, h_broadcast_t_b_b(rmRes_h(_vecif_493_s), True))
					_vecif_493__br_flag_188 = h_broadcast_t_b_b(_vecif_493__br_flag_188, True)
					_vecif_491_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_493_exp, _vecif_493_s, _vecif_491_s)
					_vecif_491__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_493_exp, _vecif_493__br_flag_188, _vecif_491__br_flag_188)
				_vecif_494_exp = h_not_t_n(_vecif_491__br_flag_188)
				if any_ifexp_true_t_n(_vecif_494_exp):
					_vecif_494_s = h_copy_t_rmRes(_vecif_491_s)
					_vecif_494__br_flag_188 = _vecif_491__br_flag_188
					_vecif_495_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_494_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_495_exp):
						_vecif_495__br_flag_188 = _vecif_494__br_flag_188
						_vecif_495__br_flag_188 = h_broadcast_t_b_b(_vecif_495__br_flag_188, True)
						_vecif_494__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_495_exp, _vecif_495__br_flag_188, _vecif_494__br_flag_188)
					_vecif_496_exp = h_not_t_n(_vecif_494__br_flag_188)
					if any_ifexp_true_t_n(_vecif_496_exp):
						_vecif_496_s = h_copy_t_rmRes(_vecif_494_s)
						rmRes_p_set(_vecif_496_s, h_add_t_vf_t_vf(rmRes_p(_vecif_496_s), h_mul_t_f_t_vf(_vecif_491_d, r)))
						_vecif_494_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_496_exp, _vecif_496_s, _vecif_494_s)
					_vecif_491_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_494_exp, _vecif_494_s, _vecif_491_s)
					_vecif_491__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_494_exp, _vecif_494__br_flag_188, _vecif_491__br_flag_188)
				_vecif_490_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_491_exp_0, _vecif_491_s, _vecif_490_s)
				_vecif_490_d = h_where_t_n_t_n_t_n(_vecif_491_exp_0, _vecif_491_d, _vecif_490_d)
				_vecif_490_m = h_where_t_n_t_n_t_n(_vecif_491_exp_0, _vecif_491_m, _vecif_490_m)
				_vecif_490__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_491_exp_0, _vecif_491__br_flag_188, _vecif_490__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_490_exp, _vecif_490_s, s)
			d = h_where_n_t_n_t_n(_vecif_490_exp, _vecif_490_d, d)
			m = h_where_n_t_n_t_n(_vecif_490_exp, _vecif_490_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_490_exp, _vecif_490__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_497_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_497_exp):
			_vecif_497_s = h_copy_t_rmRes(s)
			_vecif_497_d = d
			_vecif_497_m = m
			_vecif_497__br_flag_188 = _br_flag_188
			_vecif_498_exp_0 = h_not_t_n(_vecif_497__br_flag_188)
			if any_ifexp_true_t_n(_vecif_498_exp_0):
				_vecif_498_s = h_copy_t_rmRes(_vecif_497_s)
				_vecif_498_d = _vecif_497_d
				_vecif_498_m = _vecif_497_m
				_vecif_498__br_flag_188 = _vecif_497__br_flag_188
				_vecif_498_d = tuple_get_retval((_call_ret_499 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_498_s)), _vecif_498_m), _vecif_498_m := tuple_get_outparam(_call_ret_499, 1)))
				_vecif_500_exp = h_less_than_t_n_n(_vecif_498_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_500_exp):
					_vecif_500_s = h_copy_t_rmRes(_vecif_498_s)
					_vecif_500__br_flag_188 = _vecif_498__br_flag_188
					rmRes_h_set(_vecif_500_s, h_broadcast_t_b_b(rmRes_h(_vecif_500_s), True))
					_vecif_500__br_flag_188 = h_broadcast_t_b_b(_vecif_500__br_flag_188, True)
					_vecif_498_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_500_exp, _vecif_500_s, _vecif_498_s)
					_vecif_498__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_500_exp, _vecif_500__br_flag_188, _vecif_498__br_flag_188)
				_vecif_501_exp = h_not_t_n(_vecif_498__br_flag_188)
				if any_ifexp_true_t_n(_vecif_501_exp):
					_vecif_501_s = h_copy_t_rmRes(_vecif_498_s)
					_vecif_501__br_flag_188 = _vecif_498__br_flag_188
					_vecif_502_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_501_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_502_exp):
						_vecif_502__br_flag_188 = _vecif_501__br_flag_188
						_vecif_502__br_flag_188 = h_broadcast_t_b_b(_vecif_502__br_flag_188, True)
						_vecif_501__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_502_exp, _vecif_502__br_flag_188, _vecif_501__br_flag_188)
					_vecif_503_exp = h_not_t_n(_vecif_501__br_flag_188)
					if any_ifexp_true_t_n(_vecif_503_exp):
						_vecif_503_s = h_copy_t_rmRes(_vecif_501_s)
						rmRes_p_set(_vecif_503_s, h_add_t_vf_t_vf(rmRes_p(_vecif_503_s), h_mul_t_f_t_vf(_vecif_498_d, r)))
						_vecif_501_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_503_exp, _vecif_503_s, _vecif_501_s)
					_vecif_498_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_501_exp, _vecif_501_s, _vecif_498_s)
					_vecif_498__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_501_exp, _vecif_501__br_flag_188, _vecif_498__br_flag_188)
				_vecif_497_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_498_exp_0, _vecif_498_s, _vecif_497_s)
				_vecif_497_d = h_where_t_n_t_n_t_n(_vecif_498_exp_0, _vecif_498_d, _vecif_497_d)
				_vecif_497_m = h_where_t_n_t_n_t_n(_vecif_498_exp_0, _vecif_498_m, _vecif_497_m)
				_vecif_497__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_498_exp_0, _vecif_498__br_flag_188, _vecif_497__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_497_exp, _vecif_497_s, s)
			d = h_where_n_t_n_t_n(_vecif_497_exp, _vecif_497_d, d)
			m = h_where_n_t_n_t_n(_vecif_497_exp, _vecif_497_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_497_exp, _vecif_497__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_504_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_504_exp):
			_vecif_504_s = h_copy_t_rmRes(s)
			_vecif_504_d = d
			_vecif_504_m = m
			_vecif_504__br_flag_188 = _br_flag_188
			_vecif_505_exp_0 = h_not_t_n(_vecif_504__br_flag_188)
			if any_ifexp_true_t_n(_vecif_505_exp_0):
				_vecif_505_s = h_copy_t_rmRes(_vecif_504_s)
				_vecif_505_d = _vecif_504_d
				_vecif_505_m = _vecif_504_m
				_vecif_505__br_flag_188 = _vecif_504__br_flag_188
				_vecif_505_d = tuple_get_retval((_call_ret_506 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_505_s)), _vecif_505_m), _vecif_505_m := tuple_get_outparam(_call_ret_506, 1)))
				_vecif_507_exp = h_less_than_t_n_n(_vecif_505_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_507_exp):
					_vecif_507_s = h_copy_t_rmRes(_vecif_505_s)
					_vecif_507__br_flag_188 = _vecif_505__br_flag_188
					rmRes_h_set(_vecif_507_s, h_broadcast_t_b_b(rmRes_h(_vecif_507_s), True))
					_vecif_507__br_flag_188 = h_broadcast_t_b_b(_vecif_507__br_flag_188, True)
					_vecif_505_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_507_exp, _vecif_507_s, _vecif_505_s)
					_vecif_505__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_507_exp, _vecif_507__br_flag_188, _vecif_505__br_flag_188)
				_vecif_508_exp = h_not_t_n(_vecif_505__br_flag_188)
				if any_ifexp_true_t_n(_vecif_508_exp):
					_vecif_508_s = h_copy_t_rmRes(_vecif_505_s)
					_vecif_508__br_flag_188 = _vecif_505__br_flag_188
					_vecif_509_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_508_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_509_exp):
						_vecif_509__br_flag_188 = _vecif_508__br_flag_188
						_vecif_509__br_flag_188 = h_broadcast_t_b_b(_vecif_509__br_flag_188, True)
						_vecif_508__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_509_exp, _vecif_509__br_flag_188, _vecif_508__br_flag_188)
					_vecif_510_exp = h_not_t_n(_vecif_508__br_flag_188)
					if any_ifexp_true_t_n(_vecif_510_exp):
						_vecif_510_s = h_copy_t_rmRes(_vecif_508_s)
						rmRes_p_set(_vecif_510_s, h_add_t_vf_t_vf(rmRes_p(_vecif_510_s), h_mul_t_f_t_vf(_vecif_505_d, r)))
						_vecif_508_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_510_exp, _vecif_510_s, _vecif_508_s)
					_vecif_505_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_508_exp, _vecif_508_s, _vecif_505_s)
					_vecif_505__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_508_exp, _vecif_508__br_flag_188, _vecif_505__br_flag_188)
				_vecif_504_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_505_exp_0, _vecif_505_s, _vecif_504_s)
				_vecif_504_d = h_where_t_n_t_n_t_n(_vecif_505_exp_0, _vecif_505_d, _vecif_504_d)
				_vecif_504_m = h_where_t_n_t_n_t_n(_vecif_505_exp_0, _vecif_505_m, _vecif_504_m)
				_vecif_504__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_505_exp_0, _vecif_505__br_flag_188, _vecif_504__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_504_exp, _vecif_504_s, s)
			d = h_where_n_t_n_t_n(_vecif_504_exp, _vecif_504_d, d)
			m = h_where_n_t_n_t_n(_vecif_504_exp, _vecif_504_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_504_exp, _vecif_504__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_511_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_511_exp):
			_vecif_511_s = h_copy_t_rmRes(s)
			_vecif_511_d = d
			_vecif_511_m = m
			_vecif_511__br_flag_188 = _br_flag_188
			_vecif_512_exp_0 = h_not_t_n(_vecif_511__br_flag_188)
			if any_ifexp_true_t_n(_vecif_512_exp_0):
				_vecif_512_s = h_copy_t_rmRes(_vecif_511_s)
				_vecif_512_d = _vecif_511_d
				_vecif_512_m = _vecif_511_m
				_vecif_512__br_flag_188 = _vecif_511__br_flag_188
				_vecif_512_d = tuple_get_retval((_call_ret_513 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_512_s)), _vecif_512_m), _vecif_512_m := tuple_get_outparam(_call_ret_513, 1)))
				_vecif_514_exp = h_less_than_t_n_n(_vecif_512_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_514_exp):
					_vecif_514_s = h_copy_t_rmRes(_vecif_512_s)
					_vecif_514__br_flag_188 = _vecif_512__br_flag_188
					rmRes_h_set(_vecif_514_s, h_broadcast_t_b_b(rmRes_h(_vecif_514_s), True))
					_vecif_514__br_flag_188 = h_broadcast_t_b_b(_vecif_514__br_flag_188, True)
					_vecif_512_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_514_exp, _vecif_514_s, _vecif_512_s)
					_vecif_512__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_514_exp, _vecif_514__br_flag_188, _vecif_512__br_flag_188)
				_vecif_515_exp = h_not_t_n(_vecif_512__br_flag_188)
				if any_ifexp_true_t_n(_vecif_515_exp):
					_vecif_515_s = h_copy_t_rmRes(_vecif_512_s)
					_vecif_515__br_flag_188 = _vecif_512__br_flag_188
					_vecif_516_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_515_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_516_exp):
						_vecif_516__br_flag_188 = _vecif_515__br_flag_188
						_vecif_516__br_flag_188 = h_broadcast_t_b_b(_vecif_516__br_flag_188, True)
						_vecif_515__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_516_exp, _vecif_516__br_flag_188, _vecif_515__br_flag_188)
					_vecif_517_exp = h_not_t_n(_vecif_515__br_flag_188)
					if any_ifexp_true_t_n(_vecif_517_exp):
						_vecif_517_s = h_copy_t_rmRes(_vecif_515_s)
						rmRes_p_set(_vecif_517_s, h_add_t_vf_t_vf(rmRes_p(_vecif_517_s), h_mul_t_f_t_vf(_vecif_512_d, r)))
						_vecif_515_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_517_exp, _vecif_517_s, _vecif_515_s)
					_vecif_512_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_515_exp, _vecif_515_s, _vecif_512_s)
					_vecif_512__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_515_exp, _vecif_515__br_flag_188, _vecif_512__br_flag_188)
				_vecif_511_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_512_exp_0, _vecif_512_s, _vecif_511_s)
				_vecif_511_d = h_where_t_n_t_n_t_n(_vecif_512_exp_0, _vecif_512_d, _vecif_511_d)
				_vecif_511_m = h_where_t_n_t_n_t_n(_vecif_512_exp_0, _vecif_512_m, _vecif_511_m)
				_vecif_511__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_512_exp_0, _vecif_512__br_flag_188, _vecif_511__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_511_exp, _vecif_511_s, s)
			d = h_where_n_t_n_t_n(_vecif_511_exp, _vecif_511_d, d)
			m = h_where_n_t_n_t_n(_vecif_511_exp, _vecif_511_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_511_exp, _vecif_511__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_518_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_518_exp):
			_vecif_518_s = h_copy_t_rmRes(s)
			_vecif_518_d = d
			_vecif_518_m = m
			_vecif_518__br_flag_188 = _br_flag_188
			_vecif_519_exp_0 = h_not_t_n(_vecif_518__br_flag_188)
			if any_ifexp_true_t_n(_vecif_519_exp_0):
				_vecif_519_s = h_copy_t_rmRes(_vecif_518_s)
				_vecif_519_d = _vecif_518_d
				_vecif_519_m = _vecif_518_m
				_vecif_519__br_flag_188 = _vecif_518__br_flag_188
				_vecif_519_d = tuple_get_retval((_call_ret_520 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_519_s)), _vecif_519_m), _vecif_519_m := tuple_get_outparam(_call_ret_520, 1)))
				_vecif_521_exp = h_less_than_t_n_n(_vecif_519_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_521_exp):
					_vecif_521_s = h_copy_t_rmRes(_vecif_519_s)
					_vecif_521__br_flag_188 = _vecif_519__br_flag_188
					rmRes_h_set(_vecif_521_s, h_broadcast_t_b_b(rmRes_h(_vecif_521_s), True))
					_vecif_521__br_flag_188 = h_broadcast_t_b_b(_vecif_521__br_flag_188, True)
					_vecif_519_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_521_exp, _vecif_521_s, _vecif_519_s)
					_vecif_519__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_521_exp, _vecif_521__br_flag_188, _vecif_519__br_flag_188)
				_vecif_522_exp = h_not_t_n(_vecif_519__br_flag_188)
				if any_ifexp_true_t_n(_vecif_522_exp):
					_vecif_522_s = h_copy_t_rmRes(_vecif_519_s)
					_vecif_522__br_flag_188 = _vecif_519__br_flag_188
					_vecif_523_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_522_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_523_exp):
						_vecif_523__br_flag_188 = _vecif_522__br_flag_188
						_vecif_523__br_flag_188 = h_broadcast_t_b_b(_vecif_523__br_flag_188, True)
						_vecif_522__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_523_exp, _vecif_523__br_flag_188, _vecif_522__br_flag_188)
					_vecif_524_exp = h_not_t_n(_vecif_522__br_flag_188)
					if any_ifexp_true_t_n(_vecif_524_exp):
						_vecif_524_s = h_copy_t_rmRes(_vecif_522_s)
						rmRes_p_set(_vecif_524_s, h_add_t_vf_t_vf(rmRes_p(_vecif_524_s), h_mul_t_f_t_vf(_vecif_519_d, r)))
						_vecif_522_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_524_exp, _vecif_524_s, _vecif_522_s)
					_vecif_519_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_522_exp, _vecif_522_s, _vecif_519_s)
					_vecif_519__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_522_exp, _vecif_522__br_flag_188, _vecif_519__br_flag_188)
				_vecif_518_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_519_exp_0, _vecif_519_s, _vecif_518_s)
				_vecif_518_d = h_where_t_n_t_n_t_n(_vecif_519_exp_0, _vecif_519_d, _vecif_518_d)
				_vecif_518_m = h_where_t_n_t_n_t_n(_vecif_519_exp_0, _vecif_519_m, _vecif_518_m)
				_vecif_518__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_519_exp_0, _vecif_519__br_flag_188, _vecif_518__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_518_exp, _vecif_518_s, s)
			d = h_where_n_t_n_t_n(_vecif_518_exp, _vecif_518_d, d)
			m = h_where_n_t_n_t_n(_vecif_518_exp, _vecif_518_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_518_exp, _vecif_518__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_525_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_525_exp):
			_vecif_525_s = h_copy_t_rmRes(s)
			_vecif_525_d = d
			_vecif_525_m = m
			_vecif_525__br_flag_188 = _br_flag_188
			_vecif_526_exp_0 = h_not_t_n(_vecif_525__br_flag_188)
			if any_ifexp_true_t_n(_vecif_526_exp_0):
				_vecif_526_s = h_copy_t_rmRes(_vecif_525_s)
				_vecif_526_d = _vecif_525_d
				_vecif_526_m = _vecif_525_m
				_vecif_526__br_flag_188 = _vecif_525__br_flag_188
				_vecif_526_d = tuple_get_retval((_call_ret_527 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_526_s)), _vecif_526_m), _vecif_526_m := tuple_get_outparam(_call_ret_527, 1)))
				_vecif_528_exp = h_less_than_t_n_n(_vecif_526_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_528_exp):
					_vecif_528_s = h_copy_t_rmRes(_vecif_526_s)
					_vecif_528__br_flag_188 = _vecif_526__br_flag_188
					rmRes_h_set(_vecif_528_s, h_broadcast_t_b_b(rmRes_h(_vecif_528_s), True))
					_vecif_528__br_flag_188 = h_broadcast_t_b_b(_vecif_528__br_flag_188, True)
					_vecif_526_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_528_exp, _vecif_528_s, _vecif_526_s)
					_vecif_526__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_528_exp, _vecif_528__br_flag_188, _vecif_526__br_flag_188)
				_vecif_529_exp = h_not_t_n(_vecif_526__br_flag_188)
				if any_ifexp_true_t_n(_vecif_529_exp):
					_vecif_529_s = h_copy_t_rmRes(_vecif_526_s)
					_vecif_529__br_flag_188 = _vecif_526__br_flag_188
					_vecif_530_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_529_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_530_exp):
						_vecif_530__br_flag_188 = _vecif_529__br_flag_188
						_vecif_530__br_flag_188 = h_broadcast_t_b_b(_vecif_530__br_flag_188, True)
						_vecif_529__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_530_exp, _vecif_530__br_flag_188, _vecif_529__br_flag_188)
					_vecif_531_exp = h_not_t_n(_vecif_529__br_flag_188)
					if any_ifexp_true_t_n(_vecif_531_exp):
						_vecif_531_s = h_copy_t_rmRes(_vecif_529_s)
						rmRes_p_set(_vecif_531_s, h_add_t_vf_t_vf(rmRes_p(_vecif_531_s), h_mul_t_f_t_vf(_vecif_526_d, r)))
						_vecif_529_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_531_exp, _vecif_531_s, _vecif_529_s)
					_vecif_526_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_529_exp, _vecif_529_s, _vecif_526_s)
					_vecif_526__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_529_exp, _vecif_529__br_flag_188, _vecif_526__br_flag_188)
				_vecif_525_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_526_exp_0, _vecif_526_s, _vecif_525_s)
				_vecif_525_d = h_where_t_n_t_n_t_n(_vecif_526_exp_0, _vecif_526_d, _vecif_525_d)
				_vecif_525_m = h_where_t_n_t_n_t_n(_vecif_526_exp_0, _vecif_526_m, _vecif_525_m)
				_vecif_525__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_526_exp_0, _vecif_526__br_flag_188, _vecif_525__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_525_exp, _vecif_525_s, s)
			d = h_where_n_t_n_t_n(_vecif_525_exp, _vecif_525_d, d)
			m = h_where_n_t_n_t_n(_vecif_525_exp, _vecif_525_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_525_exp, _vecif_525__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_532_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_532_exp):
			_vecif_532_s = h_copy_t_rmRes(s)
			_vecif_532_d = d
			_vecif_532_m = m
			_vecif_532__br_flag_188 = _br_flag_188
			_vecif_533_exp_0 = h_not_t_n(_vecif_532__br_flag_188)
			if any_ifexp_true_t_n(_vecif_533_exp_0):
				_vecif_533_s = h_copy_t_rmRes(_vecif_532_s)
				_vecif_533_d = _vecif_532_d
				_vecif_533_m = _vecif_532_m
				_vecif_533__br_flag_188 = _vecif_532__br_flag_188
				_vecif_533_d = tuple_get_retval((_call_ret_534 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_533_s)), _vecif_533_m), _vecif_533_m := tuple_get_outparam(_call_ret_534, 1)))
				_vecif_535_exp = h_less_than_t_n_n(_vecif_533_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_535_exp):
					_vecif_535_s = h_copy_t_rmRes(_vecif_533_s)
					_vecif_535__br_flag_188 = _vecif_533__br_flag_188
					rmRes_h_set(_vecif_535_s, h_broadcast_t_b_b(rmRes_h(_vecif_535_s), True))
					_vecif_535__br_flag_188 = h_broadcast_t_b_b(_vecif_535__br_flag_188, True)
					_vecif_533_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_535_exp, _vecif_535_s, _vecif_533_s)
					_vecif_533__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_535_exp, _vecif_535__br_flag_188, _vecif_533__br_flag_188)
				_vecif_536_exp = h_not_t_n(_vecif_533__br_flag_188)
				if any_ifexp_true_t_n(_vecif_536_exp):
					_vecif_536_s = h_copy_t_rmRes(_vecif_533_s)
					_vecif_536__br_flag_188 = _vecif_533__br_flag_188
					_vecif_537_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_536_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_537_exp):
						_vecif_537__br_flag_188 = _vecif_536__br_flag_188
						_vecif_537__br_flag_188 = h_broadcast_t_b_b(_vecif_537__br_flag_188, True)
						_vecif_536__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_537_exp, _vecif_537__br_flag_188, _vecif_536__br_flag_188)
					_vecif_538_exp = h_not_t_n(_vecif_536__br_flag_188)
					if any_ifexp_true_t_n(_vecif_538_exp):
						_vecif_538_s = h_copy_t_rmRes(_vecif_536_s)
						rmRes_p_set(_vecif_538_s, h_add_t_vf_t_vf(rmRes_p(_vecif_538_s), h_mul_t_f_t_vf(_vecif_533_d, r)))
						_vecif_536_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_538_exp, _vecif_538_s, _vecif_536_s)
					_vecif_533_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_536_exp, _vecif_536_s, _vecif_533_s)
					_vecif_533__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_536_exp, _vecif_536__br_flag_188, _vecif_533__br_flag_188)
				_vecif_532_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_533_exp_0, _vecif_533_s, _vecif_532_s)
				_vecif_532_d = h_where_t_n_t_n_t_n(_vecif_533_exp_0, _vecif_533_d, _vecif_532_d)
				_vecif_532_m = h_where_t_n_t_n_t_n(_vecif_533_exp_0, _vecif_533_m, _vecif_532_m)
				_vecif_532__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_533_exp_0, _vecif_533__br_flag_188, _vecif_532__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_532_exp, _vecif_532_s, s)
			d = h_where_n_t_n_t_n(_vecif_532_exp, _vecif_532_d, d)
			m = h_where_n_t_n_t_n(_vecif_532_exp, _vecif_532_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_532_exp, _vecif_532__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_539_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_539_exp):
			_vecif_539_s = h_copy_t_rmRes(s)
			_vecif_539_d = d
			_vecif_539_m = m
			_vecif_539__br_flag_188 = _br_flag_188
			_vecif_540_exp_0 = h_not_t_n(_vecif_539__br_flag_188)
			if any_ifexp_true_t_n(_vecif_540_exp_0):
				_vecif_540_s = h_copy_t_rmRes(_vecif_539_s)
				_vecif_540_d = _vecif_539_d
				_vecif_540_m = _vecif_539_m
				_vecif_540__br_flag_188 = _vecif_539__br_flag_188
				_vecif_540_d = tuple_get_retval((_call_ret_541 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_540_s)), _vecif_540_m), _vecif_540_m := tuple_get_outparam(_call_ret_541, 1)))
				_vecif_542_exp = h_less_than_t_n_n(_vecif_540_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_542_exp):
					_vecif_542_s = h_copy_t_rmRes(_vecif_540_s)
					_vecif_542__br_flag_188 = _vecif_540__br_flag_188
					rmRes_h_set(_vecif_542_s, h_broadcast_t_b_b(rmRes_h(_vecif_542_s), True))
					_vecif_542__br_flag_188 = h_broadcast_t_b_b(_vecif_542__br_flag_188, True)
					_vecif_540_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_542_exp, _vecif_542_s, _vecif_540_s)
					_vecif_540__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_542_exp, _vecif_542__br_flag_188, _vecif_540__br_flag_188)
				_vecif_543_exp = h_not_t_n(_vecif_540__br_flag_188)
				if any_ifexp_true_t_n(_vecif_543_exp):
					_vecif_543_s = h_copy_t_rmRes(_vecif_540_s)
					_vecif_543__br_flag_188 = _vecif_540__br_flag_188
					_vecif_544_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_543_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_544_exp):
						_vecif_544__br_flag_188 = _vecif_543__br_flag_188
						_vecif_544__br_flag_188 = h_broadcast_t_b_b(_vecif_544__br_flag_188, True)
						_vecif_543__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_544_exp, _vecif_544__br_flag_188, _vecif_543__br_flag_188)
					_vecif_545_exp = h_not_t_n(_vecif_543__br_flag_188)
					if any_ifexp_true_t_n(_vecif_545_exp):
						_vecif_545_s = h_copy_t_rmRes(_vecif_543_s)
						rmRes_p_set(_vecif_545_s, h_add_t_vf_t_vf(rmRes_p(_vecif_545_s), h_mul_t_f_t_vf(_vecif_540_d, r)))
						_vecif_543_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_545_exp, _vecif_545_s, _vecif_543_s)
					_vecif_540_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_543_exp, _vecif_543_s, _vecif_540_s)
					_vecif_540__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_543_exp, _vecif_543__br_flag_188, _vecif_540__br_flag_188)
				_vecif_539_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_540_exp_0, _vecif_540_s, _vecif_539_s)
				_vecif_539_d = h_where_t_n_t_n_t_n(_vecif_540_exp_0, _vecif_540_d, _vecif_539_d)
				_vecif_539_m = h_where_t_n_t_n_t_n(_vecif_540_exp_0, _vecif_540_m, _vecif_539_m)
				_vecif_539__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_540_exp_0, _vecif_540__br_flag_188, _vecif_539__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_539_exp, _vecif_539_s, s)
			d = h_where_n_t_n_t_n(_vecif_539_exp, _vecif_539_d, d)
			m = h_where_n_t_n_t_n(_vecif_539_exp, _vecif_539_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_539_exp, _vecif_539__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_546_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_546_exp):
			_vecif_546_s = h_copy_t_rmRes(s)
			_vecif_546_d = d
			_vecif_546_m = m
			_vecif_546__br_flag_188 = _br_flag_188
			_vecif_547_exp_0 = h_not_t_n(_vecif_546__br_flag_188)
			if any_ifexp_true_t_n(_vecif_547_exp_0):
				_vecif_547_s = h_copy_t_rmRes(_vecif_546_s)
				_vecif_547_d = _vecif_546_d
				_vecif_547_m = _vecif_546_m
				_vecif_547__br_flag_188 = _vecif_546__br_flag_188
				_vecif_547_d = tuple_get_retval((_call_ret_548 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_547_s)), _vecif_547_m), _vecif_547_m := tuple_get_outparam(_call_ret_548, 1)))
				_vecif_549_exp = h_less_than_t_n_n(_vecif_547_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_549_exp):
					_vecif_549_s = h_copy_t_rmRes(_vecif_547_s)
					_vecif_549__br_flag_188 = _vecif_547__br_flag_188
					rmRes_h_set(_vecif_549_s, h_broadcast_t_b_b(rmRes_h(_vecif_549_s), True))
					_vecif_549__br_flag_188 = h_broadcast_t_b_b(_vecif_549__br_flag_188, True)
					_vecif_547_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_549_exp, _vecif_549_s, _vecif_547_s)
					_vecif_547__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_549_exp, _vecif_549__br_flag_188, _vecif_547__br_flag_188)
				_vecif_550_exp = h_not_t_n(_vecif_547__br_flag_188)
				if any_ifexp_true_t_n(_vecif_550_exp):
					_vecif_550_s = h_copy_t_rmRes(_vecif_547_s)
					_vecif_550__br_flag_188 = _vecif_547__br_flag_188
					_vecif_551_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_550_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_551_exp):
						_vecif_551__br_flag_188 = _vecif_550__br_flag_188
						_vecif_551__br_flag_188 = h_broadcast_t_b_b(_vecif_551__br_flag_188, True)
						_vecif_550__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_551_exp, _vecif_551__br_flag_188, _vecif_550__br_flag_188)
					_vecif_552_exp = h_not_t_n(_vecif_550__br_flag_188)
					if any_ifexp_true_t_n(_vecif_552_exp):
						_vecif_552_s = h_copy_t_rmRes(_vecif_550_s)
						rmRes_p_set(_vecif_552_s, h_add_t_vf_t_vf(rmRes_p(_vecif_552_s), h_mul_t_f_t_vf(_vecif_547_d, r)))
						_vecif_550_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_552_exp, _vecif_552_s, _vecif_550_s)
					_vecif_547_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_550_exp, _vecif_550_s, _vecif_547_s)
					_vecif_547__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_550_exp, _vecif_550__br_flag_188, _vecif_547__br_flag_188)
				_vecif_546_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_547_exp_0, _vecif_547_s, _vecif_546_s)
				_vecif_546_d = h_where_t_n_t_n_t_n(_vecif_547_exp_0, _vecif_547_d, _vecif_546_d)
				_vecif_546_m = h_where_t_n_t_n_t_n(_vecif_547_exp_0, _vecif_547_m, _vecif_546_m)
				_vecif_546__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_547_exp_0, _vecif_547__br_flag_188, _vecif_546__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_546_exp, _vecif_546_s, s)
			d = h_where_n_t_n_t_n(_vecif_546_exp, _vecif_546_d, d)
			m = h_where_n_t_n_t_n(_vecif_546_exp, _vecif_546_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_546_exp, _vecif_546__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_553_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_553_exp):
			_vecif_553_s = h_copy_t_rmRes(s)
			_vecif_553_d = d
			_vecif_553_m = m
			_vecif_553__br_flag_188 = _br_flag_188
			_vecif_554_exp_0 = h_not_t_n(_vecif_553__br_flag_188)
			if any_ifexp_true_t_n(_vecif_554_exp_0):
				_vecif_554_s = h_copy_t_rmRes(_vecif_553_s)
				_vecif_554_d = _vecif_553_d
				_vecif_554_m = _vecif_553_m
				_vecif_554__br_flag_188 = _vecif_553__br_flag_188
				_vecif_554_d = tuple_get_retval((_call_ret_555 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_554_s)), _vecif_554_m), _vecif_554_m := tuple_get_outparam(_call_ret_555, 1)))
				_vecif_556_exp = h_less_than_t_n_n(_vecif_554_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_556_exp):
					_vecif_556_s = h_copy_t_rmRes(_vecif_554_s)
					_vecif_556__br_flag_188 = _vecif_554__br_flag_188
					rmRes_h_set(_vecif_556_s, h_broadcast_t_b_b(rmRes_h(_vecif_556_s), True))
					_vecif_556__br_flag_188 = h_broadcast_t_b_b(_vecif_556__br_flag_188, True)
					_vecif_554_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_556_exp, _vecif_556_s, _vecif_554_s)
					_vecif_554__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_556_exp, _vecif_556__br_flag_188, _vecif_554__br_flag_188)
				_vecif_557_exp = h_not_t_n(_vecif_554__br_flag_188)
				if any_ifexp_true_t_n(_vecif_557_exp):
					_vecif_557_s = h_copy_t_rmRes(_vecif_554_s)
					_vecif_557__br_flag_188 = _vecif_554__br_flag_188
					_vecif_558_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_557_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_558_exp):
						_vecif_558__br_flag_188 = _vecif_557__br_flag_188
						_vecif_558__br_flag_188 = h_broadcast_t_b_b(_vecif_558__br_flag_188, True)
						_vecif_557__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_558_exp, _vecif_558__br_flag_188, _vecif_557__br_flag_188)
					_vecif_559_exp = h_not_t_n(_vecif_557__br_flag_188)
					if any_ifexp_true_t_n(_vecif_559_exp):
						_vecif_559_s = h_copy_t_rmRes(_vecif_557_s)
						rmRes_p_set(_vecif_559_s, h_add_t_vf_t_vf(rmRes_p(_vecif_559_s), h_mul_t_f_t_vf(_vecif_554_d, r)))
						_vecif_557_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_559_exp, _vecif_559_s, _vecif_557_s)
					_vecif_554_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_557_exp, _vecif_557_s, _vecif_554_s)
					_vecif_554__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_557_exp, _vecif_557__br_flag_188, _vecif_554__br_flag_188)
				_vecif_553_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_554_exp_0, _vecif_554_s, _vecif_553_s)
				_vecif_553_d = h_where_t_n_t_n_t_n(_vecif_554_exp_0, _vecif_554_d, _vecif_553_d)
				_vecif_553_m = h_where_t_n_t_n_t_n(_vecif_554_exp_0, _vecif_554_m, _vecif_553_m)
				_vecif_553__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_554_exp_0, _vecif_554__br_flag_188, _vecif_553__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_553_exp, _vecif_553_s, s)
			d = h_where_n_t_n_t_n(_vecif_553_exp, _vecif_553_d, d)
			m = h_where_n_t_n_t_n(_vecif_553_exp, _vecif_553_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_553_exp, _vecif_553__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_560_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_560_exp):
			_vecif_560_s = h_copy_t_rmRes(s)
			_vecif_560_d = d
			_vecif_560_m = m
			_vecif_560__br_flag_188 = _br_flag_188
			_vecif_561_exp_0 = h_not_t_n(_vecif_560__br_flag_188)
			if any_ifexp_true_t_n(_vecif_561_exp_0):
				_vecif_561_s = h_copy_t_rmRes(_vecif_560_s)
				_vecif_561_d = _vecif_560_d
				_vecif_561_m = _vecif_560_m
				_vecif_561__br_flag_188 = _vecif_560__br_flag_188
				_vecif_561_d = tuple_get_retval((_call_ret_562 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_561_s)), _vecif_561_m), _vecif_561_m := tuple_get_outparam(_call_ret_562, 1)))
				_vecif_563_exp = h_less_than_t_n_n(_vecif_561_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_563_exp):
					_vecif_563_s = h_copy_t_rmRes(_vecif_561_s)
					_vecif_563__br_flag_188 = _vecif_561__br_flag_188
					rmRes_h_set(_vecif_563_s, h_broadcast_t_b_b(rmRes_h(_vecif_563_s), True))
					_vecif_563__br_flag_188 = h_broadcast_t_b_b(_vecif_563__br_flag_188, True)
					_vecif_561_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_563_exp, _vecif_563_s, _vecif_561_s)
					_vecif_561__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_563_exp, _vecif_563__br_flag_188, _vecif_561__br_flag_188)
				_vecif_564_exp = h_not_t_n(_vecif_561__br_flag_188)
				if any_ifexp_true_t_n(_vecif_564_exp):
					_vecif_564_s = h_copy_t_rmRes(_vecif_561_s)
					_vecif_564__br_flag_188 = _vecif_561__br_flag_188
					_vecif_565_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_564_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_565_exp):
						_vecif_565__br_flag_188 = _vecif_564__br_flag_188
						_vecif_565__br_flag_188 = h_broadcast_t_b_b(_vecif_565__br_flag_188, True)
						_vecif_564__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_565_exp, _vecif_565__br_flag_188, _vecif_564__br_flag_188)
					_vecif_566_exp = h_not_t_n(_vecif_564__br_flag_188)
					if any_ifexp_true_t_n(_vecif_566_exp):
						_vecif_566_s = h_copy_t_rmRes(_vecif_564_s)
						rmRes_p_set(_vecif_566_s, h_add_t_vf_t_vf(rmRes_p(_vecif_566_s), h_mul_t_f_t_vf(_vecif_561_d, r)))
						_vecif_564_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_566_exp, _vecif_566_s, _vecif_564_s)
					_vecif_561_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_564_exp, _vecif_564_s, _vecif_561_s)
					_vecif_561__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_564_exp, _vecif_564__br_flag_188, _vecif_561__br_flag_188)
				_vecif_560_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_561_exp_0, _vecif_561_s, _vecif_560_s)
				_vecif_560_d = h_where_t_n_t_n_t_n(_vecif_561_exp_0, _vecif_561_d, _vecif_560_d)
				_vecif_560_m = h_where_t_n_t_n_t_n(_vecif_561_exp_0, _vecif_561_m, _vecif_560_m)
				_vecif_560__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_561_exp_0, _vecif_561__br_flag_188, _vecif_560__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_560_exp, _vecif_560_s, s)
			d = h_where_n_t_n_t_n(_vecif_560_exp, _vecif_560_d, d)
			m = h_where_n_t_n_t_n(_vecif_560_exp, _vecif_560_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_560_exp, _vecif_560__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_567_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_567_exp):
			_vecif_567_s = h_copy_t_rmRes(s)
			_vecif_567_d = d
			_vecif_567_m = m
			_vecif_567__br_flag_188 = _br_flag_188
			_vecif_568_exp_0 = h_not_t_n(_vecif_567__br_flag_188)
			if any_ifexp_true_t_n(_vecif_568_exp_0):
				_vecif_568_s = h_copy_t_rmRes(_vecif_567_s)
				_vecif_568_d = _vecif_567_d
				_vecif_568_m = _vecif_567_m
				_vecif_568__br_flag_188 = _vecif_567__br_flag_188
				_vecif_568_d = tuple_get_retval((_call_ret_569 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_568_s)), _vecif_568_m), _vecif_568_m := tuple_get_outparam(_call_ret_569, 1)))
				_vecif_570_exp = h_less_than_t_n_n(_vecif_568_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_570_exp):
					_vecif_570_s = h_copy_t_rmRes(_vecif_568_s)
					_vecif_570__br_flag_188 = _vecif_568__br_flag_188
					rmRes_h_set(_vecif_570_s, h_broadcast_t_b_b(rmRes_h(_vecif_570_s), True))
					_vecif_570__br_flag_188 = h_broadcast_t_b_b(_vecif_570__br_flag_188, True)
					_vecif_568_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_570_exp, _vecif_570_s, _vecif_568_s)
					_vecif_568__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_570_exp, _vecif_570__br_flag_188, _vecif_568__br_flag_188)
				_vecif_571_exp = h_not_t_n(_vecif_568__br_flag_188)
				if any_ifexp_true_t_n(_vecif_571_exp):
					_vecif_571_s = h_copy_t_rmRes(_vecif_568_s)
					_vecif_571__br_flag_188 = _vecif_568__br_flag_188
					_vecif_572_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_571_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_572_exp):
						_vecif_572__br_flag_188 = _vecif_571__br_flag_188
						_vecif_572__br_flag_188 = h_broadcast_t_b_b(_vecif_572__br_flag_188, True)
						_vecif_571__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_572_exp, _vecif_572__br_flag_188, _vecif_571__br_flag_188)
					_vecif_573_exp = h_not_t_n(_vecif_571__br_flag_188)
					if any_ifexp_true_t_n(_vecif_573_exp):
						_vecif_573_s = h_copy_t_rmRes(_vecif_571_s)
						rmRes_p_set(_vecif_573_s, h_add_t_vf_t_vf(rmRes_p(_vecif_573_s), h_mul_t_f_t_vf(_vecif_568_d, r)))
						_vecif_571_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_573_exp, _vecif_573_s, _vecif_571_s)
					_vecif_568_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_571_exp, _vecif_571_s, _vecif_568_s)
					_vecif_568__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_571_exp, _vecif_571__br_flag_188, _vecif_568__br_flag_188)
				_vecif_567_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_568_exp_0, _vecif_568_s, _vecif_567_s)
				_vecif_567_d = h_where_t_n_t_n_t_n(_vecif_568_exp_0, _vecif_568_d, _vecif_567_d)
				_vecif_567_m = h_where_t_n_t_n_t_n(_vecif_568_exp_0, _vecif_568_m, _vecif_567_m)
				_vecif_567__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_568_exp_0, _vecif_568__br_flag_188, _vecif_567__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_567_exp, _vecif_567_s, s)
			d = h_where_n_t_n_t_n(_vecif_567_exp, _vecif_567_d, d)
			m = h_where_n_t_n_t_n(_vecif_567_exp, _vecif_567_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_567_exp, _vecif_567__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_574_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_574_exp):
			_vecif_574_s = h_copy_t_rmRes(s)
			_vecif_574_d = d
			_vecif_574_m = m
			_vecif_574__br_flag_188 = _br_flag_188
			_vecif_575_exp_0 = h_not_t_n(_vecif_574__br_flag_188)
			if any_ifexp_true_t_n(_vecif_575_exp_0):
				_vecif_575_s = h_copy_t_rmRes(_vecif_574_s)
				_vecif_575_d = _vecif_574_d
				_vecif_575_m = _vecif_574_m
				_vecif_575__br_flag_188 = _vecif_574__br_flag_188
				_vecif_575_d = tuple_get_retval((_call_ret_576 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_575_s)), _vecif_575_m), _vecif_575_m := tuple_get_outparam(_call_ret_576, 1)))
				_vecif_577_exp = h_less_than_t_n_n(_vecif_575_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_577_exp):
					_vecif_577_s = h_copy_t_rmRes(_vecif_575_s)
					_vecif_577__br_flag_188 = _vecif_575__br_flag_188
					rmRes_h_set(_vecif_577_s, h_broadcast_t_b_b(rmRes_h(_vecif_577_s), True))
					_vecif_577__br_flag_188 = h_broadcast_t_b_b(_vecif_577__br_flag_188, True)
					_vecif_575_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_577_exp, _vecif_577_s, _vecif_575_s)
					_vecif_575__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_577_exp, _vecif_577__br_flag_188, _vecif_575__br_flag_188)
				_vecif_578_exp = h_not_t_n(_vecif_575__br_flag_188)
				if any_ifexp_true_t_n(_vecif_578_exp):
					_vecif_578_s = h_copy_t_rmRes(_vecif_575_s)
					_vecif_578__br_flag_188 = _vecif_575__br_flag_188
					_vecif_579_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_578_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_579_exp):
						_vecif_579__br_flag_188 = _vecif_578__br_flag_188
						_vecif_579__br_flag_188 = h_broadcast_t_b_b(_vecif_579__br_flag_188, True)
						_vecif_578__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_579_exp, _vecif_579__br_flag_188, _vecif_578__br_flag_188)
					_vecif_580_exp = h_not_t_n(_vecif_578__br_flag_188)
					if any_ifexp_true_t_n(_vecif_580_exp):
						_vecif_580_s = h_copy_t_rmRes(_vecif_578_s)
						rmRes_p_set(_vecif_580_s, h_add_t_vf_t_vf(rmRes_p(_vecif_580_s), h_mul_t_f_t_vf(_vecif_575_d, r)))
						_vecif_578_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_580_exp, _vecif_580_s, _vecif_578_s)
					_vecif_575_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_578_exp, _vecif_578_s, _vecif_575_s)
					_vecif_575__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_578_exp, _vecif_578__br_flag_188, _vecif_575__br_flag_188)
				_vecif_574_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_575_exp_0, _vecif_575_s, _vecif_574_s)
				_vecif_574_d = h_where_t_n_t_n_t_n(_vecif_575_exp_0, _vecif_575_d, _vecif_574_d)
				_vecif_574_m = h_where_t_n_t_n_t_n(_vecif_575_exp_0, _vecif_575_m, _vecif_574_m)
				_vecif_574__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_575_exp_0, _vecif_575__br_flag_188, _vecif_574__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_574_exp, _vecif_574_s, s)
			d = h_where_n_t_n_t_n(_vecif_574_exp, _vecif_574_d, d)
			m = h_where_n_t_n_t_n(_vecif_574_exp, _vecif_574_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_574_exp, _vecif_574__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_581_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_581_exp):
			_vecif_581_s = h_copy_t_rmRes(s)
			_vecif_581_d = d
			_vecif_581_m = m
			_vecif_581__br_flag_188 = _br_flag_188
			_vecif_582_exp_0 = h_not_t_n(_vecif_581__br_flag_188)
			if any_ifexp_true_t_n(_vecif_582_exp_0):
				_vecif_582_s = h_copy_t_rmRes(_vecif_581_s)
				_vecif_582_d = _vecif_581_d
				_vecif_582_m = _vecif_581_m
				_vecif_582__br_flag_188 = _vecif_581__br_flag_188
				_vecif_582_d = tuple_get_retval((_call_ret_583 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_582_s)), _vecif_582_m), _vecif_582_m := tuple_get_outparam(_call_ret_583, 1)))
				_vecif_584_exp = h_less_than_t_n_n(_vecif_582_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_584_exp):
					_vecif_584_s = h_copy_t_rmRes(_vecif_582_s)
					_vecif_584__br_flag_188 = _vecif_582__br_flag_188
					rmRes_h_set(_vecif_584_s, h_broadcast_t_b_b(rmRes_h(_vecif_584_s), True))
					_vecif_584__br_flag_188 = h_broadcast_t_b_b(_vecif_584__br_flag_188, True)
					_vecif_582_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_584_exp, _vecif_584_s, _vecif_582_s)
					_vecif_582__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_584_exp, _vecif_584__br_flag_188, _vecif_582__br_flag_188)
				_vecif_585_exp = h_not_t_n(_vecif_582__br_flag_188)
				if any_ifexp_true_t_n(_vecif_585_exp):
					_vecif_585_s = h_copy_t_rmRes(_vecif_582_s)
					_vecif_585__br_flag_188 = _vecif_582__br_flag_188
					_vecif_586_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_585_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_586_exp):
						_vecif_586__br_flag_188 = _vecif_585__br_flag_188
						_vecif_586__br_flag_188 = h_broadcast_t_b_b(_vecif_586__br_flag_188, True)
						_vecif_585__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_586_exp, _vecif_586__br_flag_188, _vecif_585__br_flag_188)
					_vecif_587_exp = h_not_t_n(_vecif_585__br_flag_188)
					if any_ifexp_true_t_n(_vecif_587_exp):
						_vecif_587_s = h_copy_t_rmRes(_vecif_585_s)
						rmRes_p_set(_vecif_587_s, h_add_t_vf_t_vf(rmRes_p(_vecif_587_s), h_mul_t_f_t_vf(_vecif_582_d, r)))
						_vecif_585_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_587_exp, _vecif_587_s, _vecif_585_s)
					_vecif_582_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_585_exp, _vecif_585_s, _vecif_582_s)
					_vecif_582__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_585_exp, _vecif_585__br_flag_188, _vecif_582__br_flag_188)
				_vecif_581_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_582_exp_0, _vecif_582_s, _vecif_581_s)
				_vecif_581_d = h_where_t_n_t_n_t_n(_vecif_582_exp_0, _vecif_582_d, _vecif_581_d)
				_vecif_581_m = h_where_t_n_t_n_t_n(_vecif_582_exp_0, _vecif_582_m, _vecif_581_m)
				_vecif_581__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_582_exp_0, _vecif_582__br_flag_188, _vecif_581__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_581_exp, _vecif_581_s, s)
			d = h_where_n_t_n_t_n(_vecif_581_exp, _vecif_581_d, d)
			m = h_where_n_t_n_t_n(_vecif_581_exp, _vecif_581_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_581_exp, _vecif_581__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_588_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_588_exp):
			_vecif_588_s = h_copy_t_rmRes(s)
			_vecif_588_d = d
			_vecif_588_m = m
			_vecif_588__br_flag_188 = _br_flag_188
			_vecif_589_exp_0 = h_not_t_n(_vecif_588__br_flag_188)
			if any_ifexp_true_t_n(_vecif_589_exp_0):
				_vecif_589_s = h_copy_t_rmRes(_vecif_588_s)
				_vecif_589_d = _vecif_588_d
				_vecif_589_m = _vecif_588_m
				_vecif_589__br_flag_188 = _vecif_588__br_flag_188
				_vecif_589_d = tuple_get_retval((_call_ret_590 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_589_s)), _vecif_589_m), _vecif_589_m := tuple_get_outparam(_call_ret_590, 1)))
				_vecif_591_exp = h_less_than_t_n_n(_vecif_589_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_591_exp):
					_vecif_591_s = h_copy_t_rmRes(_vecif_589_s)
					_vecif_591__br_flag_188 = _vecif_589__br_flag_188
					rmRes_h_set(_vecif_591_s, h_broadcast_t_b_b(rmRes_h(_vecif_591_s), True))
					_vecif_591__br_flag_188 = h_broadcast_t_b_b(_vecif_591__br_flag_188, True)
					_vecif_589_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_591_exp, _vecif_591_s, _vecif_589_s)
					_vecif_589__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_591_exp, _vecif_591__br_flag_188, _vecif_589__br_flag_188)
				_vecif_592_exp = h_not_t_n(_vecif_589__br_flag_188)
				if any_ifexp_true_t_n(_vecif_592_exp):
					_vecif_592_s = h_copy_t_rmRes(_vecif_589_s)
					_vecif_592__br_flag_188 = _vecif_589__br_flag_188
					_vecif_593_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_592_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_593_exp):
						_vecif_593__br_flag_188 = _vecif_592__br_flag_188
						_vecif_593__br_flag_188 = h_broadcast_t_b_b(_vecif_593__br_flag_188, True)
						_vecif_592__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_593_exp, _vecif_593__br_flag_188, _vecif_592__br_flag_188)
					_vecif_594_exp = h_not_t_n(_vecif_592__br_flag_188)
					if any_ifexp_true_t_n(_vecif_594_exp):
						_vecif_594_s = h_copy_t_rmRes(_vecif_592_s)
						rmRes_p_set(_vecif_594_s, h_add_t_vf_t_vf(rmRes_p(_vecif_594_s), h_mul_t_f_t_vf(_vecif_589_d, r)))
						_vecif_592_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_594_exp, _vecif_594_s, _vecif_592_s)
					_vecif_589_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_592_exp, _vecif_592_s, _vecif_589_s)
					_vecif_589__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_592_exp, _vecif_592__br_flag_188, _vecif_589__br_flag_188)
				_vecif_588_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_589_exp_0, _vecif_589_s, _vecif_588_s)
				_vecif_588_d = h_where_t_n_t_n_t_n(_vecif_589_exp_0, _vecif_589_d, _vecif_588_d)
				_vecif_588_m = h_where_t_n_t_n_t_n(_vecif_589_exp_0, _vecif_589_m, _vecif_588_m)
				_vecif_588__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_589_exp_0, _vecif_589__br_flag_188, _vecif_588__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_588_exp, _vecif_588_s, s)
			d = h_where_n_t_n_t_n(_vecif_588_exp, _vecif_588_d, d)
			m = h_where_n_t_n_t_n(_vecif_588_exp, _vecif_588_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_588_exp, _vecif_588__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_595_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_595_exp):
			_vecif_595_s = h_copy_t_rmRes(s)
			_vecif_595_d = d
			_vecif_595_m = m
			_vecif_595__br_flag_188 = _br_flag_188
			_vecif_596_exp_0 = h_not_t_n(_vecif_595__br_flag_188)
			if any_ifexp_true_t_n(_vecif_596_exp_0):
				_vecif_596_s = h_copy_t_rmRes(_vecif_595_s)
				_vecif_596_d = _vecif_595_d
				_vecif_596_m = _vecif_595_m
				_vecif_596__br_flag_188 = _vecif_595__br_flag_188
				_vecif_596_d = tuple_get_retval((_call_ret_597 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_596_s)), _vecif_596_m), _vecif_596_m := tuple_get_outparam(_call_ret_597, 1)))
				_vecif_598_exp = h_less_than_t_n_n(_vecif_596_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_598_exp):
					_vecif_598_s = h_copy_t_rmRes(_vecif_596_s)
					_vecif_598__br_flag_188 = _vecif_596__br_flag_188
					rmRes_h_set(_vecif_598_s, h_broadcast_t_b_b(rmRes_h(_vecif_598_s), True))
					_vecif_598__br_flag_188 = h_broadcast_t_b_b(_vecif_598__br_flag_188, True)
					_vecif_596_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_598_exp, _vecif_598_s, _vecif_596_s)
					_vecif_596__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_598_exp, _vecif_598__br_flag_188, _vecif_596__br_flag_188)
				_vecif_599_exp = h_not_t_n(_vecif_596__br_flag_188)
				if any_ifexp_true_t_n(_vecif_599_exp):
					_vecif_599_s = h_copy_t_rmRes(_vecif_596_s)
					_vecif_599__br_flag_188 = _vecif_596__br_flag_188
					_vecif_600_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_599_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_600_exp):
						_vecif_600__br_flag_188 = _vecif_599__br_flag_188
						_vecif_600__br_flag_188 = h_broadcast_t_b_b(_vecif_600__br_flag_188, True)
						_vecif_599__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_600_exp, _vecif_600__br_flag_188, _vecif_599__br_flag_188)
					_vecif_601_exp = h_not_t_n(_vecif_599__br_flag_188)
					if any_ifexp_true_t_n(_vecif_601_exp):
						_vecif_601_s = h_copy_t_rmRes(_vecif_599_s)
						rmRes_p_set(_vecif_601_s, h_add_t_vf_t_vf(rmRes_p(_vecif_601_s), h_mul_t_f_t_vf(_vecif_596_d, r)))
						_vecif_599_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_601_exp, _vecif_601_s, _vecif_599_s)
					_vecif_596_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_599_exp, _vecif_599_s, _vecif_596_s)
					_vecif_596__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_599_exp, _vecif_599__br_flag_188, _vecif_596__br_flag_188)
				_vecif_595_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_596_exp_0, _vecif_596_s, _vecif_595_s)
				_vecif_595_d = h_where_t_n_t_n_t_n(_vecif_596_exp_0, _vecif_596_d, _vecif_595_d)
				_vecif_595_m = h_where_t_n_t_n_t_n(_vecif_596_exp_0, _vecif_596_m, _vecif_595_m)
				_vecif_595__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_596_exp_0, _vecif_596__br_flag_188, _vecif_595__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_595_exp, _vecif_595_s, s)
			d = h_where_n_t_n_t_n(_vecif_595_exp, _vecif_595_d, d)
			m = h_where_n_t_n_t_n(_vecif_595_exp, _vecif_595_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_595_exp, _vecif_595__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_602_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_602_exp):
			_vecif_602_s = h_copy_t_rmRes(s)
			_vecif_602_d = d
			_vecif_602_m = m
			_vecif_602__br_flag_188 = _br_flag_188
			_vecif_603_exp_0 = h_not_t_n(_vecif_602__br_flag_188)
			if any_ifexp_true_t_n(_vecif_603_exp_0):
				_vecif_603_s = h_copy_t_rmRes(_vecif_602_s)
				_vecif_603_d = _vecif_602_d
				_vecif_603_m = _vecif_602_m
				_vecif_603__br_flag_188 = _vecif_602__br_flag_188
				_vecif_603_d = tuple_get_retval((_call_ret_604 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_603_s)), _vecif_603_m), _vecif_603_m := tuple_get_outparam(_call_ret_604, 1)))
				_vecif_605_exp = h_less_than_t_n_n(_vecif_603_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_605_exp):
					_vecif_605_s = h_copy_t_rmRes(_vecif_603_s)
					_vecif_605__br_flag_188 = _vecif_603__br_flag_188
					rmRes_h_set(_vecif_605_s, h_broadcast_t_b_b(rmRes_h(_vecif_605_s), True))
					_vecif_605__br_flag_188 = h_broadcast_t_b_b(_vecif_605__br_flag_188, True)
					_vecif_603_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_605_exp, _vecif_605_s, _vecif_603_s)
					_vecif_603__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_605_exp, _vecif_605__br_flag_188, _vecif_603__br_flag_188)
				_vecif_606_exp = h_not_t_n(_vecif_603__br_flag_188)
				if any_ifexp_true_t_n(_vecif_606_exp):
					_vecif_606_s = h_copy_t_rmRes(_vecif_603_s)
					_vecif_606__br_flag_188 = _vecif_603__br_flag_188
					_vecif_607_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_606_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_607_exp):
						_vecif_607__br_flag_188 = _vecif_606__br_flag_188
						_vecif_607__br_flag_188 = h_broadcast_t_b_b(_vecif_607__br_flag_188, True)
						_vecif_606__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_607_exp, _vecif_607__br_flag_188, _vecif_606__br_flag_188)
					_vecif_608_exp = h_not_t_n(_vecif_606__br_flag_188)
					if any_ifexp_true_t_n(_vecif_608_exp):
						_vecif_608_s = h_copy_t_rmRes(_vecif_606_s)
						rmRes_p_set(_vecif_608_s, h_add_t_vf_t_vf(rmRes_p(_vecif_608_s), h_mul_t_f_t_vf(_vecif_603_d, r)))
						_vecif_606_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_608_exp, _vecif_608_s, _vecif_606_s)
					_vecif_603_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_606_exp, _vecif_606_s, _vecif_603_s)
					_vecif_603__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_606_exp, _vecif_606__br_flag_188, _vecif_603__br_flag_188)
				_vecif_602_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_603_exp_0, _vecif_603_s, _vecif_602_s)
				_vecif_602_d = h_where_t_n_t_n_t_n(_vecif_603_exp_0, _vecif_603_d, _vecif_602_d)
				_vecif_602_m = h_where_t_n_t_n_t_n(_vecif_603_exp_0, _vecif_603_m, _vecif_602_m)
				_vecif_602__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_603_exp_0, _vecif_603__br_flag_188, _vecif_602__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_602_exp, _vecif_602_s, s)
			d = h_where_n_t_n_t_n(_vecif_602_exp, _vecif_602_d, d)
			m = h_where_n_t_n_t_n(_vecif_602_exp, _vecif_602_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_602_exp, _vecif_602__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_609_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_609_exp):
			_vecif_609_s = h_copy_t_rmRes(s)
			_vecif_609_d = d
			_vecif_609_m = m
			_vecif_609__br_flag_188 = _br_flag_188
			_vecif_610_exp_0 = h_not_t_n(_vecif_609__br_flag_188)
			if any_ifexp_true_t_n(_vecif_610_exp_0):
				_vecif_610_s = h_copy_t_rmRes(_vecif_609_s)
				_vecif_610_d = _vecif_609_d
				_vecif_610_m = _vecif_609_m
				_vecif_610__br_flag_188 = _vecif_609__br_flag_188
				_vecif_610_d = tuple_get_retval((_call_ret_611 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_610_s)), _vecif_610_m), _vecif_610_m := tuple_get_outparam(_call_ret_611, 1)))
				_vecif_612_exp = h_less_than_t_n_n(_vecif_610_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_612_exp):
					_vecif_612_s = h_copy_t_rmRes(_vecif_610_s)
					_vecif_612__br_flag_188 = _vecif_610__br_flag_188
					rmRes_h_set(_vecif_612_s, h_broadcast_t_b_b(rmRes_h(_vecif_612_s), True))
					_vecif_612__br_flag_188 = h_broadcast_t_b_b(_vecif_612__br_flag_188, True)
					_vecif_610_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_612_exp, _vecif_612_s, _vecif_610_s)
					_vecif_610__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_612_exp, _vecif_612__br_flag_188, _vecif_610__br_flag_188)
				_vecif_613_exp = h_not_t_n(_vecif_610__br_flag_188)
				if any_ifexp_true_t_n(_vecif_613_exp):
					_vecif_613_s = h_copy_t_rmRes(_vecif_610_s)
					_vecif_613__br_flag_188 = _vecif_610__br_flag_188
					_vecif_614_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_613_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_614_exp):
						_vecif_614__br_flag_188 = _vecif_613__br_flag_188
						_vecif_614__br_flag_188 = h_broadcast_t_b_b(_vecif_614__br_flag_188, True)
						_vecif_613__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_614_exp, _vecif_614__br_flag_188, _vecif_613__br_flag_188)
					_vecif_615_exp = h_not_t_n(_vecif_613__br_flag_188)
					if any_ifexp_true_t_n(_vecif_615_exp):
						_vecif_615_s = h_copy_t_rmRes(_vecif_613_s)
						rmRes_p_set(_vecif_615_s, h_add_t_vf_t_vf(rmRes_p(_vecif_615_s), h_mul_t_f_t_vf(_vecif_610_d, r)))
						_vecif_613_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_615_exp, _vecif_615_s, _vecif_613_s)
					_vecif_610_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_613_exp, _vecif_613_s, _vecif_610_s)
					_vecif_610__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_613_exp, _vecif_613__br_flag_188, _vecif_610__br_flag_188)
				_vecif_609_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_610_exp_0, _vecif_610_s, _vecif_609_s)
				_vecif_609_d = h_where_t_n_t_n_t_n(_vecif_610_exp_0, _vecif_610_d, _vecif_609_d)
				_vecif_609_m = h_where_t_n_t_n_t_n(_vecif_610_exp_0, _vecif_610_m, _vecif_609_m)
				_vecif_609__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_610_exp_0, _vecif_610__br_flag_188, _vecif_609__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_609_exp, _vecif_609_s, s)
			d = h_where_n_t_n_t_n(_vecif_609_exp, _vecif_609_d, d)
			m = h_where_n_t_n_t_n(_vecif_609_exp, _vecif_609_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_609_exp, _vecif_609__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_616_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_616_exp):
			_vecif_616_s = h_copy_t_rmRes(s)
			_vecif_616_d = d
			_vecif_616_m = m
			_vecif_616__br_flag_188 = _br_flag_188
			_vecif_617_exp_0 = h_not_t_n(_vecif_616__br_flag_188)
			if any_ifexp_true_t_n(_vecif_617_exp_0):
				_vecif_617_s = h_copy_t_rmRes(_vecif_616_s)
				_vecif_617_d = _vecif_616_d
				_vecif_617_m = _vecif_616_m
				_vecif_617__br_flag_188 = _vecif_616__br_flag_188
				_vecif_617_d = tuple_get_retval((_call_ret_618 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_617_s)), _vecif_617_m), _vecif_617_m := tuple_get_outparam(_call_ret_618, 1)))
				_vecif_619_exp = h_less_than_t_n_n(_vecif_617_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_619_exp):
					_vecif_619_s = h_copy_t_rmRes(_vecif_617_s)
					_vecif_619__br_flag_188 = _vecif_617__br_flag_188
					rmRes_h_set(_vecif_619_s, h_broadcast_t_b_b(rmRes_h(_vecif_619_s), True))
					_vecif_619__br_flag_188 = h_broadcast_t_b_b(_vecif_619__br_flag_188, True)
					_vecif_617_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_619_exp, _vecif_619_s, _vecif_617_s)
					_vecif_617__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_619_exp, _vecif_619__br_flag_188, _vecif_617__br_flag_188)
				_vecif_620_exp = h_not_t_n(_vecif_617__br_flag_188)
				if any_ifexp_true_t_n(_vecif_620_exp):
					_vecif_620_s = h_copy_t_rmRes(_vecif_617_s)
					_vecif_620__br_flag_188 = _vecif_617__br_flag_188
					_vecif_621_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_620_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_621_exp):
						_vecif_621__br_flag_188 = _vecif_620__br_flag_188
						_vecif_621__br_flag_188 = h_broadcast_t_b_b(_vecif_621__br_flag_188, True)
						_vecif_620__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_621_exp, _vecif_621__br_flag_188, _vecif_620__br_flag_188)
					_vecif_622_exp = h_not_t_n(_vecif_620__br_flag_188)
					if any_ifexp_true_t_n(_vecif_622_exp):
						_vecif_622_s = h_copy_t_rmRes(_vecif_620_s)
						rmRes_p_set(_vecif_622_s, h_add_t_vf_t_vf(rmRes_p(_vecif_622_s), h_mul_t_f_t_vf(_vecif_617_d, r)))
						_vecif_620_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_622_exp, _vecif_622_s, _vecif_620_s)
					_vecif_617_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_620_exp, _vecif_620_s, _vecif_617_s)
					_vecif_617__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_620_exp, _vecif_620__br_flag_188, _vecif_617__br_flag_188)
				_vecif_616_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_617_exp_0, _vecif_617_s, _vecif_616_s)
				_vecif_616_d = h_where_t_n_t_n_t_n(_vecif_617_exp_0, _vecif_617_d, _vecif_616_d)
				_vecif_616_m = h_where_t_n_t_n_t_n(_vecif_617_exp_0, _vecif_617_m, _vecif_616_m)
				_vecif_616__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_617_exp_0, _vecif_617__br_flag_188, _vecif_616__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_616_exp, _vecif_616_s, s)
			d = h_where_n_t_n_t_n(_vecif_616_exp, _vecif_616_d, d)
			m = h_where_n_t_n_t_n(_vecif_616_exp, _vecif_616_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_616_exp, _vecif_616__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_623_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_623_exp):
			_vecif_623_s = h_copy_t_rmRes(s)
			_vecif_623_d = d
			_vecif_623_m = m
			_vecif_623__br_flag_188 = _br_flag_188
			_vecif_624_exp_0 = h_not_t_n(_vecif_623__br_flag_188)
			if any_ifexp_true_t_n(_vecif_624_exp_0):
				_vecif_624_s = h_copy_t_rmRes(_vecif_623_s)
				_vecif_624_d = _vecif_623_d
				_vecif_624_m = _vecif_623_m
				_vecif_624__br_flag_188 = _vecif_623__br_flag_188
				_vecif_624_d = tuple_get_retval((_call_ret_625 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_624_s)), _vecif_624_m), _vecif_624_m := tuple_get_outparam(_call_ret_625, 1)))
				_vecif_626_exp = h_less_than_t_n_n(_vecif_624_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_626_exp):
					_vecif_626_s = h_copy_t_rmRes(_vecif_624_s)
					_vecif_626__br_flag_188 = _vecif_624__br_flag_188
					rmRes_h_set(_vecif_626_s, h_broadcast_t_b_b(rmRes_h(_vecif_626_s), True))
					_vecif_626__br_flag_188 = h_broadcast_t_b_b(_vecif_626__br_flag_188, True)
					_vecif_624_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_626_exp, _vecif_626_s, _vecif_624_s)
					_vecif_624__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_626_exp, _vecif_626__br_flag_188, _vecif_624__br_flag_188)
				_vecif_627_exp = h_not_t_n(_vecif_624__br_flag_188)
				if any_ifexp_true_t_n(_vecif_627_exp):
					_vecif_627_s = h_copy_t_rmRes(_vecif_624_s)
					_vecif_627__br_flag_188 = _vecif_624__br_flag_188
					_vecif_628_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_627_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_628_exp):
						_vecif_628__br_flag_188 = _vecif_627__br_flag_188
						_vecif_628__br_flag_188 = h_broadcast_t_b_b(_vecif_628__br_flag_188, True)
						_vecif_627__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_628_exp, _vecif_628__br_flag_188, _vecif_627__br_flag_188)
					_vecif_629_exp = h_not_t_n(_vecif_627__br_flag_188)
					if any_ifexp_true_t_n(_vecif_629_exp):
						_vecif_629_s = h_copy_t_rmRes(_vecif_627_s)
						rmRes_p_set(_vecif_629_s, h_add_t_vf_t_vf(rmRes_p(_vecif_629_s), h_mul_t_f_t_vf(_vecif_624_d, r)))
						_vecif_627_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_629_exp, _vecif_629_s, _vecif_627_s)
					_vecif_624_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_627_exp, _vecif_627_s, _vecif_624_s)
					_vecif_624__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_627_exp, _vecif_627__br_flag_188, _vecif_624__br_flag_188)
				_vecif_623_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_624_exp_0, _vecif_624_s, _vecif_623_s)
				_vecif_623_d = h_where_t_n_t_n_t_n(_vecif_624_exp_0, _vecif_624_d, _vecif_623_d)
				_vecif_623_m = h_where_t_n_t_n_t_n(_vecif_624_exp_0, _vecif_624_m, _vecif_623_m)
				_vecif_623__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_624_exp_0, _vecif_624__br_flag_188, _vecif_623__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_623_exp, _vecif_623_s, s)
			d = h_where_n_t_n_t_n(_vecif_623_exp, _vecif_623_d, d)
			m = h_where_n_t_n_t_n(_vecif_623_exp, _vecif_623_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_623_exp, _vecif_623__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_630_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_630_exp):
			_vecif_630_s = h_copy_t_rmRes(s)
			_vecif_630_d = d
			_vecif_630_m = m
			_vecif_630__br_flag_188 = _br_flag_188
			_vecif_631_exp_0 = h_not_t_n(_vecif_630__br_flag_188)
			if any_ifexp_true_t_n(_vecif_631_exp_0):
				_vecif_631_s = h_copy_t_rmRes(_vecif_630_s)
				_vecif_631_d = _vecif_630_d
				_vecif_631_m = _vecif_630_m
				_vecif_631__br_flag_188 = _vecif_630__br_flag_188
				_vecif_631_d = tuple_get_retval((_call_ret_632 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_631_s)), _vecif_631_m), _vecif_631_m := tuple_get_outparam(_call_ret_632, 1)))
				_vecif_633_exp = h_less_than_t_n_n(_vecif_631_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_633_exp):
					_vecif_633_s = h_copy_t_rmRes(_vecif_631_s)
					_vecif_633__br_flag_188 = _vecif_631__br_flag_188
					rmRes_h_set(_vecif_633_s, h_broadcast_t_b_b(rmRes_h(_vecif_633_s), True))
					_vecif_633__br_flag_188 = h_broadcast_t_b_b(_vecif_633__br_flag_188, True)
					_vecif_631_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_633_exp, _vecif_633_s, _vecif_631_s)
					_vecif_631__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_633_exp, _vecif_633__br_flag_188, _vecif_631__br_flag_188)
				_vecif_634_exp = h_not_t_n(_vecif_631__br_flag_188)
				if any_ifexp_true_t_n(_vecif_634_exp):
					_vecif_634_s = h_copy_t_rmRes(_vecif_631_s)
					_vecif_634__br_flag_188 = _vecif_631__br_flag_188
					_vecif_635_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_634_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_635_exp):
						_vecif_635__br_flag_188 = _vecif_634__br_flag_188
						_vecif_635__br_flag_188 = h_broadcast_t_b_b(_vecif_635__br_flag_188, True)
						_vecif_634__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_635_exp, _vecif_635__br_flag_188, _vecif_634__br_flag_188)
					_vecif_636_exp = h_not_t_n(_vecif_634__br_flag_188)
					if any_ifexp_true_t_n(_vecif_636_exp):
						_vecif_636_s = h_copy_t_rmRes(_vecif_634_s)
						rmRes_p_set(_vecif_636_s, h_add_t_vf_t_vf(rmRes_p(_vecif_636_s), h_mul_t_f_t_vf(_vecif_631_d, r)))
						_vecif_634_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_636_exp, _vecif_636_s, _vecif_634_s)
					_vecif_631_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_634_exp, _vecif_634_s, _vecif_631_s)
					_vecif_631__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_634_exp, _vecif_634__br_flag_188, _vecif_631__br_flag_188)
				_vecif_630_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_631_exp_0, _vecif_631_s, _vecif_630_s)
				_vecif_630_d = h_where_t_n_t_n_t_n(_vecif_631_exp_0, _vecif_631_d, _vecif_630_d)
				_vecif_630_m = h_where_t_n_t_n_t_n(_vecif_631_exp_0, _vecif_631_m, _vecif_630_m)
				_vecif_630__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_631_exp_0, _vecif_631__br_flag_188, _vecif_630__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_630_exp, _vecif_630_s, s)
			d = h_where_n_t_n_t_n(_vecif_630_exp, _vecif_630_d, d)
			m = h_where_n_t_n_t_n(_vecif_630_exp, _vecif_630_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_630_exp, _vecif_630__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_637_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_637_exp):
			_vecif_637_s = h_copy_t_rmRes(s)
			_vecif_637_d = d
			_vecif_637_m = m
			_vecif_637__br_flag_188 = _br_flag_188
			_vecif_638_exp_0 = h_not_t_n(_vecif_637__br_flag_188)
			if any_ifexp_true_t_n(_vecif_638_exp_0):
				_vecif_638_s = h_copy_t_rmRes(_vecif_637_s)
				_vecif_638_d = _vecif_637_d
				_vecif_638_m = _vecif_637_m
				_vecif_638__br_flag_188 = _vecif_637__br_flag_188
				_vecif_638_d = tuple_get_retval((_call_ret_639 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_638_s)), _vecif_638_m), _vecif_638_m := tuple_get_outparam(_call_ret_639, 1)))
				_vecif_640_exp = h_less_than_t_n_n(_vecif_638_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_640_exp):
					_vecif_640_s = h_copy_t_rmRes(_vecif_638_s)
					_vecif_640__br_flag_188 = _vecif_638__br_flag_188
					rmRes_h_set(_vecif_640_s, h_broadcast_t_b_b(rmRes_h(_vecif_640_s), True))
					_vecif_640__br_flag_188 = h_broadcast_t_b_b(_vecif_640__br_flag_188, True)
					_vecif_638_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_640_exp, _vecif_640_s, _vecif_638_s)
					_vecif_638__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_640_exp, _vecif_640__br_flag_188, _vecif_638__br_flag_188)
				_vecif_641_exp = h_not_t_n(_vecif_638__br_flag_188)
				if any_ifexp_true_t_n(_vecif_641_exp):
					_vecif_641_s = h_copy_t_rmRes(_vecif_638_s)
					_vecif_641__br_flag_188 = _vecif_638__br_flag_188
					_vecif_642_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_641_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_642_exp):
						_vecif_642__br_flag_188 = _vecif_641__br_flag_188
						_vecif_642__br_flag_188 = h_broadcast_t_b_b(_vecif_642__br_flag_188, True)
						_vecif_641__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_642_exp, _vecif_642__br_flag_188, _vecif_641__br_flag_188)
					_vecif_643_exp = h_not_t_n(_vecif_641__br_flag_188)
					if any_ifexp_true_t_n(_vecif_643_exp):
						_vecif_643_s = h_copy_t_rmRes(_vecif_641_s)
						rmRes_p_set(_vecif_643_s, h_add_t_vf_t_vf(rmRes_p(_vecif_643_s), h_mul_t_f_t_vf(_vecif_638_d, r)))
						_vecif_641_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_643_exp, _vecif_643_s, _vecif_641_s)
					_vecif_638_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_641_exp, _vecif_641_s, _vecif_638_s)
					_vecif_638__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_641_exp, _vecif_641__br_flag_188, _vecif_638__br_flag_188)
				_vecif_637_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_638_exp_0, _vecif_638_s, _vecif_637_s)
				_vecif_637_d = h_where_t_n_t_n_t_n(_vecif_638_exp_0, _vecif_638_d, _vecif_637_d)
				_vecif_637_m = h_where_t_n_t_n_t_n(_vecif_638_exp_0, _vecif_638_m, _vecif_637_m)
				_vecif_637__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_638_exp_0, _vecif_638__br_flag_188, _vecif_637__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_637_exp, _vecif_637_s, s)
			d = h_where_n_t_n_t_n(_vecif_637_exp, _vecif_637_d, d)
			m = h_where_n_t_n_t_n(_vecif_637_exp, _vecif_637_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_637_exp, _vecif_637__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_644_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_644_exp):
			_vecif_644_s = h_copy_t_rmRes(s)
			_vecif_644_d = d
			_vecif_644_m = m
			_vecif_644__br_flag_188 = _br_flag_188
			_vecif_645_exp_0 = h_not_t_n(_vecif_644__br_flag_188)
			if any_ifexp_true_t_n(_vecif_645_exp_0):
				_vecif_645_s = h_copy_t_rmRes(_vecif_644_s)
				_vecif_645_d = _vecif_644_d
				_vecif_645_m = _vecif_644_m
				_vecif_645__br_flag_188 = _vecif_644__br_flag_188
				_vecif_645_d = tuple_get_retval((_call_ret_646 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_645_s)), _vecif_645_m), _vecif_645_m := tuple_get_outparam(_call_ret_646, 1)))
				_vecif_647_exp = h_less_than_t_n_n(_vecif_645_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_647_exp):
					_vecif_647_s = h_copy_t_rmRes(_vecif_645_s)
					_vecif_647__br_flag_188 = _vecif_645__br_flag_188
					rmRes_h_set(_vecif_647_s, h_broadcast_t_b_b(rmRes_h(_vecif_647_s), True))
					_vecif_647__br_flag_188 = h_broadcast_t_b_b(_vecif_647__br_flag_188, True)
					_vecif_645_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_647_exp, _vecif_647_s, _vecif_645_s)
					_vecif_645__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_647_exp, _vecif_647__br_flag_188, _vecif_645__br_flag_188)
				_vecif_648_exp = h_not_t_n(_vecif_645__br_flag_188)
				if any_ifexp_true_t_n(_vecif_648_exp):
					_vecif_648_s = h_copy_t_rmRes(_vecif_645_s)
					_vecif_648__br_flag_188 = _vecif_645__br_flag_188
					_vecif_649_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_648_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_649_exp):
						_vecif_649__br_flag_188 = _vecif_648__br_flag_188
						_vecif_649__br_flag_188 = h_broadcast_t_b_b(_vecif_649__br_flag_188, True)
						_vecif_648__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_649_exp, _vecif_649__br_flag_188, _vecif_648__br_flag_188)
					_vecif_650_exp = h_not_t_n(_vecif_648__br_flag_188)
					if any_ifexp_true_t_n(_vecif_650_exp):
						_vecif_650_s = h_copy_t_rmRes(_vecif_648_s)
						rmRes_p_set(_vecif_650_s, h_add_t_vf_t_vf(rmRes_p(_vecif_650_s), h_mul_t_f_t_vf(_vecif_645_d, r)))
						_vecif_648_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_650_exp, _vecif_650_s, _vecif_648_s)
					_vecif_645_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_648_exp, _vecif_648_s, _vecif_645_s)
					_vecif_645__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_648_exp, _vecif_648__br_flag_188, _vecif_645__br_flag_188)
				_vecif_644_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_645_exp_0, _vecif_645_s, _vecif_644_s)
				_vecif_644_d = h_where_t_n_t_n_t_n(_vecif_645_exp_0, _vecif_645_d, _vecif_644_d)
				_vecif_644_m = h_where_t_n_t_n_t_n(_vecif_645_exp_0, _vecif_645_m, _vecif_644_m)
				_vecif_644__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_645_exp_0, _vecif_645__br_flag_188, _vecif_644__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_644_exp, _vecif_644_s, s)
			d = h_where_n_t_n_t_n(_vecif_644_exp, _vecif_644_d, d)
			m = h_where_n_t_n_t_n(_vecif_644_exp, _vecif_644_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_644_exp, _vecif_644__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_651_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_651_exp):
			_vecif_651_s = h_copy_t_rmRes(s)
			_vecif_651_d = d
			_vecif_651_m = m
			_vecif_651__br_flag_188 = _br_flag_188
			_vecif_652_exp_0 = h_not_t_n(_vecif_651__br_flag_188)
			if any_ifexp_true_t_n(_vecif_652_exp_0):
				_vecif_652_s = h_copy_t_rmRes(_vecif_651_s)
				_vecif_652_d = _vecif_651_d
				_vecif_652_m = _vecif_651_m
				_vecif_652__br_flag_188 = _vecif_651__br_flag_188
				_vecif_652_d = tuple_get_retval((_call_ret_653 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_652_s)), _vecif_652_m), _vecif_652_m := tuple_get_outparam(_call_ret_653, 1)))
				_vecif_654_exp = h_less_than_t_n_n(_vecif_652_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_654_exp):
					_vecif_654_s = h_copy_t_rmRes(_vecif_652_s)
					_vecif_654__br_flag_188 = _vecif_652__br_flag_188
					rmRes_h_set(_vecif_654_s, h_broadcast_t_b_b(rmRes_h(_vecif_654_s), True))
					_vecif_654__br_flag_188 = h_broadcast_t_b_b(_vecif_654__br_flag_188, True)
					_vecif_652_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_654_exp, _vecif_654_s, _vecif_652_s)
					_vecif_652__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_654_exp, _vecif_654__br_flag_188, _vecif_652__br_flag_188)
				_vecif_655_exp = h_not_t_n(_vecif_652__br_flag_188)
				if any_ifexp_true_t_n(_vecif_655_exp):
					_vecif_655_s = h_copy_t_rmRes(_vecif_652_s)
					_vecif_655__br_flag_188 = _vecif_652__br_flag_188
					_vecif_656_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_655_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_656_exp):
						_vecif_656__br_flag_188 = _vecif_655__br_flag_188
						_vecif_656__br_flag_188 = h_broadcast_t_b_b(_vecif_656__br_flag_188, True)
						_vecif_655__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_656_exp, _vecif_656__br_flag_188, _vecif_655__br_flag_188)
					_vecif_657_exp = h_not_t_n(_vecif_655__br_flag_188)
					if any_ifexp_true_t_n(_vecif_657_exp):
						_vecif_657_s = h_copy_t_rmRes(_vecif_655_s)
						rmRes_p_set(_vecif_657_s, h_add_t_vf_t_vf(rmRes_p(_vecif_657_s), h_mul_t_f_t_vf(_vecif_652_d, r)))
						_vecif_655_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_657_exp, _vecif_657_s, _vecif_655_s)
					_vecif_652_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_655_exp, _vecif_655_s, _vecif_652_s)
					_vecif_652__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_655_exp, _vecif_655__br_flag_188, _vecif_652__br_flag_188)
				_vecif_651_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_652_exp_0, _vecif_652_s, _vecif_651_s)
				_vecif_651_d = h_where_t_n_t_n_t_n(_vecif_652_exp_0, _vecif_652_d, _vecif_651_d)
				_vecif_651_m = h_where_t_n_t_n_t_n(_vecif_652_exp_0, _vecif_652_m, _vecif_651_m)
				_vecif_651__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_652_exp_0, _vecif_652__br_flag_188, _vecif_651__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_651_exp, _vecif_651_s, s)
			d = h_where_n_t_n_t_n(_vecif_651_exp, _vecif_651_d, d)
			m = h_where_n_t_n_t_n(_vecif_651_exp, _vecif_651_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_651_exp, _vecif_651__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_658_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_658_exp):
			_vecif_658_s = h_copy_t_rmRes(s)
			_vecif_658_d = d
			_vecif_658_m = m
			_vecif_658__br_flag_188 = _br_flag_188
			_vecif_659_exp_0 = h_not_t_n(_vecif_658__br_flag_188)
			if any_ifexp_true_t_n(_vecif_659_exp_0):
				_vecif_659_s = h_copy_t_rmRes(_vecif_658_s)
				_vecif_659_d = _vecif_658_d
				_vecif_659_m = _vecif_658_m
				_vecif_659__br_flag_188 = _vecif_658__br_flag_188
				_vecif_659_d = tuple_get_retval((_call_ret_660 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_659_s)), _vecif_659_m), _vecif_659_m := tuple_get_outparam(_call_ret_660, 1)))
				_vecif_661_exp = h_less_than_t_n_n(_vecif_659_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_661_exp):
					_vecif_661_s = h_copy_t_rmRes(_vecif_659_s)
					_vecif_661__br_flag_188 = _vecif_659__br_flag_188
					rmRes_h_set(_vecif_661_s, h_broadcast_t_b_b(rmRes_h(_vecif_661_s), True))
					_vecif_661__br_flag_188 = h_broadcast_t_b_b(_vecif_661__br_flag_188, True)
					_vecif_659_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_661_exp, _vecif_661_s, _vecif_659_s)
					_vecif_659__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_661_exp, _vecif_661__br_flag_188, _vecif_659__br_flag_188)
				_vecif_662_exp = h_not_t_n(_vecif_659__br_flag_188)
				if any_ifexp_true_t_n(_vecif_662_exp):
					_vecif_662_s = h_copy_t_rmRes(_vecif_659_s)
					_vecif_662__br_flag_188 = _vecif_659__br_flag_188
					_vecif_663_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_662_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_663_exp):
						_vecif_663__br_flag_188 = _vecif_662__br_flag_188
						_vecif_663__br_flag_188 = h_broadcast_t_b_b(_vecif_663__br_flag_188, True)
						_vecif_662__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_663_exp, _vecif_663__br_flag_188, _vecif_662__br_flag_188)
					_vecif_664_exp = h_not_t_n(_vecif_662__br_flag_188)
					if any_ifexp_true_t_n(_vecif_664_exp):
						_vecif_664_s = h_copy_t_rmRes(_vecif_662_s)
						rmRes_p_set(_vecif_664_s, h_add_t_vf_t_vf(rmRes_p(_vecif_664_s), h_mul_t_f_t_vf(_vecif_659_d, r)))
						_vecif_662_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_664_exp, _vecif_664_s, _vecif_662_s)
					_vecif_659_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_662_exp, _vecif_662_s, _vecif_659_s)
					_vecif_659__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_662_exp, _vecif_662__br_flag_188, _vecif_659__br_flag_188)
				_vecif_658_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_659_exp_0, _vecif_659_s, _vecif_658_s)
				_vecif_658_d = h_where_t_n_t_n_t_n(_vecif_659_exp_0, _vecif_659_d, _vecif_658_d)
				_vecif_658_m = h_where_t_n_t_n_t_n(_vecif_659_exp_0, _vecif_659_m, _vecif_658_m)
				_vecif_658__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_659_exp_0, _vecif_659__br_flag_188, _vecif_658__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_658_exp, _vecif_658_s, s)
			d = h_where_n_t_n_t_n(_vecif_658_exp, _vecif_658_d, d)
			m = h_where_n_t_n_t_n(_vecif_658_exp, _vecif_658_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_658_exp, _vecif_658__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_665_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_665_exp):
			_vecif_665_s = h_copy_t_rmRes(s)
			_vecif_665_d = d
			_vecif_665_m = m
			_vecif_665__br_flag_188 = _br_flag_188
			_vecif_666_exp_0 = h_not_t_n(_vecif_665__br_flag_188)
			if any_ifexp_true_t_n(_vecif_666_exp_0):
				_vecif_666_s = h_copy_t_rmRes(_vecif_665_s)
				_vecif_666_d = _vecif_665_d
				_vecif_666_m = _vecif_665_m
				_vecif_666__br_flag_188 = _vecif_665__br_flag_188
				_vecif_666_d = tuple_get_retval((_call_ret_667 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_666_s)), _vecif_666_m), _vecif_666_m := tuple_get_outparam(_call_ret_667, 1)))
				_vecif_668_exp = h_less_than_t_n_n(_vecif_666_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_668_exp):
					_vecif_668_s = h_copy_t_rmRes(_vecif_666_s)
					_vecif_668__br_flag_188 = _vecif_666__br_flag_188
					rmRes_h_set(_vecif_668_s, h_broadcast_t_b_b(rmRes_h(_vecif_668_s), True))
					_vecif_668__br_flag_188 = h_broadcast_t_b_b(_vecif_668__br_flag_188, True)
					_vecif_666_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_668_exp, _vecif_668_s, _vecif_666_s)
					_vecif_666__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_668_exp, _vecif_668__br_flag_188, _vecif_666__br_flag_188)
				_vecif_669_exp = h_not_t_n(_vecif_666__br_flag_188)
				if any_ifexp_true_t_n(_vecif_669_exp):
					_vecif_669_s = h_copy_t_rmRes(_vecif_666_s)
					_vecif_669__br_flag_188 = _vecif_666__br_flag_188
					_vecif_670_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_669_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_670_exp):
						_vecif_670__br_flag_188 = _vecif_669__br_flag_188
						_vecif_670__br_flag_188 = h_broadcast_t_b_b(_vecif_670__br_flag_188, True)
						_vecif_669__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_670_exp, _vecif_670__br_flag_188, _vecif_669__br_flag_188)
					_vecif_671_exp = h_not_t_n(_vecif_669__br_flag_188)
					if any_ifexp_true_t_n(_vecif_671_exp):
						_vecif_671_s = h_copy_t_rmRes(_vecif_669_s)
						rmRes_p_set(_vecif_671_s, h_add_t_vf_t_vf(rmRes_p(_vecif_671_s), h_mul_t_f_t_vf(_vecif_666_d, r)))
						_vecif_669_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_671_exp, _vecif_671_s, _vecif_669_s)
					_vecif_666_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_669_exp, _vecif_669_s, _vecif_666_s)
					_vecif_666__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_669_exp, _vecif_669__br_flag_188, _vecif_666__br_flag_188)
				_vecif_665_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_666_exp_0, _vecif_666_s, _vecif_665_s)
				_vecif_665_d = h_where_t_n_t_n_t_n(_vecif_666_exp_0, _vecif_666_d, _vecif_665_d)
				_vecif_665_m = h_where_t_n_t_n_t_n(_vecif_666_exp_0, _vecif_666_m, _vecif_665_m)
				_vecif_665__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_666_exp_0, _vecif_666__br_flag_188, _vecif_665__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_665_exp, _vecif_665_s, s)
			d = h_where_n_t_n_t_n(_vecif_665_exp, _vecif_665_d, d)
			m = h_where_n_t_n_t_n(_vecif_665_exp, _vecif_665_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_665_exp, _vecif_665__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_672_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_672_exp):
			_vecif_672_s = h_copy_t_rmRes(s)
			_vecif_672_d = d
			_vecif_672_m = m
			_vecif_672__br_flag_188 = _br_flag_188
			_vecif_673_exp_0 = h_not_t_n(_vecif_672__br_flag_188)
			if any_ifexp_true_t_n(_vecif_673_exp_0):
				_vecif_673_s = h_copy_t_rmRes(_vecif_672_s)
				_vecif_673_d = _vecif_672_d
				_vecif_673_m = _vecif_672_m
				_vecif_673__br_flag_188 = _vecif_672__br_flag_188
				_vecif_673_d = tuple_get_retval((_call_ret_674 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_673_s)), _vecif_673_m), _vecif_673_m := tuple_get_outparam(_call_ret_674, 1)))
				_vecif_675_exp = h_less_than_t_n_n(_vecif_673_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_675_exp):
					_vecif_675_s = h_copy_t_rmRes(_vecif_673_s)
					_vecif_675__br_flag_188 = _vecif_673__br_flag_188
					rmRes_h_set(_vecif_675_s, h_broadcast_t_b_b(rmRes_h(_vecif_675_s), True))
					_vecif_675__br_flag_188 = h_broadcast_t_b_b(_vecif_675__br_flag_188, True)
					_vecif_673_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_675_exp, _vecif_675_s, _vecif_673_s)
					_vecif_673__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_675_exp, _vecif_675__br_flag_188, _vecif_673__br_flag_188)
				_vecif_676_exp = h_not_t_n(_vecif_673__br_flag_188)
				if any_ifexp_true_t_n(_vecif_676_exp):
					_vecif_676_s = h_copy_t_rmRes(_vecif_673_s)
					_vecif_676__br_flag_188 = _vecif_673__br_flag_188
					_vecif_677_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_676_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_677_exp):
						_vecif_677__br_flag_188 = _vecif_676__br_flag_188
						_vecif_677__br_flag_188 = h_broadcast_t_b_b(_vecif_677__br_flag_188, True)
						_vecif_676__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_677_exp, _vecif_677__br_flag_188, _vecif_676__br_flag_188)
					_vecif_678_exp = h_not_t_n(_vecif_676__br_flag_188)
					if any_ifexp_true_t_n(_vecif_678_exp):
						_vecif_678_s = h_copy_t_rmRes(_vecif_676_s)
						rmRes_p_set(_vecif_678_s, h_add_t_vf_t_vf(rmRes_p(_vecif_678_s), h_mul_t_f_t_vf(_vecif_673_d, r)))
						_vecif_676_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_678_exp, _vecif_678_s, _vecif_676_s)
					_vecif_673_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_676_exp, _vecif_676_s, _vecif_673_s)
					_vecif_673__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_676_exp, _vecif_676__br_flag_188, _vecif_673__br_flag_188)
				_vecif_672_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_673_exp_0, _vecif_673_s, _vecif_672_s)
				_vecif_672_d = h_where_t_n_t_n_t_n(_vecif_673_exp_0, _vecif_673_d, _vecif_672_d)
				_vecif_672_m = h_where_t_n_t_n_t_n(_vecif_673_exp_0, _vecif_673_m, _vecif_672_m)
				_vecif_672__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_673_exp_0, _vecif_673__br_flag_188, _vecif_672__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_672_exp, _vecif_672_s, s)
			d = h_where_n_t_n_t_n(_vecif_672_exp, _vecif_672_d, d)
			m = h_where_n_t_n_t_n(_vecif_672_exp, _vecif_672_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_672_exp, _vecif_672__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_679_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_679_exp):
			_vecif_679_s = h_copy_t_rmRes(s)
			_vecif_679_d = d
			_vecif_679_m = m
			_vecif_679__br_flag_188 = _br_flag_188
			_vecif_680_exp_0 = h_not_t_n(_vecif_679__br_flag_188)
			if any_ifexp_true_t_n(_vecif_680_exp_0):
				_vecif_680_s = h_copy_t_rmRes(_vecif_679_s)
				_vecif_680_d = _vecif_679_d
				_vecif_680_m = _vecif_679_m
				_vecif_680__br_flag_188 = _vecif_679__br_flag_188
				_vecif_680_d = tuple_get_retval((_call_ret_681 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_680_s)), _vecif_680_m), _vecif_680_m := tuple_get_outparam(_call_ret_681, 1)))
				_vecif_682_exp = h_less_than_t_n_n(_vecif_680_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_682_exp):
					_vecif_682_s = h_copy_t_rmRes(_vecif_680_s)
					_vecif_682__br_flag_188 = _vecif_680__br_flag_188
					rmRes_h_set(_vecif_682_s, h_broadcast_t_b_b(rmRes_h(_vecif_682_s), True))
					_vecif_682__br_flag_188 = h_broadcast_t_b_b(_vecif_682__br_flag_188, True)
					_vecif_680_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_682_exp, _vecif_682_s, _vecif_680_s)
					_vecif_680__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_682_exp, _vecif_682__br_flag_188, _vecif_680__br_flag_188)
				_vecif_683_exp = h_not_t_n(_vecif_680__br_flag_188)
				if any_ifexp_true_t_n(_vecif_683_exp):
					_vecif_683_s = h_copy_t_rmRes(_vecif_680_s)
					_vecif_683__br_flag_188 = _vecif_680__br_flag_188
					_vecif_684_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_683_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_684_exp):
						_vecif_684__br_flag_188 = _vecif_683__br_flag_188
						_vecif_684__br_flag_188 = h_broadcast_t_b_b(_vecif_684__br_flag_188, True)
						_vecif_683__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_684_exp, _vecif_684__br_flag_188, _vecif_683__br_flag_188)
					_vecif_685_exp = h_not_t_n(_vecif_683__br_flag_188)
					if any_ifexp_true_t_n(_vecif_685_exp):
						_vecif_685_s = h_copy_t_rmRes(_vecif_683_s)
						rmRes_p_set(_vecif_685_s, h_add_t_vf_t_vf(rmRes_p(_vecif_685_s), h_mul_t_f_t_vf(_vecif_680_d, r)))
						_vecif_683_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_685_exp, _vecif_685_s, _vecif_683_s)
					_vecif_680_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_683_exp, _vecif_683_s, _vecif_680_s)
					_vecif_680__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_683_exp, _vecif_683__br_flag_188, _vecif_680__br_flag_188)
				_vecif_679_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_680_exp_0, _vecif_680_s, _vecif_679_s)
				_vecif_679_d = h_where_t_n_t_n_t_n(_vecif_680_exp_0, _vecif_680_d, _vecif_679_d)
				_vecif_679_m = h_where_t_n_t_n_t_n(_vecif_680_exp_0, _vecif_680_m, _vecif_679_m)
				_vecif_679__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_680_exp_0, _vecif_680__br_flag_188, _vecif_679__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_679_exp, _vecif_679_s, s)
			d = h_where_n_t_n_t_n(_vecif_679_exp, _vecif_679_d, d)
			m = h_where_n_t_n_t_n(_vecif_679_exp, _vecif_679_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_679_exp, _vecif_679__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_686_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_686_exp):
			_vecif_686_s = h_copy_t_rmRes(s)
			_vecif_686_d = d
			_vecif_686_m = m
			_vecif_686__br_flag_188 = _br_flag_188
			_vecif_687_exp_0 = h_not_t_n(_vecif_686__br_flag_188)
			if any_ifexp_true_t_n(_vecif_687_exp_0):
				_vecif_687_s = h_copy_t_rmRes(_vecif_686_s)
				_vecif_687_d = _vecif_686_d
				_vecif_687_m = _vecif_686_m
				_vecif_687__br_flag_188 = _vecif_686__br_flag_188
				_vecif_687_d = tuple_get_retval((_call_ret_688 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_687_s)), _vecif_687_m), _vecif_687_m := tuple_get_outparam(_call_ret_688, 1)))
				_vecif_689_exp = h_less_than_t_n_n(_vecif_687_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_689_exp):
					_vecif_689_s = h_copy_t_rmRes(_vecif_687_s)
					_vecif_689__br_flag_188 = _vecif_687__br_flag_188
					rmRes_h_set(_vecif_689_s, h_broadcast_t_b_b(rmRes_h(_vecif_689_s), True))
					_vecif_689__br_flag_188 = h_broadcast_t_b_b(_vecif_689__br_flag_188, True)
					_vecif_687_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_689_exp, _vecif_689_s, _vecif_687_s)
					_vecif_687__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_689_exp, _vecif_689__br_flag_188, _vecif_687__br_flag_188)
				_vecif_690_exp = h_not_t_n(_vecif_687__br_flag_188)
				if any_ifexp_true_t_n(_vecif_690_exp):
					_vecif_690_s = h_copy_t_rmRes(_vecif_687_s)
					_vecif_690__br_flag_188 = _vecif_687__br_flag_188
					_vecif_691_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_690_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_691_exp):
						_vecif_691__br_flag_188 = _vecif_690__br_flag_188
						_vecif_691__br_flag_188 = h_broadcast_t_b_b(_vecif_691__br_flag_188, True)
						_vecif_690__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_691_exp, _vecif_691__br_flag_188, _vecif_690__br_flag_188)
					_vecif_692_exp = h_not_t_n(_vecif_690__br_flag_188)
					if any_ifexp_true_t_n(_vecif_692_exp):
						_vecif_692_s = h_copy_t_rmRes(_vecif_690_s)
						rmRes_p_set(_vecif_692_s, h_add_t_vf_t_vf(rmRes_p(_vecif_692_s), h_mul_t_f_t_vf(_vecif_687_d, r)))
						_vecif_690_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_692_exp, _vecif_692_s, _vecif_690_s)
					_vecif_687_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_690_exp, _vecif_690_s, _vecif_687_s)
					_vecif_687__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_690_exp, _vecif_690__br_flag_188, _vecif_687__br_flag_188)
				_vecif_686_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_687_exp_0, _vecif_687_s, _vecif_686_s)
				_vecif_686_d = h_where_t_n_t_n_t_n(_vecif_687_exp_0, _vecif_687_d, _vecif_686_d)
				_vecif_686_m = h_where_t_n_t_n_t_n(_vecif_687_exp_0, _vecif_687_m, _vecif_686_m)
				_vecif_686__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_687_exp_0, _vecif_687__br_flag_188, _vecif_686__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_686_exp, _vecif_686_s, s)
			d = h_where_n_t_n_t_n(_vecif_686_exp, _vecif_686_d, d)
			m = h_where_n_t_n_t_n(_vecif_686_exp, _vecif_686_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_686_exp, _vecif_686__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_693_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_693_exp):
			_vecif_693_s = h_copy_t_rmRes(s)
			_vecif_693_d = d
			_vecif_693_m = m
			_vecif_693__br_flag_188 = _br_flag_188
			_vecif_694_exp_0 = h_not_t_n(_vecif_693__br_flag_188)
			if any_ifexp_true_t_n(_vecif_694_exp_0):
				_vecif_694_s = h_copy_t_rmRes(_vecif_693_s)
				_vecif_694_d = _vecif_693_d
				_vecif_694_m = _vecif_693_m
				_vecif_694__br_flag_188 = _vecif_693__br_flag_188
				_vecif_694_d = tuple_get_retval((_call_ret_695 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_694_s)), _vecif_694_m), _vecif_694_m := tuple_get_outparam(_call_ret_695, 1)))
				_vecif_696_exp = h_less_than_t_n_n(_vecif_694_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_696_exp):
					_vecif_696_s = h_copy_t_rmRes(_vecif_694_s)
					_vecif_696__br_flag_188 = _vecif_694__br_flag_188
					rmRes_h_set(_vecif_696_s, h_broadcast_t_b_b(rmRes_h(_vecif_696_s), True))
					_vecif_696__br_flag_188 = h_broadcast_t_b_b(_vecif_696__br_flag_188, True)
					_vecif_694_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_696_exp, _vecif_696_s, _vecif_694_s)
					_vecif_694__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_696_exp, _vecif_696__br_flag_188, _vecif_694__br_flag_188)
				_vecif_697_exp = h_not_t_n(_vecif_694__br_flag_188)
				if any_ifexp_true_t_n(_vecif_697_exp):
					_vecif_697_s = h_copy_t_rmRes(_vecif_694_s)
					_vecif_697__br_flag_188 = _vecif_694__br_flag_188
					_vecif_698_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_697_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_698_exp):
						_vecif_698__br_flag_188 = _vecif_697__br_flag_188
						_vecif_698__br_flag_188 = h_broadcast_t_b_b(_vecif_698__br_flag_188, True)
						_vecif_697__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_698_exp, _vecif_698__br_flag_188, _vecif_697__br_flag_188)
					_vecif_699_exp = h_not_t_n(_vecif_697__br_flag_188)
					if any_ifexp_true_t_n(_vecif_699_exp):
						_vecif_699_s = h_copy_t_rmRes(_vecif_697_s)
						rmRes_p_set(_vecif_699_s, h_add_t_vf_t_vf(rmRes_p(_vecif_699_s), h_mul_t_f_t_vf(_vecif_694_d, r)))
						_vecif_697_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_699_exp, _vecif_699_s, _vecif_697_s)
					_vecif_694_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_697_exp, _vecif_697_s, _vecif_694_s)
					_vecif_694__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_697_exp, _vecif_697__br_flag_188, _vecif_694__br_flag_188)
				_vecif_693_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_694_exp_0, _vecif_694_s, _vecif_693_s)
				_vecif_693_d = h_where_t_n_t_n_t_n(_vecif_694_exp_0, _vecif_694_d, _vecif_693_d)
				_vecif_693_m = h_where_t_n_t_n_t_n(_vecif_694_exp_0, _vecif_694_m, _vecif_693_m)
				_vecif_693__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_694_exp_0, _vecif_694__br_flag_188, _vecif_693__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_693_exp, _vecif_693_s, s)
			d = h_where_n_t_n_t_n(_vecif_693_exp, _vecif_693_d, d)
			m = h_where_n_t_n_t_n(_vecif_693_exp, _vecif_693_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_693_exp, _vecif_693__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_700_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_700_exp):
			_vecif_700_s = h_copy_t_rmRes(s)
			_vecif_700_d = d
			_vecif_700_m = m
			_vecif_700__br_flag_188 = _br_flag_188
			_vecif_701_exp_0 = h_not_t_n(_vecif_700__br_flag_188)
			if any_ifexp_true_t_n(_vecif_701_exp_0):
				_vecif_701_s = h_copy_t_rmRes(_vecif_700_s)
				_vecif_701_d = _vecif_700_d
				_vecif_701_m = _vecif_700_m
				_vecif_701__br_flag_188 = _vecif_700__br_flag_188
				_vecif_701_d = tuple_get_retval((_call_ret_702 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_701_s)), _vecif_701_m), _vecif_701_m := tuple_get_outparam(_call_ret_702, 1)))
				_vecif_703_exp = h_less_than_t_n_n(_vecif_701_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_703_exp):
					_vecif_703_s = h_copy_t_rmRes(_vecif_701_s)
					_vecif_703__br_flag_188 = _vecif_701__br_flag_188
					rmRes_h_set(_vecif_703_s, h_broadcast_t_b_b(rmRes_h(_vecif_703_s), True))
					_vecif_703__br_flag_188 = h_broadcast_t_b_b(_vecif_703__br_flag_188, True)
					_vecif_701_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_703_exp, _vecif_703_s, _vecif_701_s)
					_vecif_701__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_703_exp, _vecif_703__br_flag_188, _vecif_701__br_flag_188)
				_vecif_704_exp = h_not_t_n(_vecif_701__br_flag_188)
				if any_ifexp_true_t_n(_vecif_704_exp):
					_vecif_704_s = h_copy_t_rmRes(_vecif_701_s)
					_vecif_704__br_flag_188 = _vecif_701__br_flag_188
					_vecif_705_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_704_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_705_exp):
						_vecif_705__br_flag_188 = _vecif_704__br_flag_188
						_vecif_705__br_flag_188 = h_broadcast_t_b_b(_vecif_705__br_flag_188, True)
						_vecif_704__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_705_exp, _vecif_705__br_flag_188, _vecif_704__br_flag_188)
					_vecif_706_exp = h_not_t_n(_vecif_704__br_flag_188)
					if any_ifexp_true_t_n(_vecif_706_exp):
						_vecif_706_s = h_copy_t_rmRes(_vecif_704_s)
						rmRes_p_set(_vecif_706_s, h_add_t_vf_t_vf(rmRes_p(_vecif_706_s), h_mul_t_f_t_vf(_vecif_701_d, r)))
						_vecif_704_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_706_exp, _vecif_706_s, _vecif_704_s)
					_vecif_701_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_704_exp, _vecif_704_s, _vecif_701_s)
					_vecif_701__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_704_exp, _vecif_704__br_flag_188, _vecif_701__br_flag_188)
				_vecif_700_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_701_exp_0, _vecif_701_s, _vecif_700_s)
				_vecif_700_d = h_where_t_n_t_n_t_n(_vecif_701_exp_0, _vecif_701_d, _vecif_700_d)
				_vecif_700_m = h_where_t_n_t_n_t_n(_vecif_701_exp_0, _vecif_701_m, _vecif_700_m)
				_vecif_700__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_701_exp_0, _vecif_701__br_flag_188, _vecif_700__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_700_exp, _vecif_700_s, s)
			d = h_where_n_t_n_t_n(_vecif_700_exp, _vecif_700_d, d)
			m = h_where_n_t_n_t_n(_vecif_700_exp, _vecif_700_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_700_exp, _vecif_700__br_flag_188, _br_flag_188)
		i = h_inc_i(i)
		_vecif_707_exp = h_less_than_n_n(i, 16)
		if any_ifexp_true_n(_vecif_707_exp):
			_vecif_707_s = h_copy_t_rmRes(s)
			_vecif_707_d = d
			_vecif_707_m = m
			_vecif_707__br_flag_188 = _br_flag_188
			_vecif_708_exp_0 = h_not_t_n(_vecif_707__br_flag_188)
			if any_ifexp_true_t_n(_vecif_708_exp_0):
				_vecif_708_s = h_copy_t_rmRes(_vecif_707_s)
				_vecif_708_d = _vecif_707_d
				_vecif_708_m = _vecif_707_m
				_vecif_708__br_flag_188 = _vecif_707__br_flag_188
				_vecif_708_d = tuple_get_retval((_call_ret_709 := df_f3_i_arr1(h_copy_t_f3(rmRes_p(_vecif_708_s)), _vecif_708_m), _vecif_708_m := tuple_get_outparam(_call_ret_709, 1)))
				_vecif_710_exp = h_less_than_t_n_n(_vecif_708_d, 2.0000000000000001E-4)
				if any_ifexp_true_t_n(_vecif_710_exp):
					_vecif_710_s = h_copy_t_rmRes(_vecif_708_s)
					_vecif_710__br_flag_188 = _vecif_708__br_flag_188
					rmRes_h_set(_vecif_710_s, h_broadcast_t_b_b(rmRes_h(_vecif_710_s), True))
					_vecif_710__br_flag_188 = h_broadcast_t_b_b(_vecif_710__br_flag_188, True)
					_vecif_708_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_710_exp, _vecif_710_s, _vecif_708_s)
					_vecif_708__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_710_exp, _vecif_710__br_flag_188, _vecif_708__br_flag_188)
				_vecif_711_exp = h_not_t_n(_vecif_708__br_flag_188)
				if any_ifexp_true_t_n(_vecif_711_exp):
					_vecif_711_s = h_copy_t_rmRes(_vecif_708_s)
					_vecif_711__br_flag_188 = _vecif_708__br_flag_188
					_vecif_712_exp = h_greater_than_t_n_n(h_distance_v_t_v(c, rmRes_p(_vecif_711_s)), 30.0)
					if any_ifexp_true_t_n(_vecif_712_exp):
						_vecif_712__br_flag_188 = _vecif_711__br_flag_188
						_vecif_712__br_flag_188 = h_broadcast_t_b_b(_vecif_712__br_flag_188, True)
						_vecif_711__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_712_exp, _vecif_712__br_flag_188, _vecif_711__br_flag_188)
					_vecif_713_exp = h_not_t_n(_vecif_711__br_flag_188)
					if any_ifexp_true_t_n(_vecif_713_exp):
						_vecif_713_s = h_copy_t_rmRes(_vecif_711_s)
						rmRes_p_set(_vecif_713_s, h_add_t_vf_t_vf(rmRes_p(_vecif_713_s), h_mul_t_f_t_vf(_vecif_708_d, r)))
						_vecif_711_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_713_exp, _vecif_713_s, _vecif_711_s)
					_vecif_708_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_711_exp, _vecif_711_s, _vecif_708_s)
					_vecif_708__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_711_exp, _vecif_711__br_flag_188, _vecif_708__br_flag_188)
				_vecif_707_s = h_where_t_n_t_rmRes_t_rmRes(_vecif_708_exp_0, _vecif_708_s, _vecif_707_s)
				_vecif_707_d = h_where_t_n_t_n_t_n(_vecif_708_exp_0, _vecif_708_d, _vecif_707_d)
				_vecif_707_m = h_where_t_n_t_n_t_n(_vecif_708_exp_0, _vecif_708_m, _vecif_707_m)
				_vecif_707__br_flag_188 = h_where_t_n_t_n_t_n(_vecif_708_exp_0, _vecif_708__br_flag_188, _vecif_707__br_flag_188)
			s = h_where_n_t_rmRes_t_rmRes(_vecif_707_exp, _vecif_707_s, s)
			d = h_where_n_t_n_t_n(_vecif_707_exp, _vecif_707_d, d)
			m = h_where_n_t_n_t_n(_vecif_707_exp, _vecif_707_m, m)
			_br_flag_188 = h_where_n_t_n_t_n(_vecif_707_exp, _vecif_707__br_flag_188, _br_flag_188)
	rmRes_i_set(s, h_broadcast_t_i_i(rmRes_i(s), i))
	return s, m

def mainImage_f4_f2_arr(fragColor, fragCoord):
	global iResolution, gameTime, iTime
	uv = h_div_vf_vf(h_f2_n_n(1071.0, 503.0), swizzle_n3_xy(iResolution))
	uv = h_f2_n_n(0.59499999999999997, 0.4965)
	coord = h_mul_vf_vf(uv, swizzle_n3_xy(iResolution))
	coord = fragCoord
	m = 0
	st = h_div_t_vf_f(h_sub_t_vf_vf(coord, h_mul_vf_f(swizzle_n3_xy(iResolution), 0.5)), swizzle_n3_x(iResolution))
	gameTime = iTime
	c = h_f3_n_n_n(0.0, 0.0, -10.0)
	r = h_normalize_t_v(h_t_f3_t_n2_n(st, 1.0))
	res = tuple_get_retval((_call_ret_263 := rm_f3_f3_i_arr(c, r, m), m := tuple_get_outparam(_call_ret_263, 1)))
	sky = h_sub_vf_t_f(h_f3_n_n_n(0.95499999999999996, 0.91200000000000003, 0.93100000000000005), h_mul_t_f_f(h_dot_t_v_t_v(st, st), 0.20000000000000001))
	color = sky
	_vecif_264_exp = rmRes_h(res)
	if any_ifexp_true_t_n(_vecif_264_exp):
		_vecif_264_m = m
		_vecif_264_color = color
		n = tuple_get_retval((_call_ret_265 := normal_f3_i_arr(rmRes_p(res), _vecif_264_m), _vecif_264_m := tuple_get_outparam(_call_ret_265, 1)))
		ld = h_normalize_v(h_f3_n_n_n(0.0, 1.0, -0.10000000000000001))
		d = h_max_n_t_n(0.0, h_dot_t_v_v(n, ld))
		s = h_pow_t_n_n(h_max_n_t_n(0.0, h_dot_t_v_t_v(r, h_reflect_v_t_v(ld, n))), 1.0)
		_vecif_264_color = h_lerp_v_v_t_n(h_f3_n_n_n(0.5, 0.76300000000000001, 0.91500000000000004), glsl_vec3_f(1.0), d)
		_vecif_264_color = h_mul_t_vf_t_vf(_vecif_264_color, h_where_t_n_v_v(h_equal_t_n_n(_vecif_264_m, 1), h_f3_n_n_n(0.90500000000000003, 0.17000000000000001, 0.29199999999999998), h_f3_n_n_n(0.88500000000000001, 0.88200000000000001, 0.94499999999999995)))
		_vecif_264_color = h_lerp_t_v_t_v_t_n(_vecif_264_color, sky, h_smoothstep_n_n_t_n(20.0, 25.0, h_distance_t_v_v(rmRes_p(res), c)))
		_vecif_264_color = h_lerp_t_v_t_v_t_n(_vecif_264_color, sky, h_smoothstep_n_n_t_n(0.5, 3.0, h_mul_t_f_f(h_dot_t_v_t_v(st, st), 10.0)))
		m = h_where_t_n_t_n_t_n(_vecif_264_exp, _vecif_264_m, m)
		color = h_where_t_n_t_v_t_v(_vecif_264_exp, _vecif_264_color, color)
	fragColor = h_t_f4_t_n3_n(color, 1.0)
	return fragColor

iResolution = h_f3_n_n_n(1.0, 1.0, 1.0)
iTime = 0.0
iTimeDelta = 0.0
iFrameRate = 10.0
iFrame = 0
iChannelTime = glsl_float_x4_ctor_f_f_f_f(0.0, 0.0, 0.0, 0.0)
iChannelResolution = glsl_vec3_x4_ctor_f3_f3_f3_f3(h_f3_n_n_n(1.0, 1.0, 1.0), h_f3_n_n_n(1.0, 1.0, 1.0), h_f3_n_n_n(1.0, 1.0, 1.0), h_f3_n_n_n(1.0, 1.0, 1.0))
iMouse = h_f4_n_n_n_n(0.0, 0.0, 0.0, 0.0)
iDate = h_f4_n_n_n_n(0.0, 0.0, 0.0, 0.0)
iSampleRate = 44100.0
s_linear_clamp_sampler = None
iChannel0 = None
iChannel1 = None
iChannel2 = None
iChannel3 = None
gameTime = 0.0

def init_globals():
	pass

def shader_main(fc, fcd):
	global g_main_iChannel0, g_main_iChannel1, g_main_iChannel2, g_main_iChannel3, iChannel0, iChannel1, iChannel2, iChannel3
	init_globals()
	iChannelTime[0] = iTime
	iChannelTime[1] = iTime
	iChannelTime[2] = iTime
	iChannelTime[3] = iTime

	iChannel0 = g_main_iChannel0
	set_channel_resolution(0, iChannel0)
	iChannel1 = g_main_iChannel1
	set_channel_resolution(1, iChannel1)
	iChannel2 = g_main_iChannel2
	set_channel_resolution(2, iChannel2)
	iChannel3 = g_main_iChannel3
	set_channel_resolution(3, iChannel3)
	return mainImage_f4_f2_arr(fc, fcd)

if __name__ == "__main__":
	g_show_with_opengl = True
	g_is_autodiff = False
	g_is_profiling = False
	g_is_full_vectorized = True
	g_face_color = "gray"
	g_win_zoom = 1
	g_win_size = None
	iResolution = torch.asarray([320, 240, 1])

	iMouse[0] = iResolution[0] * 0.5
	iMouse[1] = iResolution[1] * 0.5
	iMouse[2] = iResolution[0] * 0
	iMouse[3] = iResolution[1] * 0

	g_main_iChannel0 = load_tex_2d("shaderlib/rgbanoise256.png")
	g_main_iChannel1 = load_tex_2d("shaderlib/abstract1.jpg")
	g_main_iChannel2 = load_tex_2d("shaderlib/lichen.jpg")
	g_main_iChannel3 = load_tex_2d("shaderlib/font.png")
	if g_is_autodiff and g_is_profiling:
		profile_entry(main_entry_autodiff)
	elif g_is_autodiff:
		main_entry_autodiff()
	elif g_is_profiling:
		profile_entry(main_entry)
	else:
		main_entry()

