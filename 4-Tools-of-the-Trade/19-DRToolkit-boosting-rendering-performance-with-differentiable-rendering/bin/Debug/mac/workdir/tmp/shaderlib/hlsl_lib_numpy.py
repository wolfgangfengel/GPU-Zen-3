#pip install matplotlib numpy numba cupy-cuda11x imageio PyOpenGL glfw

import sys
import numpy as np
from numba import njit
#import pyjion #conflict with matplotlib
#pyjion.enable()

pool_N = {}

def poolGetN(num):
    arr = pool_N.get(num)
    if arr is None:
        arr = np.arange(0, num, 1, dtype=np.int64)
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
        if type(v) == np.ndarray and len(v) > 4:
            return v
        else:
            return np.asarray([v])
    elif dim2 is None:
        if type(v) == np.ndarray and np.ndim(v) == 2 and len(v) > 4 and len(v[0]) == dim:
            return v
        else:
            return np.asarray([v])
    else:
        if type(v) == np.ndarray and np.ndim(v) == 3 and len(v) > 4 and len(v[0]) == dim and len(v[0, 0]) == dim2:
            return v
        else:
            return np.asarray([v])
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
        return type(v) == np.ndarray and np.ndim(v) == 1 and len(v) > 4
    elif dim2 is None:
        return type(v) == np.ndarray and np.ndim(v) == 2 and len(v) > 4 and len(v[0]) == dim
    else:
        return type(v) == np.ndarray and np.ndim(v) == 3 and len(v) > 4 and len(v[0]) == dim and len(v[0, 0]) == dim2
def maybe_svm_array(v):
    return type(v) == np.ndarray and np.ndim(v) >= 1 and len(v) > 4
def maybe_scalar_array(v):
    return type(v) == np.ndarray and np.ndim(v) == 1 and len(v) > 4
def maybe_vec_mat_array(v):
    return type(v) == np.ndarray and np.ndim(v) >= 2 and len(v) > 4

def new_tensor(type, n, dim = None, dim2 = None):
    if (dim is None or dim == 1) and dim2 is None:
        return np.empty(n)
    elif dim2 is None:
        return np.empty((n, dim))
    else:
        return np.empty((n, dim, dim2))

##@njit
def change_to_same_f(a, b):
    if a.dtype != b.dtype:
        if a.dtype == np.float32:
            a = a.astype(np.float64)
        if b.dtype == np.float32:
            b = b.astype(np.float64)
    return a, b

@njit
def h_clamp_v_v_v(v, a, b):
    return np.clip(v, a, b)
@njit
def h_clamp_n_n_n(v, a, b):
    if v < a:
        v = a
    if v > b:
        v = b
    return v
@njit
def h_clamp_v_n_n(v, a, b):
    return np.clip(v, a, b)
@njit
def h_clamp_t_n_n_n(v, a, b):
    return np.clip(v, a, b)
@njit
def h_clamp_t_n_n_t_n(v, a, b):
    return np.clip(v, a, b)
@njit
def h_clamp_t_n_t_n_t_n(v, a, b):
    return np.clip(v, a, b)
@njit
def h_clamp_t_v_n_n(v, a, b):
    return np.clip(v, a, b)
@njit
def h_clamp_t_v_v_v(v, a, b):
    return np.clip(v, a, b)

@njit
def h_lerp_n_n_n(a, b, h):
    return (1 - h) * a + h * b
@njit
def h_lerp_v_v_n(a, b, h):
    return (1 - h) * a + h * b
@njit
def h_lerp_v_v_v(a, b, h):
    return (1 - h) * a + h * b
##@njit
def h_lerp_v_v_t_n(a, b, h):
    a = np.broadcast_to(a, (len(h), len(a)))
    b = np.broadcast_to(b, (len(h), len(b)))
    return ((1 - h) * a.T + h * b.T).T
##@njit
def h_lerp_v_t_v_t_n(a, b, h):
    a = np.broadcast_to(a, (len(b), len(a)))
    return ((1 - h) * a.T + h * b.T).T
##@njit
def h_lerp_t_v_v_t_n(a, b, h):
    b = np.broadcast_to(b, (len(a), len(b)))
    return ((1 - h) * a.T + h * b.T).T
##@njit
def h_lerp_v_v_t_v(a, b, h):
    r = ((1 - h) * a + h * b)
    return r
@njit
def h_lerp_n_n_t_n(a, b, h):
    return (1 - h) * a + h * b
@njit
def h_lerp_t_n_t_n_t_n(a, b, h):
    return (1 - h) * a + h * b
@njit
def h_lerp_t_n_t_n_n(a, b, h):
    h = np.broadcast_to(h, len(a))
    return (1 - h) * a + h * b
@njit
def h_lerp_n_t_n_t_n(a, b, h):
    a = np.broadcast_to(a, len(b))
    return (1 - h) * a + h * b
@njit
def h_lerp_t_n_n_t_n(a, b, h):
    b = np.broadcast_to(b, len(a))
    return (1 - h) * a + h * b
@njit
def h_lerp_t_n_n_n(a, b, h):
    b = np.broadcast_to(b, len(a))
    h = np.broadcast_to(h, len(a))
    return (1 - h) * a + h * b
@njit
def h_lerp_n_t_n_n(a, b, h):
    a = np.broadcast_to(a, len(b))
    h = np.broadcast_to(h, len(b))
    return (1 - h) * a + h * b
@njit
def h_lerp_t_v_t_v_t_v(a, b, h):
    return (1 - h) * a + h * b
@njit
def h_lerp_t_v_v_t_v(a, b, h):
    m = len(a)
    n = len(b)
    b = np.broadcast_to(b, (m, n))
    return (1 - h) * a + h * b
@njit
def h_lerp_t_v_t_v_t_n(a, b, h):
    return ((1 - h) * a.T + h * b.T).T
@njit
def h_lerp_t_v_t_v_n(a, b, h):
    h = np.broadcast_to(h, len(a))
    return ((1 - h) * a.T + h * b.T).T
@njit
def h_lerp_t_v_v_n(a, b, h):
    h = np.broadcast_to(h, len(a))
    b = np.broadcast_to(b, (len(a), len(b)))
    return ((1 - h) * a.T + h * b.T).T

##@njit
def h_smoothstep_n_n_n(a, b, v):
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_n_n_v(a, b, v):
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_n_n_t_n(a, b, v):
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_n_t_n_t_n(a, b, v):
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_t_n_n_t_n(a, b, v):
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_n_n_t_v(a, b, v):
    m = len(v)
    n = len(v[0])
    a = np.broadcast_to(a, (m, n))
    b = np.broadcast_to(b, (m, n))
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_v_v_t_v(a, b, v):
    m = len(v)
    n = len(a)
    a = np.broadcast_to(a, (m, n))
    b = np.broadcast_to(b, (m, n))
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_t_n_t_n_t_n(a, b, v):
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)
@njit
def h_smoothstep_t_n_t_n_n(a, b, v):
    v = np.broadcast_to(v, len(a))
    t = (v - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

@njit
def h_sin_n(v):
    return np.sin(v)
@njit
def h_sin_v(v):
    return np.sin(v)
@njit
def h_sin_t_n(v):
    return np.sin(v)
@njit
def h_sin_t_v(v):
    return np.sin(v)

@njit
def h_cos_n(v):
    return np.cos(v)
@njit
def h_cos_v(v):
    return np.cos(v)
@njit
def h_cos_t_n(v):
    return np.cos(v)
@njit
def h_cos_t_v(v):
    return np.cos(v)

@njit
def h_asin_n(v):
    return np.arcsin(v)
@njit
def h_asin_v(v):
    return np.arcsin(v)
@njit
def h_asin_t_n(v):
    return np.arcsin(v)
@njit
def h_asin_t_v(v):
    return np.arcsin(v)

@njit
def h_acos_n(v):
    return np.arccos(v)
@njit
def h_acos_v(v):
    return np.arccos(v)
@njit
def h_acos_t_n(v):
    return np.arccos(v)
@njit
def h_acos_t_v(v):
    return np.arccos(v)

@njit
def h_tan_n(v):
    return np.tan(v)
@njit
def h_tan_t_n(v):
    return np.tan(v)

@njit
def h_atan_n(v):
    return np.arctan(v)
@njit
def h_atan_t_n(v):
    return np.arctan(v)
@njit
def h_atan2_n_n(y, x):
    return np.arctan2(y, x)
@njit
def h_atan2_t_n_t_n(y, x):
    return np.arctan2(y, x)

@njit
def h_radians_n(v):
    return np.radians(v)
@njit
def h_radians_t_n(v):
    return np.radians(v)

@njit
def h_degrees_n(v):
    return np.degrees(v)
@njit
def h_degrees_t_n(v):
    return np.degrees(v)

@njit
def h_frac_n(v):
    return v - np.floor(v)
@njit
def h_frac_v(v):
    return v - np.floor(v)
@njit
def h_frac_t_n(v):
    return v - np.floor(v)
@njit
def h_frac_t_v(v):
    return v - np.floor(v)

@njit
def h_fmod_n_n(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_n_v(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_v_n(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_v_v(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_t_n_n(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_t_v_n(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_t_v_v(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_n_t_n(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_n_t_v(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_v_t_v(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_t_n_t_n(v1, v2):
    return np.fmod(v1, v2)
@njit
def h_fmod_t_v_t_v(v1, v2):
    return np.fmod(v1, v2)

@njit
def h_dot_n_n(v1, v2):
    return np.dot(v1, v2)
##@njit
def h_dot_v_v(v1, v2):
    return np.dot(v1, v2)
@njit
def h_dot_t_n_n(v1, v2):
    return np.dot(v1, v2)
##@njit
def h_dot_t_v_v(v1, v2):
    m = len(v1)
    v2 = np.broadcast_to(v2, (m, len(v2)))
    return np.einsum('ij,ij->i', v1, v2)
##@njit
def h_dot_v_t_v(v1, v2):
    m = len(v2)
    v1 = np.broadcast_to(v1, (m, len(v1)))
    return np.einsum('ij,ij->i', v1, v2)
@njit
def h_dot_t_n_t_n(v1, v2):
    return v1 * v2
##@njit
def h_dot_t_v_t_v(v1, v2):
    return np.einsum('ij,ij->i', v1, v2)
    #return (v1 * v2).sum(1)

@njit
def h_reflect_v_v(v1, v2):
    dot = np.dot(v1, v2)
    return v1 - 2 * dot * v2
##@njit
def h_reflect_t_v_v(v1, v2):
    m = len(v1)
    v2 = np.broadcast_to(v2, (m, len(v2)))
    dot = np.einsum('ij,ij->i', v1, v2)
    _2_dot_v2 = np.multiply(dot, v2.T).T * 2.0
    return v1 - _2_dot_v2
##@njit
def h_reflect_v_t_v(v1, v2):
    m = len(v2)
    v1 = np.broadcast_to(v1, (m, len(v1)))
    dot = np.einsum('ij,ij->i', v1, v2)
    _2_dot_v2 = np.multiply(dot, v2.T).T * 2.0
    return v1 - _2_dot_v2
##@njit
def h_reflect_t_v_t_v(v1, v2):
    dot = np.einsum('ij,ij->i', v1, v2)
    _2_dot_v2 = np.multiply(dot, v2.T).T * 2.0
    return v1 - _2_dot_v2

##@njit
def h_refract_v_v_n(I, N, eta):
    m = len(I)
    dotval = h_dot_v_v(N, I)
    k = 1.0 - eta * eta * (1.0 - dotval * dotval)
    R0 = np.broadcast_to(0.0, m)
    bvs = k >= 0.0
    bvs = np.broadcast_to(bvs, m)
    R = np.where(bvs, eta * I - (eta * dotval + np.sqrt(np.abs(k))) * N, R0)
    return R
##@njit
def h_refract_t_v_t_v_n(I, N, eta):
    m = len(I)
    n = len(I[0])
    dotval = h_dot_t_v_t_v(N, I)
    k = 1.0 - eta * eta * (1.0 - dotval * dotval)
    R0 = np.broadcast_to(0.0, (m, n))
    bvs = k >= 0.0
    bvs = np.broadcast_to(bvs, (n, m)).T
    R = np.where(bvs, eta * I - ((eta * dotval + np.sqrt(np.abs(k))) * N.T).T, R0)
    return R
##@njit
def h_refract_t_v_t_v_t_n(I, N, eta):
    m = len(I)
    n = len(I[0])
    dotval = h_dot_t_v_t_v(N, I)
    k = 1.0 - eta * eta * (1.0 - dotval * dotval)
    R0 = np.broadcast_to(0.0, (m, n))
    bvs = k >= 0.0
    bvs = np.broadcast_to(bvs, (n, m)).T
    R = np.where(bvs, (eta * I.T).T - ((eta * dotval + np.sqrt(np.abs(k))) * N.T).T, R0)
    return R


@njit
def h_floor_n(v):
    return np.floor(v)
@njit
def h_floor_v(v):
    return np.floor(v)
@njit
def h_floor_t_n(v):
    return np.floor(v)
@njit
def h_floor_t_v(v):
    return np.floor(v)

@njit
def h_ceil_n(v):
    return np.ceil(v)
@njit
def h_ceil_v(v):
    return np.ceil(v)
@njit
def h_ceil_t_n(v):
    return np.ceil(v)
@njit
def h_ceil_t_v(v):
    return np.ceil(v)

@njit
def h_round_n(v):
    return np.round(v)
@njit
def h_round_v(v):
    return np.round(v)
@njit
def h_round_t_n(v):
    return np.round(v)
##@njit
def h_round_t_v(v):
    return np.round(v)

@njit
def h_length_n(v):
    return np.abs(v)
@njit
def h_length_t_n(v):
    return np.abs(v)
@njit
def h_length_v(v):
    return np.sqrt(np.sum(np.power(v, 2)))
##@njit
def h_length_t_v(v):
    return np.sqrt(np.einsum('ij,ij->i', v, v))

@njit
def h_distance_v_v(v1, v2):
    v = v1 - v2
    return np.sqrt(np.sum(np.power(v, 2)))
##@njit
def h_distance_t_v_v(v1, v2):
    v = v1 - v2
    return np.sqrt(np.einsum('ij,ij->i', v, v))
##@njit
def h_distance_v_t_v(v1, v2):
    v = v1 - v2
    return np.sqrt(np.einsum('ij,ij->i', v, v))
##@njit
def h_distance_t_v_t_v(v1, v2):
    v = v1 - v2
    return np.sqrt(np.einsum('ij,ij->i', v, v))


##@njit
def h_normalize_v(v):
    return v / (np.linalg.norm(v) + sys.float_info.epsilon)
##@njit
def h_normalize_t_v(v):
    r = (v.T /(np.linalg.norm(v, axis=1) + sys.float_info.epsilon)).T
    return r

@njit
def h_cross_v_v(a, b):
    return np.cross(a, b)
@njit
def h_cross_t_v_v(a, b):
    return np.cross(a, b)
@njit
def h_cross_t_v_t_v(a, b):
    return np.cross(a, b)

@njit
def h_sqrt_n(v):
    return np.sqrt(v)
@njit
def h_sqrt_v(v):
    return np.sqrt(v)
@njit
def h_sqrt_t_n(v):
    return np.sqrt(v)
@njit
def h_sqrt_t_v(v):
    return np.sqrt(v)

@njit
def h_pow_n_n(v, n):
    return np.power(v, n)
@njit
def h_pow_v_v(v, n):
    return np.power(v, n)
@njit
def h_pow_v_n(v, n):
    return np.power(v, n)
@njit
def h_pow_t_n_n(v, n):
    return np.power(v, n)
@njit
def h_pow_n_t_n(v, n):
    return np.power(v, n)
@njit
def h_pow_t_n_t_n(v, n):
    return np.power(v, n)
@njit
def h_pow_t_v_v(v, n):
    return np.power(v, n)

@njit
def h_log_n(v):
    return np.log(v)
@njit
def h_log_v(v):
    return np.log(v)
@njit
def h_log_t_n(v):
    return np.log(v)
@njit
def h_log_t_v(v):
    return np.log(v)

@njit
def h_log2_n(v):
    return np.log2(v)
@njit
def h_log2_v(v):
    return np.log2(v)
@njit
def h_log2_t_n(v):
    return np.log2(v)
@njit
def h_log2_t_v(v):
    return np.log2(v)

@njit
def h_log10_n(v):
    return np.log10(v)
@njit
def h_log10_v(v):
    return np.log10(v)
@njit
def h_log10_t_n(v):
    return np.log10(v)
@njit
def h_log10_t_v(v):
    return np.log10(v)

@njit
def h_exp_n(v):
    return np.exp(v)
@njit
def h_exp_v(v):
    return np.exp(v)
@njit
def h_exp_t_n(v):
    return np.exp(v)
@njit
def h_exp_t_v(v):
    return np.exp(v)

@njit
def h_exp2_n(v):
    return np.exp2(v)
@njit
def h_exp2_v(v):
    return np.exp2(v)
@njit
def h_exp2_t_n(v):
    return np.exp2(v)
@njit
def h_exp2_t_v(v):
    return np.exp2(v)

@njit
def h_sign_n(v):
    return np.sign(v)
@njit
def h_sign_v(v):
    return np.sign(v)
@njit
def h_sign_t_n(v):
    return np.sign(v)
@njit
def h_sign_t_v(v):
    return np.sign(v)

@njit
def h_ddx_n(v):
    return 0.001
@njit
def h_ddy_n(v):
    return 0.001
@njit
def h_ddx_v(v):
    return np.asarray([0.0, 0.0])
@njit
def h_ddy_v(v):
    return np.asarray([0.0, 0.0])
@njit
def h_ddx_fine_n(v):
    return 0.0
@njit
def h_ddy_fine_n(v):
    return 0.0
@njit
def h_ddx_coarse_n(v):
    return 0.0
@njit
def h_ddy_coarse_n(v):
    return 0.0
@njit
def h_ddx_t_n(v):
    return np.broadcast_to(0.001, len(v))
@njit
def h_ddy_t_n(v):
    return np.broadcast_to(0.001, len(v))
@njit
def h_ddx_t_v(v):
    return np.broadcast_to([0.0, 0.0], (len(v), 2))
@njit
def h_ddy_t_v(v):
    return np.broadcast_to([0.0, 0.0], (len(v), 2))
@njit
def h_ddx_fine_t_n(v):
    return np.broadcast_to(0.0, len(v))
@njit
def h_ddy_fine_t_n(v):
    return np.broadcast_to(0.0, len(v))
@njit
def h_ddx_coarse_t_n(v):
    return np.broadcast_to(0.0, len(v))
@njit
def h_ddy_coarse_t_n(v):
    return np.broadcast_to(0.0, len(v))

@njit
def h_fwidth_n(v):
    return h_abs_n(h_ddx_n(v)) + h_abs_n(h_ddy_n(v))
@njit
def h_fwidth_t_n(v):
    return h_abs_t_n(h_ddx_t_n(v)) + h_abs_t_n(h_ddy_t_n(v))
@njit
def h_fwidth_v(v):
    return h_abs_v(h_ddx_v(v)) + h_abs_v(h_ddy_v(v))
@njit
def h_fwidth_t_v(v):
    return h_abs_t_v(h_ddx_t_v(v)) + h_abs_t_v(h_ddy_t_v(v))

@njit
def h_transpose_m(m):
    return m.transpose()
@njit
def h_transpose_t_m(m):
    return m.transpose((0,1,2))

##@njit
def h_matmul_f2x2_f2(m, v):
    return np.matmul(m, v)
##@njit
def h_matmul_f2x2_t_f2(m, v):
    return np.matmul(m, v.T).T
##@njit
def h_matmul_t_f2x2_f2(m, v):
    return np.matmul(m, v.T).T
##@njit
def h_matmul_t_f2x2_t_f2(m, v):
    r = -np.matmul(np.broadcast_to(v, (2, len(v), 2)).transpose(1, 0, 2), m).transpose(1, 0, 2)[0, ...]
    return r
##@njit
def h_matmul_f2_f2x2(v, m):
    return np.matmul(v, m)
##@njit
def h_matmul_t_f2_f2x2(v, m):
    return np.matmul(v, m)
##@njit
def h_matmul_t_f2_t_f2x2(v, m):
    return np.matmul(np.broadcast_to(v, (2, len(v), 2)).transpose(1, 0, 2), m).transpose(1, 0, 2)[0, ...]
##@njit
def h_matmul_f3x3_f3(m, v):
    return np.matmul(m, v)
##@njit
def h_matmul_f3x3_f3x3(m1, m2):
    return np.matmul(m1, m2)
##@njit
def h_matmul_f3x3_t_f3(m, v):
    return np.matmul(m, v.T).T
##@njit
def h_matmul_f3_f3x3(v, m):
    return np.matmul(v, m)
##@njit
def h_matmul_t_f3_f3x3(v, m):
    return np.matmul(v, m)
##@njit
def h_matmul_t_f3_t_f3x3(v, m):
    return np.matmul(np.broadcast_to(v, (3, len(v), 3)).transpose(1, 0, 2), m).transpose(1, 0, 2)[0, ...]
##@njit
def h_matmul_f4x4_f4(m, v):
    return np.matmul(m, v)
##@njit
def h_matmul_f4x4_f4x4(m1, m2):
    return np.matmul(m1, m2)
##@njit
def h_matmul_f4x4_t_f4(m, v):
    return np.matmul(m, v.T).T
##@njit
def h_matmul_f4_f4x4(v, m):
    return np.matmul(v, m)
##@njit
def h_matmul_t_f4_f4x4(v, m):
    return np.matmul(v, m)
##@njit
def h_matmul_t_f4_t_f4x4(v, m):
    return np.matmul(np.broadcast_to(v, (4, len(v), 4)).transpose(1, 0, 2), m).transpose(1, 0, 2)[0, ...]

@njit
def h_max_n_n(a, b):
    return np.maximum(a, b)
@njit
def h_max_v_n(a, b):
    return np.maximum(a, b)
@njit
def h_max_n_v(a, b):
    return np.maximum(a, b)
@njit
def h_max_v_v(a, b):
    return np.maximum(a, b)
@njit
def h_max_v_t_v(a, b):
    return np.maximum(a, b)
@njit
def h_max_t_v_v(a, b):
    return np.maximum(a, b)
@njit
def h_max_t_n_n(a, b):
    return np.maximum(a, b)
@njit
def h_max_n_t_n(a, b):
    return np.maximum(a, b)
@njit
def h_max_t_n_t_n(a, b):
    return np.maximum(a, b)
@njit
def h_max_t_v_t_v(a, b):
    return np.maximum(a, b)
@njit
def h_max_t_v_n(a, b):
    return np.maximum(a, b)
@njit
def h_max_n_t_v(a, b):
    return np.maximum(a, b)

@njit
def h_min_n_n(a, b):
    return np.minimum(a, b)
@njit
def h_min_v_n(a, b):
    return np.minimum(a, b)
@njit
def h_min_n_v(a, b):
    return np.minimum(a, b)
@njit
def h_min_v_v(a, b):
    return np.minimum(a, b)
@njit
def h_min_v_t_v(a, b):
    return np.minimum(a, b)
@njit
def h_min_t_v_v(a, b):
    return np.minimum(a, b)
@njit
def h_min_t_n_n(a, b):
    return np.minimum(a, b)
@njit
def h_min_n_t_n(a, b):
    return np.minimum(a, b)
@njit
def h_min_t_n_t_n(a, b):
    return np.minimum(a, b)
@njit
def h_min_t_v_t_v(a, b):
    return np.minimum(a, b)
@njit
def h_min_t_v_n(a, b):
    return np.minimum(a, b)
@njit
def h_min_n_t_v(a, b):
    return np.minimum(a, b)

##@njit
def h_where_n_n_n(b, y, n):
    return y if b else n
##@njit
def h_where_n_v_v(b, y, n):
    return y if b else n
##@njit
def h_where_n_t_n_t_n(b, y, n):
    return y if b else n
##@njit
def h_where_n_t_v_t_v(b, y, n):
    return y if b else n
##@njit
def h_where_n_n_t_n(b, y, n):
    ct = len(n)
    y = np.broadcast_to(y, ct)
    return y if b else n
##@njit
def h_where_n_t_n_n(b, y, n):
    ct = len(y)
    n = np.broadcast_to(n, ct)
    return y if b else n
##@njit
def h_where_n_v_t_v(b, y, n):
    ct = len(n)
    y = np.broadcast_to(y, (ct, len(y)))
    return y if b else n
##@njit
def h_where_n_t_v_v(b, y, n):
    ct = len(y)
    n = np.broadcast_to(n, (ct, len(n)))
    return y if b else n
##@njit
def h_where_t_n_t_n_t_n(b, y, n):
    return np.where(b, y, n)
##@njit
def h_where_t_n_t_v_t_v(b, y, n):
    ct = len(y[0])
    b = np.broadcast_to(b, (ct, len(b))).T
    return np.where(b, y, n)
##@njit
def h_where_t_n_t_v_v(b, y, n):
    ct = len(y[0])
    b = np.broadcast_to(b, (ct, len(b))).T
    n = np.broadcast_to(n, (len(y), len(n)))
    return np.where(b, y, n)
##@njit
def h_where_t_n_v_t_v(b, y, n):
    ct = len(n[0])
    b = np.broadcast_to(b, (ct, len(b))).T
    y = np.broadcast_to(y, (len(n), len(y)))
    return np.where(b, y, n)
##@njit
def h_where_t_n_v_v(b, y, n):
    ct = len(n)
    b = np.broadcast_to(b, (ct, len(b))).T
    y = np.broadcast_to(y, (len(b), ct))
    n = np.broadcast_to(n, (len(b), ct))
    return np.where(b, y, n)
##@njit
def h_where_t_n_t_n_n(b, y, n):
    ct = len(y)
    n = np.broadcast_to(n, ct)
    return np.where(b, y, n)
##@njit
def h_where_t_n_n_t_n(b, y, n):
    ct = len(n)
    y = np.broadcast_to(y, ct)
    return np.where(b, y, n)
##@njit
def h_where_t_n_n_n(b, y, n):
    ct = len(b)
    y = np.broadcast_to(y, ct)
    n = np.broadcast_to(n, ct)
    return np.where(b, y, n)
##@njit
def h_where_t_n_t_m_t_m(b, y, n):
    ct = len(b)
    m1 = len(y[0])
    m2 = len(y[0][0])
    b = np.broadcast_to(b, (m2, m1, ct)).T
    return np.where(b, y, n)
##@njit
def h_where_n_m_m(b, y, n):
    return y if b else n
##@njit
def h_where_n_t_an_t_an(b, y, n):
    return y if b else n
##@njit
def h_where_n_t_an_an(b, y, n):
    m1 = len(n)
    m2 = len(y[0])
    n = np.broadcast_to(n, (m2, m1)).T
    return y if b else n
##@njit
def h_where_n_an_t_an(b, y, n):
    m1 = len(y)
    m2 = len(n[0])
    y = np.broadcast_to(y, (m2, m1)).T
    return y if b else n
##@njit
def h_where_t_n_t_an_t_an(b, y, n):
    ct = len(b)
    m = len(y)
    b = np.broadcast_to(b, (m, ct))
    return np.where(b, y, n)
##@njit
def h_where_t_n_t_av_t_av(b, y, n):
    ct = len(b)
    m1 = len(y)
    m2 = len(y[0][0])
    b = np.broadcast_to(b, (m1, m2, ct)).transpose(0, 2, 1)
    return np.where(b, y, n)
##@njit
def h_where_n_t_av_t_av(b, y, n):
    return y if b else n
##@njit
def h_where_v_v_v(b, y, n):
    return np.where(b, y, n)
##@njit
def h_where_t_v_v_t_v(b, y, n):
    m = len(b)
    y = np.broadcast_to(y, (m, len(y)))
    return np.where(b, y, n)

##@njit
def h_step_n_n(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_v_n(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_n_v(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_v_v(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_t_v_v(y, x):
    x = np.broadcast_to(x, (len(y), len(x)))
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_v_t_v(y, x):
    y = np.broadcast_to(y, (len(x), len(y)))
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_t_v_t_v(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_n_t_v(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_n_t_n(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_t_n_n(y, x):
    return np.heaviside(x - y, 1.0)
##@njit
def h_step_t_n_t_n(y, x):
    return np.heaviside(x - y, 1.0)

@njit
def h_abs_n(v):
    return np.abs(v)
@njit
def h_abs_v(v):
    return np.abs(v)
@njit
def h_abs_t_n(v):
    return np.abs(v)
@njit
def h_abs_t_v(v):
    return np.abs(v)

@njit
def h_any_n(v):
    return np.any(v)
@njit
def h_any_v(v):
    return np.any(v)
##@njit
def h_any_t_n(v):
    return np.any(np.column_stack((v, v)), axis=1)
##@njit
def h_any_t_v(v):
    return np.any(v, axis=1)

@njit
def h_all_n(v):
    return np.all(v)
@njit
def h_all_v(v):
    return np.all(v)
@njit
def h_all_t_n(v):
    return np.all(v)
@njit
def h_all_t_v(v):
    return np.all(v)

##@njit
def array_init_an(arr):
    return np.asarray(arr)
##@njit
def array_init_an2(arr):
    return np.asarray(arr)
##@njit
def array_init_an3(arr):
    return np.asarray(arr)
##@njit
def array_init_an4(arr):
    return np.asarray(arr)
##@njit
def array_init_t_an(arr):
    return np.stack(arr)
##@njit
def array_init_t_an2(arr):
    return np.stack(arr)
##@njit
def array_init_t_an3(arr):
    return np.stack(arr)
##@njit
def array_init_t_an4(arr):
    return np.stack(arr)
##@njit
def array_set_an_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an_n(arr, ix):
    return arr[ix]
##@njit
def array_set_t_an_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_t_an_n(arr, ix):
    return arr[ix]
##@njit
def array_set_an2_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an2_n(arr, ix):
    return arr[ix]
##@njit
def array_set_an3_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an3_n(arr, ix):
    return arr[ix]
##@njit
def array_set_an4_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an4_n(arr, ix):
    return arr[ix]
##@njit
def array_set_t_an2_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_t_an2_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_t_an3_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_t_an3_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_t_an4_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_t_an4_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_an2_t_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an2_t_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_an3_t_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an3_t_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_an4_t_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_an4_t_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_t_an_t_n(arr, ix, v):
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
##@njit
def array_get_t_an_t_n(arr, ix):
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
##@njit
def array_set_t_an2_t_n(arr, ix, v):
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
##@njit
def array_get_t_an2_t_n(arr, ix):
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
##@njit
def array_set_t_an3_t_n(arr, ix, v):
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
##@njit
def array_get_t_an3_t_n(arr, ix):
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
##@njit
def array_set_t_an4_t_n(arr, ix, v):
    n = len(ix)
    nix = poolGetN(n)
    arr[ix, nix] = v
    return v
##@njit
def array_get_t_an4_t_n(arr, ix):
    n = len(ix)
    nix = poolGetN(n)
    r = arr[ix, nix]
    return r
##@njit
def array_set_n2_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_n2_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_n3_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_n3_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_n4_n(arr, ix, v):
    arr[ix] = v
    return v
##@njit
def array_get_n4_n(arr, ix):
    r = arr[ix]
    return r
##@njit
def array_set_n2x2_n(m, ix, val):
    m[ix][0] = val[0]
    m[ix][1] = val[1]
##@njit
def array_get_n2x2_n(m, ix):
    arr = m[ix]
    return arr
##@njit
def array_set_n3x3_n(m, ix, val):
    m[ix][0] = val[0]
    m[ix][1] = val[1]
    m[ix][2] = val[2]
##@njit
def array_get_n3x3_n(m, ix):
    arr = m[ix]
    return arr
##@njit
def array_set_n4x4_n(m, ix, val):
    m[ix][0] = val[0]
    m[ix][1] = val[1]
    m[ix][2] = val[2]
    m[ix][3] = val[3]
##@njit
def array_get_n4x4_n(m, ix):
    arr = m[ix]
    return arr
##@njit
def array_set_t_n2_n(m, ix, val):
    m[..., ix] = val
    return val
##@njit
def array_get_t_n2_n(m, ix):
    v = m[..., ix]
    return v
##@njit
def array_set_t_n3_n(m, ix, val):
    m[..., ix] = val
    return val
##@njit
def array_get_t_n3_n(m, ix):
    v = m[..., ix]
    return v
##@njit
def array_set_t_n4_n(m, ix, val):
    m[..., ix] = val
    return val
##@njit
def array_get_t_n4_n(m, ix):
    v = m[..., ix]
    return v
##@njit
def array_set_t_n2x2_n(m, ix, val):
    m.swapaxes(1,2)[..., ix][..., 0] = val[..., 0]
    m.swapaxes(1,2)[..., ix][..., 1] = val[..., 1]
    return val
##@njit
def array_get_t_n2x2_n(m, ix):
    v = m.swapaxes(1,2)[..., ix]
    return v
##@njit
def array_set_t_n3x3_n(m, ix, val):
    m.swapaxes(1,2)[..., ix][..., 0] = val[..., 0]
    m.swapaxes(1,2)[..., ix][..., 1] = val[..., 1]
    m.swapaxes(1,2)[..., ix][..., 2] = val[..., 2]
    return val
##@njit
def array_get_t_n3x3_n(m, ix):
    v = m.swapaxes(1,2)[..., ix]
    return v
##@njit
def array_set_t_n4x4_n(m, ix, val):
    m.swapaxes(1,2)[..., ix][..., 0] = val[..., 0]
    m.swapaxes(1,2)[..., ix][..., 1] = val[..., 1]
    m.swapaxes(1,2)[..., ix][..., 2] = val[..., 2]
    m.swapaxes(1,2)[..., ix][..., 3] = val[..., 3]
    return val
##@njit
def array_get_t_n4x4_n(m, ix):
    v = m.swapaxes(1,2)[..., ix]
    return v
##@njit
def array_set_t_n2_n(m, ix, val):
    m[..., ix] = val
    return val
##@njit
def array_get_t_n2_n(m, ix):
    v = m[..., ix]
    return v
##@njit
def array_set_t_n3_n(m, ix, val):
    m[..., ix] = val
    return val
##@njit
def array_get_t_n3_n(m, ix):
    v = m[..., ix]
    return v
##@njit
def array_set_t_n4_n(m, ix, val):
    m[..., ix] = val
    return val
##@njit
def array_get_t_n4_n(m, ix):
    v = m[..., ix]
    return v

##@njit
def array_set_and_broadcast_an_n(arr, ix, v):
    m = len(v)
    arr = np.tile(arr, (m, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr
##@njit
def array_set_and_broadcast_an2_n(arr, ix, v):
    m = len(v)
    arr = np.tile(arr, (m, 1, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr
##@njit
def array_set_and_broadcast_an3_n(arr, ix, v):
    m = len(v)
    arr = np.tile(arr, (m, 1, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr
##@njit
def array_set_and_broadcast_an4_n(arr, ix, v):
    m = len(v)
    arr = np.tile(arr, (m, 1, 1)).swapaxes(0, 1)
    arr[ix] = v
    return v, arr

##@njit
def array_broadcast_an(ct, arr):
    arr = np.tile(arr, (ct, 1)).swapaxes(0, 1)
    return arr
##@njit
def array_broadcast_an2(ct, arr):
    arr = np.tile(arr, (ct, 1, 1)).swapaxes(0, 1)
    return arr
##@njit
def array_broadcast_an3(ct, arr):
    arr = np.tile(arr, (ct, 1, 1)).swapaxes(0, 1)
    return arr
##@njit
def array_broadcast_an4(ct, arr):
    arr = np.tile(arr, (ct, 1, 1)).swapaxes(0, 1)
    return arr

def swizzle(v, m, dim, dim2 = None):
    if maybe_vec_mat_array(v):
        if dim == 2 and dim2 is None and m == "xy":
            nv = np.copy(v)
            return nv
        elif dim == 3 and dim2 is None and m == "xyz":
            nv = np.copy(v)
            return nv
        elif dim == 4 and dim2 is None and m == "xyzw":
            nv = np.copy(v)
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
            return np.column_stack(nv)
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
            return np.asarray(nv)

def swizzle_set(v, m, val, dim, dim2 = None):
    if maybe_vec_mat_array(v):
        if dim == 2 and dim2 is None and m == "xy":
            np.copyto(v, val)
        elif dim == 3 and dim2 is None and m == "xyz":
            np.copyto(v, val)
        elif dim == 4 and dim2 is None and m == "xyzw":
            np.copyto(v, val)
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
            np.copyto(v, val)
        elif dim == 3 and dim2 is None and m == "xyz":
            np.copyto(v, val)
        elif dim == 4 and dim2 is None and m == "xyzw":
            np.copyto(v, val)
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

@njit
def h_inc_i(v):
    return v + 1
@njit
def h_dec_i(v):
    return v - 1
@njit
def h_inc_vi(v):
    return v + 1
@njit
def h_dec_vi(v):
    return v - 1
@njit
def h_inc_t_i(v):
    return v + 1
@njit
def h_dec_t_i(v):
    return v - 1
@njit
def h_inc_t_vi(v):
    return v + 1
@njit
def h_dec_t_vi(v):
    return v - 1
@njit
def h_inc_f(v):
    return v + 1
@njit
def h_dec_f(v):
    return v - 1
@njit
def h_inc_t_f(v):
    return v + 1
@njit
def h_dec_t_f(v):
    return v + 1

@njit
def h_add_f(v):
    return v
@njit
def h_sub_f(v):
    return -v
@njit
def h_not_n(v):
    return not v
@njit
def h_bitnot_i(v):
    return ~v
@njit
def h_bitnot_u(v):
    return (~v) + (1<<32)
@njit
def h_bitnot_vi(v):
    return ~v
@njit
def h_bitnot_vi(v):
    return ~v
@njit
def h_bitnot_vi(v):
    return ~v

@njit
def h_add_t_f(v):
    return +v
@njit
def h_sub_t_f(v):
    return -v
@njit
def h_add_t_vf(v):
    return +v
@njit
def h_sub_t_vf(v):
    return -v
@njit
def h_add_vf(v):
    return v
@njit
def h_sub_vf(v):
    return -v

@njit
def h_not_v(v):
    return not v
@njit
def h_not_t_n(v):
    return np.bitwise_not(v)

@njit
def h_add_f_f(a, b):
    return a + b
@njit
def h_sub_f_f(a, b):
    return a - b
@njit
def h_mul_f_f(a, b):
    return a * b
##@njit
def h_div_f_f(a, b):
    if b == 0:
        return sys.float_info.max
    return a / b
@njit
def h_mod_i_i(a, b):
    return a % b

@njit
def h_add_vf_f(a, b):
    return a + b
@njit
def h_sub_vf_f(a, b):
    return a - b
@njit
def h_mul_vf_f(a, b):
    return a * b
@njit
def h_div_vf_f(a, b):
    return a / b
@njit
def h_mod_vi_i(a, b):
    return a % b

@njit
def h_add_f_vf(a, b):
    return a + b
@njit
def h_sub_f_vf(a, b):
    return a - b
@njit
def h_mul_f_vf(a, b):
    return a * b
@njit
def h_div_f_vf(a, b):
    return a / b
@njit
def h_mod_i_vi(a, b):
    return a % b

@njit
def h_add_vf_vf(a, b):
    return a + b
@njit
def h_sub_vf_vf(a, b):
    return a - b
@njit
def h_mul_vf_vf(a, b):
    return a * b
@njit
def h_div_vf_vf(a, b):
    return a / b
@njit
def h_mod_vi_vi(a, b):
    return a % b

@njit
def h_and_n_n(a, b):
    return a and b
@njit
def h_or_n_n(a, b):
    return a or b

@njit
def h_and_v_v(a, b):
    return a and b
@njit
def h_or_v_v(a, b):
    return a or b

@njit
def h_and_t_n_n(a, b):
    b = np.broadcast_to(b, len(a))
    r = np.logical_and(a, b)
    return r
@njit
def h_and_n_t_n(a, b):
    a = np.broadcast_to(a, len(b))
    r = np.logical_and(a, b)
    return r
@njit
def h_and_t_n_t_n(a, b):
    r = np.logical_and(a, b)
    return r
@njit
def h_or_t_n_n(a, b):
    b = np.broadcast_to(b, len(a))
    r = np.logical_or(a, b)
    return r
@njit
def h_or_n_t_n(a, b):
    a = np.broadcast_to(a, len(b))
    r = np.logical_or(a, b)
    return r
@njit
def h_or_t_n_t_n(a, b):
    r = np.logical_or(a, b)
    return r

@njit
def h_add_t_f_t_f(a, b):
    return a + b
@njit
def h_add_t_vf_t_vf(a, b):
    return a + b
@njit
def h_add_f_t_f(a, b):
    a = np.broadcast_to(a, len(b))
    return a + b
@njit
def h_add_t_f_f(a, b):
    b = np.broadcast_to(b, len(a))
    r = a + b
    return r
@njit
def h_add_t_i_i(a, b):
    b = np.broadcast_to(b, len(a))
    r = a + b
    return r
@njit
def h_add_t_u_i(a, b):
    b = np.broadcast_to(b, len(a))
    r = a + b
    return r
@njit
def h_add_t_u_u(a, b):
    b = np.broadcast_to(b, len(a))
    r = a + b
    return r
@njit
def h_add_t_u_t_u(a, b):
    r = a + b
    return r
@njit
def h_add_f_t_vf(a, b):
    m = len(b)
    n = len(b[0])
    a = np.broadcast_to(a, (m, n))
    return a + b
@njit
def h_add_t_vf_f(a, b):
    m = len(a)
    n = len(a[0])
    b = np.broadcast_to(b, (m, n))
    return a + b
@njit
def h_add_t_f_vf(a, b):
    m = len(a)
    n = len(b)
    a = np.broadcast_to(a, (n, m)).T
    b = np.broadcast_to(b, (m, n))
    return a + b
@njit
def h_add_vf_t_f(a, b):
    m = len(b)
    n = len(a)
    a = np.broadcast_to(a, (m, n))
    b = np.broadcast_to(b, (n, m)).T
    return a + b
@njit
def h_add_t_vf_t_f(a, b):
    m = len(b)
    n = len(a[0])
    b = np.broadcast_to(b, (n, m)).T
    return a + b
@njit
def h_add_t_f_t_vf(a, b):
    m = len(a)
    n = len(b[0])
    a = np.broadcast_to(a, (n, m)).T
    return a + b
@njit
def h_add_vf_t_vf(a, b):
    a = np.broadcast_to(a, (len(b), len(a)))
    return a + b
@njit
def h_add_t_vf_vf(a, b):
    b = np.broadcast_to(b, (len(a), len(b)))
    return a + b
@njit
def h_add_t_vu_t_vu(a, b):
    r = a + b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_add_vu_vu(a, b):
    r = a + b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_add_i_i(a, b):
    return a + b

@njit
def h_sub_i_i(a, b):
    return a - b
@njit
def h_sub_i_t_i(a, b):
    return a - b
@njit
def h_sub_vi_vi(a, b):
    return a - b
@njit
def h_sub_vi_t_vi(a, b):
    return a - b
@njit
def h_sub_t_f_t_f(a, b):
    return a - b
@njit
def h_sub_t_vf_t_vf(a, b):
    return a - b
@njit
def h_sub_f_t_f(a, b):
    return a - b
@njit
def h_sub_t_f_f(a, b):
    return a - b
@njit
def h_sub_f_t_vf(a, b):
    return a - b
@njit
def h_sub_t_vf_f(a, b):
    r = a - b
    return r
@njit
def h_sub_t_f_vf(a, b):
    m = len(a)
    n = len(b)
    a = np.broadcast_to(a, (n, m)).T
    return a - b
@njit
def h_sub_vf_t_f(a, b):
    m = len(b)
    n = len(a)
    b = np.broadcast_to(b, (n, m)).T
    return a - b
@njit
def h_sub_vf_t_vf(a, b):
    return a - b
@njit
def h_sub_t_vf_vf(a, b):
    return a - b
@njit
def h_sub_t_vf_t_f(a, b):
    m = len(b)
    n = len(a[0])
    b = np.broadcast_to(b, (n, m)).T
    return a - b
@njit
def h_sub_t_f_t_vf(a, b):
    m = len(a)
    n = len(b[0])
    a = np.broadcast_to(a, (n, m)).T
    return a - b
@njit
def h_sub_mf_mf(a, b):
    return a - b
@njit
def h_sub_t_mf_mf(a, b):
    return a - b
@njit
def h_sub_i_f(a, b):
    return a - b

@njit
def h_mul_f_mf(a, b):
    return a * b
@njit
def h_mul_mf_f(a, b):
    return a * b
@njit
def h_mul_mf_t_mf(a, b):
    return a * b
@njit
def h_mul_i_f(a, b):
    return a * b
@njit
def h_mul_i_i(a, b):
    return a * b
@njit
def h_mul_u_u(a, b):
    return a * b
@njit
def h_mul_u_t_u(a, b):
    return a * b
@njit
def h_mul_t_f_i(a, b):
    return a * b
@njit
def h_mul_t_f_t_f(a, b):
    return a * b
@njit
def h_mul_t_vf_t_vf(a, b):
    return a * b
@njit
def h_mul_t_vf_t_f(a, b):
    return np.multiply(a.T, b).T
@njit
def h_mul_t_f_t_vf(a, b):
    return np.multiply(a, b.T).T
@njit
def h_mul_f_t_f(a, b):
    return a * b
@njit
def h_mul_t_f_f(a, b):
    return a * b
@njit
def h_mul_f_t_i(a, b):
    return a * b
@njit
def h_mul_f_t_vf(a, b):
    return a * b
@njit
def h_mul_t_vf_f(a, b):
    return a * b
##@njit
def h_mul_t_f_vf(a, b):
    a = a.reshape(len(a), 1)
    return a * b
##@njit
def h_mul_vf_t_f(a, b):
    b = b.reshape(len(b), 1)
    return a * b
@njit
def h_mul_vf_t_vf(a, b):
    return a * b
@njit
def h_mul_t_vf_vf(a, b):
    return a * b
@njit
def h_mul_t_vu_vu(a, b):
    r = a * b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_mul_t_vu_t_vu(a, b):
    r = a * b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_mul_vu_vu(a, b):
    r = a * b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_mul_vi_vi(a, b):
    r = a * b
    r = np.bitwise_and(r, 0xffffffff)
    return int(r)
@njit
def h_mul_vd_vd(a, b):
    r = a * b
    return r
@njit
def h_mul_vb_vb(a, b):
    r = a * b
    return r
@njit
def h_mul_t_i_f(a, b):
    r = a * b
    return r
@njit
def h_mul_t_i_t_f(a, b):
    r = a * b
    return r
@njit
def h_mul_t_f_t_i(a, b):
    r = a * b
    return r
@njit
def h_mul_t_u_i(a, b):
    r = a * b
    return r
@njit
def h_mul_t_u_t_u(a, b):
    r = a * b
    return r
@njit
def h_mul_t_u_t_vu(a, b):
    r = a * b
    return r

@njit
def h_div_i_i(a, b):
    return a // b
@njit
def h_div_t_i_i(a, b):
    return a // b
##@njit
def h_div_t_f_t_f(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_t_vf_t_vf(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_f_t_f(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_t_f_f(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_f_t_vf(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_t_vf_f(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_t_f_vf(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_vf_t_f(a, b):
    b = b + sys.float_info.epsilon
    return a / b
##@njit
def h_div_vf_t_vf(a, b):
    b = b + sys.float_info.epsilon
    r = a / b
    return r
##@njit
def h_div_t_vf_vf(a, b):
    b = b + sys.float_info.epsilon
    r = a / b
    return r
##@njit
def h_div_t_vf_t_f(a, b):
    r = np.divide(a.T, b + sys.float_info.epsilon).T
    return r
##@njit
def h_div_t_mf_t_f(a, b):
    r = np.divide(a.T, b + sys.float_info.epsilon).T
    return r

@njit
def h_mod_t_i_t_i(a, b):
    return a % b
@njit
def h_mod_t_vi_t_vi(a, b):
    return a % b
@njit
def h_mod_i_t_i(a, b):
    return a % b
@njit
def h_mod_t_i_i(a, b):
    return a % b
@njit
def h_mod_i_t_vi(a, b):
    return a % b
@njit
def h_mod_t_vi_i(a, b):
    return a % b
@njit
def h_mod_t_i_vi(a, b):
    return a % b
@njit
def h_mod_vi_t_i(a, b):
    return a % b
@njit
def h_mod_vi_t_vi(a, b):
    return a % b
@njit
def h_mod_t_vi_vi(a, b):
    return a % b

@njit
def h_bitand_i_i(a, b):
    return a & b
@njit
def h_bitand_t_vi_i(a, b):
    return a & b
@njit
def h_bitand_t_u_u(a, b):
    return a & b
@njit
def h_bitand_t_vu_vu(a, b):
    return a & b
@njit
def h_bitor_i_i(a, b):
    return a | b

@njit
def h_bitxor_i_i(a, b):
    return a ^ b
@njit
def h_bitxor_t_u_t_u(a, b):
    r = a ^ b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_bitxor_t_vu_t_vu(a, b):
    r = a ^ b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_bitxor_vu_vu(a, b):
    r = a ^ b
    r = np.bitwise_and(r, 0xffffffff)
    return r

@njit
def h_lshift_i_i(a, b):
    return a << b
@njit
def h_rshift_i_i(a, b):
    return a >> b
@njit
def h_lshift_t_u_i(a, b):
    r = a << b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_rshift_t_u_i(a, b):
    r = a >> b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_rshift_t_vu_i(a, b):
    r = a >> b
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_rshift_vu_i(a, b):
    r = a >> b
    r = np.bitwise_and(r, 0xffffffff)
    return r

@njit
def h_less_than_n_n(a, b):
    return a < b
@njit
def h_greater_than_n_n(a, b):
    return a > b
@njit
def h_less_equal_than_n_n(a, b):
    return a <= b
@njit
def h_greater_equal_than_n_n(a, b):
    return a >= b

@njit
def h_less_than_v_v(a, b):
    return a < b
@njit
def h_greater_than_v_v(a, b):
    return a > b
@njit
def h_less_equal_than_v_v(a, b):
    return a <= b
@njit
def h_greater_equal_than_v_v(a, b):
    return a >= b

@njit
def h_less_than_t_n_t_n(a, b):
    return a < b
@njit
def h_greater_than_t_n_t_n(a, b):
    return a > b
@njit
def h_less_equal_than_t_n_t_n(a, b):
    return a <= b
@njit
def h_greater_equal_than_t_n_t_n(a, b):
    return a >= b
@njit
def h_less_than_t_n_n(a, b):
    return a < b
@njit
def h_greater_than_t_n_n(a, b):
    return a > b
@njit
def h_less_equal_than_t_n_n(a, b):
    return a <= b
@njit
def h_greater_equal_than_t_n_n(a, b):
    return a >= b
@njit
def h_less_than_n_t_n(a, b):
    return a < b
@njit
def h_greater_than_n_t_n(a, b):
    return a > b
@njit
def h_less_equal_than_n_t_n(a, b):
    return a <= b
@njit
def h_greater_equal_than_n_t_n(a, b):
    return a >= b

@njit
def h_equal_n_n(a, b):
    return a == b
@njit
def h_not_equal_n_n(a, b):
    return a != b
@njit
def h_equal_v_v(a, b):
    return np.all(a == b)
@njit
def h_not_equal_v_v(a, b):
    return np.any(a != b)

@njit
def h_equal_t_n_n(a, b):
    return a == b
@njit
def h_not_equal_t_n_n(a, b):
    return a != b
@njit
def h_equal_t_v_v(a, b):
    return a == b
@njit
def h_not_equal_t_v_v(a, b):
    return a != b
@njit
def h_equal_n_t_n(a, b):
    return a == b
@njit
def h_not_equal_n_t_n(a, b):
    return a != b
@njit
def h_equal_t_n_t_n(a, b):
    return a == b
@njit
def h_not_equal_t_n_t_n(a, b):
    return a != b

@njit
def h_broadcast_t_f_f(t, v):
    return np.broadcast_to(v, len(t))
@njit
def h_broadcast_t_b_b(t, v):
    return np.broadcast_to(v, len(t))
@njit
def h_broadcast_t_i_i(t, v):
    return np.broadcast_to(v, len(t))
##@njit
def h_broadcast_t_f2_f2(t, v):
    return np.broadcast_to(v, (len(t), 2))
##@njit
def h_broadcast_t_f3_f3(t, v):
    return np.broadcast_to(v, (len(t), 3))
##@njit
def h_broadcast_t_f4_f4(t, v):
    return np.broadcast_to(v, (len(t), 4))

@njit
def h_broadcast_f(ct, v):
    return np.broadcast_to(v, ct)
@njit
def h_broadcast_b(ct, v):
    return np.broadcast_to(v, ct)
@njit
def h_broadcast_i(ct, v):
    return np.broadcast_to(v, ct)
##@njit
def h_broadcast_f2(ct, v):
    return np.broadcast_to(v, (ct, 2))
##@njit
def h_broadcast_f3(ct, v):
    return np.broadcast_to(v, (ct, 3))
##@njit
def h_broadcast_f4(ct, v):
    return np.broadcast_to(v, (ct, 4))

@njit
def h_copy_f(v):
    return v
@njit
def h_copy_f2(v):
    return np.copy(v)
@njit
def h_copy_f3(v):
    return np.copy(v)
@njit
def h_copy_f4(v):
    return np.copy(v)
@njit
def h_copy_t_f(v):
    return np.copy(v)
@njit
def h_copy_t_f2(v):
    return np.copy(v)
@njit
def h_copy_t_f3(v):
    return np.copy(v)
@njit
def h_copy_t_f4(v):
    return np.copy(v)
@njit
def h_copy_t_i(v):
    return np.copy(v)
@njit
def h_copy_t_i2(v):
    return np.copy(v)
@njit
def h_copy_t_i3(v):
    return np.copy(v)
@njit
def h_copy_t_i4(v):
    return np.copy(v)

@njit
def h_cast_f_h(v):
    return float(v)
@njit
def h_cast_f_d(v):
    return float(v)
@njit
def h_cast_f_i(v):
    return float(v)
@njit
def h_cast_f_u(v):
    return float(v)
@njit
def h_cast_f_b(v):
    return 1.0 if v else 0.0

@njit
def h_cast_i_h(v):
    return int(v)
@njit
def h_cast_i_f(v):
    return int(v)
@njit
def h_cast_i_d(v):
    return int(v)
@njit
def h_cast_i_u(v):
    return int(v)
@njit
def h_cast_i_b(v):
    return 1 if v else 0

@njit
def h_cast_u_h(v):
    r = int(v)
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_cast_u_f(v):
    r = int(v)
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_cast_u_d(v):
    r = v.astype(np.uint32)
    r = np.bitwise_and(r, 0xffffffff)
    return r
@njit
def h_cast_u_i(v):
    r = v if v >= 0 else v + (1 << 32)
    return r
@njit
def h_cast_u_b(v):
    return 1 if v else 0

@njit
def h_cast_b_h(v):
    return int(v) != 0
@njit
def h_cast_b_f(v):
    return int(v) != 0
@njit
def h_cast_b_d(v):
    return int(v) != 0
@njit
def h_cast_b_i(v):
    return int(v) != 0
@njit
def h_cast_b_u(v):
    return int(v) != 0

@njit
def h_cast_f2_i2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_f2_h2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_f2_d2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_f2_u2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_f2_b2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_f2_f(v):
    return np.array([v, v])

@njit
def h_cast_f3_i3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_f3_h3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_f3_d3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_f3_u3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_f3_b3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])

@njit
def h_cast_f4_i4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_f4_h4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_f4_d4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_f4_u4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_f4_b4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])

@njit
def h_cast_d2_i2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_d2_h2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_d2_f2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_d2_u2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_d2_b2(v):
    return np.array([float(v[0]), float(v[1])])

@njit
def h_cast_d3_i3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_d3_h3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_d3_f3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_d3_u3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_d3_b3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])

@njit
def h_cast_d4_i4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_d4_h4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_d4_f4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_d4_u4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_d4_b4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])

@njit
def h_cast_i2_h2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_i2_f2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_i2_d2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_i2_u2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_i2_b2(v):
    return np.array([int(v[0]), int(v[1])])

@njit
def h_cast_i3_h3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_i3_f3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_i3_d3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_i3_u3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_i3_b3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])

@njit
def h_cast_i4_h4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_i4_f4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_i4_d4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_i4_u4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_i4_b4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])

##@njit
def h_cast_u2_h2(v):
    return np.array([int(v[0]), int(v[1])]).astype(np.uint32)
##@njit
def h_cast_u2_f2(v):
    return np.array([int(v[0]), int(v[1])]).astype(np.uint32)
##@njit
def h_cast_u2_d2(v):
    return np.array([int(v[0]), int(v[1])]).astype(np.uint32)
##@njit
def h_cast_u2_i2(v):
    return np.array([int(v[0]), int(v[1])]).astype(np.uint32)
##@njit
def h_cast_u2_b2(v):
    return np.array([int(v[0]), int(v[1])]).astype(np.uint32)

##@njit
def h_cast_t_u_t_i(v):
    return v.astype(np.uint32)
##@njit
def h_cast_t_u2_t_i2(v):
    return v.astype(np.uint32)
##@njit
def h_cast_t_u_t_f(v):
    return v.astype(np.uint32)
##@njit
def h_cast_t_u2_t_f2(v):
    return v.astype(np.uint32)

##@njit
def h_cast_u3_h3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])]).astype(np.uint32)
##@njit
def h_cast_u3_f3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])]).astype(np.uint32)
##@njit
def h_cast_u3_d3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])]).astype(np.uint32)
##@njit
def h_cast_u3_i3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])]).astype(np.uint32)
##@njit
def h_cast_u3_b3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])]).astype(np.uint32)

##@njit
def h_cast_u4_h4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])]).astype(np.uint32)
##@njit
def h_cast_u4_f4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])]).astype(np.uint32)
##@njit
def h_cast_u4_d4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])]).astype(np.uint32)
##@njit
def h_cast_u4_i4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])]).astype(np.uint32)
##@njit
def h_cast_u4_b4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])]).astype(np.uint32)

@njit
def h_cast_h2_f2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_h2_d2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_h2_i2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_h2_u2(v):
    return np.array([float(v[0]), float(v[1])])
@njit
def h_cast_h2_b2(v):
    return np.array([float(v[0]), float(v[1])])

@njit
def h_cast_h3_f3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_h3_d3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_h3_i3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_h3_u3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])
@njit
def h_cast_h3_b3(v):
    return np.array([float(v[0]), float(v[1]), float(v[2])])

@njit
def h_cast_h4_f4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_h4_d4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_h4_i4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_h4_u4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
@njit
def h_cast_h4_b4(v):
    return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])])

@njit
def h_cast_b2_h2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_b2_f2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_b2_d2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_b2_i2(v):
    return np.array([int(v[0]), int(v[1])])
@njit
def h_cast_b2_u2(v):
    return np.array([int(v[0]), int(v[1])])

@njit
def h_cast_b3_h3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_b3_f3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_b3_d3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_b3_i3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])
@njit
def h_cast_b3_u3(v):
    return np.array([int(v[0]), int(v[1]), int(v[2])])

@njit
def h_cast_b4_h4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_b4_f4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_b4_d4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_b4_i4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])
@njit
def h_cast_b4_u4(v):
    return np.array([int(v[0]), int(v[1]), int(v[2]), int(v[3])])

@njit
def h_cast_f_f(v):
    return v
@njit
def h_cast_f2_f2(v):
    return v
@njit
def h_cast_f3_f3(v):
    return v
@njit
def h_cast_f4_f4(v):
    return v

@njit
def h_cast_f2_f3(v):
    return np.array([v[0], v[1]])
@njit
def h_cast_f3_f4(v):
    return np.array([v[0], v[1], v[2]])

@njit
def h_cast_i_i(v):
    return v
@njit
def h_cast_i2_i2(v):
    return v
@njit
def h_cast_i3_i3(v):
    return v
@njit
def h_cast_i4_i4(v):
    return v

@njit
def h_cast_b_b(v):
    return v
@njit
def h_cast_b2_b2(v):
    return v
@njit
def h_cast_b3_b3(v):
    return v
@njit
def h_cast_b4_b4(v):
    return v

##@njit
def h_cast_f3x3_i_x9(v):
    return h_f3x3_n_n_n_n_n_n_n_n_n(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
##@njit
def h_cast_t_f3x3_t_f_x9(v):
    return h_t_f3x3_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
##@njit
def h_cast_f4x4_i_x16(v):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15])

@njit
def h_cast_t_f_t_f(v):
    return v
@njit
def h_cast_t_f2_t_f2(v):
    return v
@njit
def h_cast_t_f3_t_f3(v):
    return v
@njit
def h_cast_t_f4_t_f4(v):
    return v
@njit
def h_cast_t_i_t_i(v):
    return v
@njit
def h_cast_t_i2_t_i2(v):
    return v
@njit
def h_cast_t_i3_t_i3(v):
    return v
@njit
def h_cast_t_i4_t_i4(v):
    return v
@njit
def h_cast_t_f_t_i(v):
    return v.astype(np.float64)
@njit
def h_cast_t_f_t_u(v):
    return v.astype(np.float64)
@njit
def h_cast_t_i_t_f(v):
    return v.astype(np.int32)
@njit
def h_cast_t_i_t_f(v):
    return v.astype(np.int32)
##@njit
def h_cast_t_i2_t_f2(v):
    return v.astype(int)
##@njit
def h_cast_t_i3_t_f3(v):
    return v.astype(int)
##@njit
def h_cast_t_i4_t_f4(v):
    return v.astype(int)
##@njit
def h_cast_t_f2_t_u2(v):
    return v.astype(float)
##@njit
def h_cast_t_f3_t_u3(v):
    return v.astype(float)
##@njit
def h_cast_t_f_t_b(v):
    return v.astype(float)

@njit
def h_f2_n_n(x, y):
    return np.array([x, y])
@njit
def h_t_f2_t_n_n(x, y):
    nx = len(x)
    ny = 1
    mn = max(nx, ny)
    xs = x
    ys = np.broadcast_to(y, mn)
    nv = np.column_stack((xs, ys))
    return nv
@njit
def h_t_f2_n_t_n(x, y):
    nx = 1
    ny = len(y)
    mn = max(nx, ny)
    xs = np.broadcast_to(x, mn)
    ys = y
    nv = np.column_stack((xs, ys))
    return nv
@njit
def h_t_f2_t_n_t_n(x, y):
    nv = np.column_stack((x, y))
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
            xs = np.repeat(x, mn)
        ys = y
        if not yisv:
            ys = np.repeat(y, mn)
        nv = np.column_stack((xs, ys))
        return nv
    else:
        if type(x) == np.ndarray:
            return np.array([x[0], x[1], y])
        elif type(y) == np.ndarray:
            return np.array([x, y[0], y[1]])
        else:
            return np.array([x, y, 1.0])
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
            xs = np.repeat(x, mn)
        ys = y
        if not yisv:
            ys = np.repeat(y, mn)
        zs = z
        if not zisv:
            zs = np.repeat(z, mn)
        nv = np.column_stack((xs, ys, zs))
        return nv
    else:
        return np.array([x, y, z])
@njit
def h_f3_n2_n(x, y):
    return np.array([x[0], x[1], y])
@njit
def h_f3_n_n2(x, y):
    return np.array([x, y[0], y[1]])
@njit
def h_f3_n_n(x, y):
    return np.array([x, y, 1.0])
@njit
def h_f3_n_n_n(x, y, z):
    return np.array([x, y, z])

@njit
def h_t_f3_t_n_t_n_t_n(x, y, z):
    return np.column_stack((x, y, z))
@njit
def h_t_f3_t_n2_t_n(x, y):
    nv = np.column_stack((x, y))
    return nv
@njit
def h_t_f3_t_n_t_n2(x, y):
    nv = np.column_stack((x, y))
    return nv
@njit
def h_t_f3_t_n2_n(x, y):
    ys = np.broadcast_to(y, len(x))
    return np.column_stack((x, ys))
@njit
def h_t_f3_t_n_t_n_n(x, y, z):
    zs = np.broadcast_to(z, len(x))
    return np.column_stack((x, y, zs))
@njit
def h_t_f3_t_n_n_n(x, y, z):
    ys = np.broadcast_to(y, len(x))
    zs = np.broadcast_to(z, len(x))
    return np.column_stack((x, ys, zs))
@njit
def h_t_f3_t_n_n_t_n(x, y, z):
    ys = np.broadcast_to(y, len(x))
    return np.column_stack((x, ys, z))
@njit
def h_t_f3_n_t_n_n(x, y, z):
    xs = np.broadcast_to(x, len(y))
    zs = np.broadcast_to(z, len(y))
    return np.column_stack((xs, y, zs))
@njit
def h_t_f3_n_t_n_t_n(x, y, z):
    xs = np.broadcast_to(x, len(y))
    return np.column_stack((xs, y, z))
@njit
def h_t_f3_n_t_n2(x, yz):
    xs = np.broadcast_to(x, len(yz))
    return np.column_stack((xs, yz))

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
            xs = np.repeat(x, mn)
        ys = y
        if not yisv:
            ys = np.repeat(y, mn)
        nv = np.column_stack((xs, ys))
        return nv
    else:
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return np.array([x[0], x[1], y[0], y[1]])
        elif type(x) == np.ndarray:
            return np.array([x[0], x[1], x[2], y])
        elif type(y) == np.ndarray:
            return np.array([x, y[0], y[1], y[2]])
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
            xs = np.repeat(x, mn)
        ys = y
        if not yisv:
            ys = np.repeat(y, mn)
        zs = z
        if not zisv:
            zs = np.repeat(z, mn)
        ws = np.repeat(1.0, mn)
        nv = np.column_stack((xs, ys, zs, ws))
        return nv
    else:
        return np.array([x, y, z, 1.0])
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
            xs = np.repeat(x, mn)
        ys = y
        if not yisv:
            ys = np.repeat(y, mn)
        zs = z
        if not zisv:
            zs = np.repeat(z, mn)
        ws = w
        if not wisv:
            ws = np.repeat(w, mn)
        nv = np.column_stack((xs, ys, zs, ws))
        return nv
    else:
        return np.array([x, y, z, w])

@njit
def h_f4_n3_n(x, y):
    return np.array([x[0], x[1], x[2], y])
@njit
def h_f4_n2_n2(x, y):
    return np.array([x[0], x[1], y[0], y[1]])
@njit
def h_f4_n_n3(x, y):
    return np.array([x, y[0], y[1], y[2]])
@njit
def h_f4_n2_n_n(x, y, z):
    return np.array([x[0], x[1], y, z])
@njit
def h_f4_n_n_n(x, y, z):
    return np.array([x, y, z, 1.0])
@njit
def h_f4_n_n_n_n(x, y, z, w):
    return np.array([x, y, z, w])

@njit
def h_t_f4_t_n_t_n_t_n_t_n(x, y, z, w):
    return np.column_stack((x, y, z, w))
@njit
def h_t_f4_t_n_t_n_t_n_n(x, y, z, w):
    w = np.broadcast_to(w, len(x))
    return np.column_stack((x, y, z, w))
@njit
def h_t_f4_t_n2_t_n2(x, y):
    return np.column_stack((x, y))
@njit
def h_t_f4_t_n2_t_n_n(xy, z, w):
    w = np.broadcast_to(w, len(xy))
    return np.column_stack((xy, z, w))
@njit
def h_t_f4_t_n2_n_t_n(xy, z, w):
    z = np.broadcast_to(z, len(xy))
    return np.column_stack((xy, z, w))
@njit
def h_t_f4_t_n2_n_n(xy, z, w):
    z = np.broadcast_to(z, len(xy))
    w = np.broadcast_to(w, len(xy))
    return np.column_stack((xy, z, w))
@njit
def h_t_f4_t_n3_n(x, y):
    y = np.broadcast_to(y, len(x))
    return np.column_stack((x, y))
@njit
def h_t_f4_t_n_t_n3(x, y):
    return np.column_stack((x, y))
@njit
def h_t_f4_t_n3_t_n(x, y):
    return np.column_stack((x, y))
@njit
def h_t_f4_t_n_t_n_n_n(x, y, z, w):
    zs = np.broadcast_to(z, len(x))
    ws = np.broadcast_to(w, len(x))
    return np.column_stack((x, y, zs, ws))
@njit
def h_t_f4_t_n_n_t_n_n(x, y, z, w):
    ys = np.broadcast_to(y, len(x))
    ws = np.broadcast_to(w, len(x))
    return np.column_stack((x, ys, z, ws))
@njit
def h_t_f4_t_n_n_n_t_n(x, y, z, w):
    ys = np.broadcast_to(y, len(x))
    zs = np.broadcast_to(z, len(x))
    return np.column_stack((x, ys, zs, w))
@njit
def h_t_f4_n_t_n_t_n_n(x, y, z, w):
    xs = np.broadcast_to(x, len(y))
    ws = np.broadcast_to(w, len(y))
    return np.column_stack((xs, y, z, ws))
@njit
def h_t_f4_n_t_n_n_t_n(x, y, z, w):
    xs = np.broadcast_to(x, len(w))
    zs = np.broadcast_to(z, len(w))
    return np.column_stack((xs, y, zs, w))
@njit
def h_t_f4_n_n_t_n_t_n(x, y, z, w):
    xs = np.broadcast_to(x, len(w))
    ys = np.broadcast_to(y, len(w))
    return np.column_stack((xs, ys, z, w))
@njit
def h_t_f4_t_n_n_n_n(x, y, z, w):
    ys = np.broadcast_to(y, len(x))
    zs = np.broadcast_to(z, len(x))
    ws = np.broadcast_to(w, len(x))
    return np.column_stack((x, ys, zs, ws))
@njit
def h_t_f4_n_t_n_n_n(x, y, z, w):
    xs = np.broadcast_to(x, len(y))
    zs = np.broadcast_to(z, len(y))
    ws = np.broadcast_to(w, len(y))
    return np.column_stack((xs, y, zs, ws))
@njit
def h_t_f4_n_n_t_n_n(x, y, z, w):
    xs = np.broadcast_to(x, len(z))
    ys = np.broadcast_to(y, len(z))
    ws = np.broadcast_to(w, len(z))
    return np.column_stack((xs, ys, z, ws))
@njit
def h_t_f4_n_n_n_t_n(x, y, z, w):
    xs = np.broadcast_to(x, len(w))
    ys = np.broadcast_to(y, len(w))
    zs = np.broadcast_to(z, len(w))
    return np.column_stack((xs, ys, zs, w))
@njit
def h_t_f4_n3_t_n(xyz, w):
    xyz = np.broadcast_to(xyz, (len(w), 3))
    return np.column_stack((xyz, w))
@njit
def h_t_f4_t_n2_t_n_n(xy, z, w):
    ws = np.broadcast_to(w, len(z))
    return np.column_stack((xy, z, ws))
@njit
def h_t_f4_n_t_n2_n(x, yz, w):
    xs = np.broadcast_to(x, len(yz))
    ws = np.broadcast_to(w, len(yz))
    return np.column_stack((xs, yz, ws))

@njit
def h_d2_n_n(x, y):
    return h_f2_n_n(x, y)

@njit
def h_d3_n2_n(x, y):
    return h_f3_n2_n(x, y)
@njit
def h_d3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)

@njit
def h_d4_n3_n(x, y):
    return h_f4_n3_n(x, y)
@njit
def h_d4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
@njit
def h_d4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

@njit
def h_h2_n_n(x, y):
    return h_f2_n_n(x, y)

@njit
def h_h3_n2_n(x, y):
    return h_f3_n2_n(x, y)
@njit
def h_h3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)

@njit
def h_h4_n3_n(x, y):
    return h_f4_n3_n(x, y)
@njit
def h_h4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
@njit
def h_h4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

@njit
def h_i2_n_n(x, y):
    return h_f2_n_n(x, y)

@njit
def h_i3_n2_n(x, y):
    return h_f3_n2_n(x, y)
@njit
def h_i3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)
@njit
def h_t_i3_t_n2_n(x, y):
    return h_t_f3_t_n2_n(x, y)

@njit
def h_i4_n3_n(x, y):
    return h_f4_n3_n(x, y)
@njit
def h_i4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
@njit
def h_i4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

@njit
def h_b2_n_n(x, y):
    return h_f2_n_n(x, y)

@njit
def h_b3_n2_n(x, y):
    return h_f3_n2_n(x, y)
@njit
def h_b3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)

@njit
def h_b4_n3_n(x, y):
    return h_f4_n3_n(x, y)
@njit
def h_b4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
@njit
def h_b4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

@njit
def h_u2_n_n(x, y):
    return h_f2_n_n(x, y)
@njit
def h_t_u2_t_n_t_n(x, y):
    return np.column_stack((x, y))

@njit
def h_u3_n2_n(x, y):
    return h_f3_n2_n(x, y)
@njit
def h_u3_n_n_n(x, y, z):
    return h_f3_n_n_n(x, y, z)
@njit
def h_t_u3_t_n_t_n_t_n(x, y, z):
    return np.column_stack((x, y, z))

@njit
def h_u4_n3_n(x, y):
    return h_f4_n3_n(x, y)
@njit
def h_u4_n2_n_n(x, y, z):
    return h_f4_n2_n_n(x, y, z)
@njit
def h_u4_n_n_n_n(x, y, z, w):
    return h_f4_n_n_n_n(x, y, z, w)

@njit
def h_f2x2_n_n_n_n(x1, y1, x2, y2):
    return np.asarray([[x1, y1], [x2, y2]])
@njit
def h_t_f2x2_t_n_t_n_t_n_t_n(x1, y1, x2, y2):
    r1 = np.column_stack((x1, y1))
    r2 = np.column_stack((x2, y2))
    return np.stack((r1, r2), axis=1)

@njit
def h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return np.asarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
@njit
def h_f3x3_n3_n3_n3(x, y, z):
    return np.stack((x, y, z))
@njit
def h_t_f3x3_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n_t_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    r1 = np.column_stack((x1, y1, z1))
    r2 = np.column_stack((x2, y2, z2))
    r3 = np.column_stack((x3, y3, z3))
    return np.stack((r1, r2, r3), axis=1)
@njit
def h_t_f3x3_t_n3_t_n3_t_n3(v1, v2, v3):
    return np.stack((v1, v2, v3), axis=1)

@njit
def h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return np.asarray([[x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]])
@njit
def h_f4x4_n4_n4_n4_n4(x, y, z, w):
    return np.stack((x, y, z, w))
@njit
def h_t_f4x4_t_n_t_n_t_n_n_t_n_t_n_t_n_n_t_n_t_n_t_n_n_n_n_n_n(xs1, ys1, zs1, w1, xs2, ys2, zs2, w2, xs3, ys3, zs3, w3, x4, y4, z4, w4):
    n = len(xs1)
    ws1 = np.broadcast_to(w1, n)
    ws2 = np.broadcast_to(w2, n)
    ws3 = np.broadcast_to(w3, n)
    xs4 = np.broadcast_to(x4, n)
    ys4 = np.broadcast_to(y4, n)
    zs4 = np.broadcast_to(z4, n)
    ws4 = np.broadcast_to(w4, n)
    r1 = np.column_stack((xs1, ys1, zs1, ws1))
    r2 = np.column_stack((xs2, ys2, zs2, ws2))
    r3 = np.column_stack((xs3, ys3, zs3, ws3))
    r4 = np.column_stack((xs4, ys4, zs4, ws4))
    return np.stack((r1, r2, r3, r4), axis=1)

@njit
def h_d2x2_n_n_n_n(x1, y1, x2, y2):
    return np.asarray([[x1, y1], [x2, y2]])

@njit
def h_d3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return np.asarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

@njit
def h_d4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return np.asarray([[x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]])

@njit
def h_h2x2_n_n_n_n(x1, y1, x2, y2):
    return np.asarray([[x1, y1], [x2, y2]])

@njit
def h_h3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return np.asarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

@njit
def h_h4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return np.asarray([[x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]])

@njit
def h_i2x2_n_n_n_n(x1, y1, x2, y2):
    return h_f2x2_n_n_n_n(x1, y1, x2, y2)
@njit
def h_i3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3)
@njit
def h_i4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4)

@njit
def h_b2x2_n_n_n_n(x1, y1, x2, y2):
    return h_f2x2_n_n_n_n(x1, y1, x2, y2)
@njit
def h_b3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3)
@njit
def h_b4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4)

@njit
def h_u2x2_n_n_n_n(x1, y1, x2, y2):
    return h_f2x2_n_n_n_n(x1, y1, x2, y2)
@njit
def h_u3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return h_f3x3_n_n_n_n_n_n_n_n_n(x1, y1, z1, x2, y2, z2, x3, y3, z3)
@njit
def h_u4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4):
    return h_f4x4_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n_n(x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3, x4, y4, z4, w4)

def h_f2_defval():
    return np.asarray([0.0, 0.0])
def h_f3_defval():
    return np.asarray([0.0, 0.0, 0.0])
def h_f4_defval():
    return np.asarray([0.0, 0.0, 0.0, 0.0])
def h_f2x2_defval():
    return np.asarray([[0.0, 0.0], [0.0, 0.0]])
def h_f3x3_defval():
    return np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
def h_f4x4_defval():
    return np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
def h_af_defval(num):
    return np.repeat(0.0, num)
def h_af2_defval(num):
    return np.tile(0.0, (num, 2))
def h_af3_defval(num):
    return np.tile(0.0, (num, 3))
def h_af4_defval(num):
    return np.tile(0.0, (num, 4))
def h_ab_defval(num):
    return np.repeat(False, num)
def h_ai_defval(num):
    return np.repeat(0, num)

##@njit
def Texture2D_Sample_n_v(tex, sampler, coord):
    texSize = np.asarray(tex.shape)[0:2]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == np.ndarray:
        if len(col) == 3:
            return np.asarray([col[0], col[1], col[2], 1.0])
        else:
            return col
    return np.asarray([col, col, col, 1.0])
##@njit
def Texture2D_SampleBias_n_v_n(tex, sampler, coord, bias):
    return Texture2D_Sample_n_v(tex, sampler, coord)
##@njit
def Texture2D_Sample_n_t_v(tex, sampler, coord):
    texSize = np.asarray(tex.shape)[0:2]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    icoord = tuple(icoord.transpose())
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = np.column_stack((cols, np.broadcast_to(1.0, len(cols))))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        nv = np.column_stack((cols, cols, cols, np.broadcast_to(1.0, len(cols))))
        return nv
##@njit
def Texture2D_SampleBias_n_t_v_n(tex, sampler, coord, bias):
    return Texture2D_Sample_n_t_v(tex, sampler, coord)
##@njit
def Texture2D_SampleLevel_n_v_n(tex, sampler, coord, level):
    texSize = np.asarray(tex.shape)[0:2]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == np.ndarray:
        if len(col) == 3:
            return np.asarray([col[0], col[1], col[2], 1.0])
        else:
            return col
    return np.asarray([col, col, col, 1.0])
##@njit
def Texture2D_SampleLevel_n_t_v_n(tex, sampler, coord, level):
    texSize = np.asarray(tex.shape)[0:2]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    icoord = tuple(icoord.transpose())
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = np.column_stack((cols, np.broadcast_to(1.0, len(cols))))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        nv = np.column_stack((cols, cols, cols, np.broadcast_to(1.0, len(cols))))
        return nv
##@njit
def Texture2D_SampleGrad_n_t_v_t_v_t_v(tex, sampler, coord, ddx, ddy):
    texSize = np.asarray(tex.shape)[0:2]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    icoord = tuple(icoord.transpose())
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = np.column_stack((cols, np.broadcast_to(1.0, len(cols))))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        nv = np.column_stack((cols, cols, cols, np.broadcast_to(1.0, len(cols))))
        return nv
##@njit
def Texture2D_Load_t_v(tex, coord):
    texSize = np.asarray(tex.shape)
    n = len(coord)
    icoord = np.mod(coord[..., 0:2].astype(int), texSize[0:2])
    icoord = tuple(icoord.transpose())
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = np.column_stack((cols, np.broadcast_to(1.0, len(cols))))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        nv = np.column_stack((cols, cols, cols, np.broadcast_to(1.0, len(cols))))
        return nv
##@njit
def TextureCube_Sample_n_v(tex, sampler, coord):
    texSize = np.asarray(tex.shape)[0:2]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord[0:2], 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == np.ndarray:
        if len(col) == 3:
            return np.asarray([col[0], col[1], col[2], 1.0])
        else:
            return col
    return np.asarray([col, col, col, 1.0])
##@njit
def TextureCube_Sample_n_t_v(tex, sampler, coord):
    texSize = np.asarray(tex.shape)[0:3]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord[..., 0:3], 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    icoord = tuple(icoord.transpose())
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = np.column_stack((cols, np.broadcast_to(1.0, len(cols))))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        nv = np.column_stack((cols, cols, cols, np.broadcast_to(1.0, len(cols))))
        return nv
##@njit
def TextureCube_SampleLevel_n_t_v_n(tex, sampler, coord, level):
    return TextureCube_Sample_n_t_v(tex, sampler, coord)
##@njit
def Texture3D_Sample_n_v(tex, sampler, coord):
    texSize = np.asarray(tex.shape)[0:3]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    col = tex[tuple(icoord)] / 255.0
    if type(col) == np.ndarray:
        if len(col) == 3:
            return np.asarray([col[0], col[1], col[2], 1.0])
        else:
            return col
    return np.asarray([col, col, col, 1.0])
##@njit
def Texture3D_Sample_n_t_v(tex, sampler, coord):
    texSize = np.asarray(tex.shape)[0:3]
    coord[np.isnan(coord[...])] = 0.0
    coord = np.mod(coord, 1.0)
    icoord = np.mod((coord * texSize).astype(int), texSize)
    icoord = tuple(icoord.transpose())
    vs = tex[icoord]
    if len(vs.shape) == 2:
        if len(vs[0]) == 3:
            cols = vs / 255.0
            nv = np.column_stack((cols, np.broadcast_to(1.0, len(cols))))
            return nv
        else:
            return vs / 255.0
    else:
        cols = vs / 255.0
        nv = np.column_stack((cols, cols, cols, np.broadcast_to(1.0, len(cols))))
        return nv
