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

#----------------------------------------
# these code generated from gen_hlsl_lib_numpy_swizzle.dsl
#---begin---

@njit
def swizzle_n_x(v):
    return v
@njit
def swizzle_n_xx(v):
    return np.asarray([v, v])
@njit
def swizzle_n_xxx(v):
    return np.asarray([v, v, v])
@njit
def swizzle_n_xxxx(v):
    return np.asarray([v, v, v, v])
@njit
def swizzle_n2_x(v):
    return v[0]
@njit
def swizzle_n2_y(v):
    return v[1]
@njit
def swizzle_n2_xx(v):
    return np.asarray([v[0], v[0]])
@njit
def swizzle_n2_xy(v):
    return np.asarray([v[0], v[1]])
@njit
def swizzle_n2_yx(v):
    return np.asarray([v[1], v[0]])
@njit
def swizzle_n2_yy(v):
    return np.asarray([v[1], v[1]])
@njit
def swizzle_n2_xxx(v):
    return np.asarray([v[0], v[0], v[0]])
@njit
def swizzle_n2_xxy(v):
    return np.asarray([v[0], v[0], v[1]])
@njit
def swizzle_n2_xyx(v):
    return np.asarray([v[0], v[1], v[0]])
@njit
def swizzle_n2_xyy(v):
    return np.asarray([v[0], v[1], v[1]])
@njit
def swizzle_n2_yxx(v):
    return np.asarray([v[1], v[0], v[0]])
@njit
def swizzle_n2_yxy(v):
    return np.asarray([v[1], v[0], v[1]])
@njit
def swizzle_n2_yyx(v):
    return np.asarray([v[1], v[1], v[0]])
@njit
def swizzle_n2_yyy(v):
    return np.asarray([v[1], v[1], v[1]])
@njit
def swizzle_n2_xxxx(v):
    return np.asarray([v[0], v[0], v[0], v[0]])
@njit
def swizzle_n2_xxxy(v):
    return np.asarray([v[0], v[0], v[0], v[1]])
@njit
def swizzle_n2_xxyx(v):
    return np.asarray([v[0], v[0], v[1], v[0]])
@njit
def swizzle_n2_xxyy(v):
    return np.asarray([v[0], v[0], v[1], v[1]])
@njit
def swizzle_n2_xyxx(v):
    return np.asarray([v[0], v[1], v[0], v[0]])
@njit
def swizzle_n2_xyxy(v):
    return np.asarray([v[0], v[1], v[0], v[1]])
@njit
def swizzle_n2_xyyx(v):
    return np.asarray([v[0], v[1], v[1], v[0]])
@njit
def swizzle_n2_xyyy(v):
    return np.asarray([v[0], v[1], v[1], v[1]])
@njit
def swizzle_n2_yxxx(v):
    return np.asarray([v[1], v[0], v[0], v[0]])
@njit
def swizzle_n2_yxxy(v):
    return np.asarray([v[1], v[0], v[0], v[1]])
@njit
def swizzle_n2_yxyx(v):
    return np.asarray([v[1], v[0], v[1], v[0]])
@njit
def swizzle_n2_yxyy(v):
    return np.asarray([v[1], v[0], v[1], v[1]])
@njit
def swizzle_n2_yyxx(v):
    return np.asarray([v[1], v[1], v[0], v[0]])
@njit
def swizzle_n2_yyxy(v):
    return np.asarray([v[1], v[1], v[0], v[1]])
@njit
def swizzle_n2_yyyx(v):
    return np.asarray([v[1], v[1], v[1], v[0]])
@njit
def swizzle_n2_yyyy(v):
    return np.asarray([v[1], v[1], v[1], v[1]])
@njit
def swizzle_n3_x(v):
    return v[0]
@njit
def swizzle_n3_y(v):
    return v[1]
@njit
def swizzle_n3_z(v):
    return v[2]
@njit
def swizzle_n3_xx(v):
    return np.asarray([v[0], v[0]])
@njit
def swizzle_n3_xy(v):
    return np.asarray([v[0], v[1]])
@njit
def swizzle_n3_xz(v):
    return np.asarray([v[0], v[2]])
@njit
def swizzle_n3_yx(v):
    return np.asarray([v[1], v[0]])
@njit
def swizzle_n3_yy(v):
    return np.asarray([v[1], v[1]])
@njit
def swizzle_n3_yz(v):
    return np.asarray([v[1], v[2]])
@njit
def swizzle_n3_zx(v):
    return np.asarray([v[2], v[0]])
@njit
def swizzle_n3_zy(v):
    return np.asarray([v[2], v[1]])
@njit
def swizzle_n3_zz(v):
    return np.asarray([v[2], v[2]])
@njit
def swizzle_n3_xxx(v):
    return np.asarray([v[0], v[0], v[0]])
@njit
def swizzle_n3_xxy(v):
    return np.asarray([v[0], v[0], v[1]])
@njit
def swizzle_n3_xxz(v):
    return np.asarray([v[0], v[0], v[2]])
@njit
def swizzle_n3_xyx(v):
    return np.asarray([v[0], v[1], v[0]])
@njit
def swizzle_n3_xyy(v):
    return np.asarray([v[0], v[1], v[1]])
@njit
def swizzle_n3_xyz(v):
    return np.asarray([v[0], v[1], v[2]])
@njit
def swizzle_n3_xzx(v):
    return np.asarray([v[0], v[2], v[0]])
@njit
def swizzle_n3_xzy(v):
    return np.asarray([v[0], v[2], v[1]])
@njit
def swizzle_n3_xzz(v):
    return np.asarray([v[0], v[2], v[2]])
@njit
def swizzle_n3_yxx(v):
    return np.asarray([v[1], v[0], v[0]])
@njit
def swizzle_n3_yxy(v):
    return np.asarray([v[1], v[0], v[1]])
@njit
def swizzle_n3_yxz(v):
    return np.asarray([v[1], v[0], v[2]])
@njit
def swizzle_n3_yyx(v):
    return np.asarray([v[1], v[1], v[0]])
@njit
def swizzle_n3_yyy(v):
    return np.asarray([v[1], v[1], v[1]])
@njit
def swizzle_n3_yyz(v):
    return np.asarray([v[1], v[1], v[2]])
@njit
def swizzle_n3_yzx(v):
    return np.asarray([v[1], v[2], v[0]])
@njit
def swizzle_n3_yzy(v):
    return np.asarray([v[1], v[2], v[1]])
@njit
def swizzle_n3_yzz(v):
    return np.asarray([v[1], v[2], v[2]])
@njit
def swizzle_n3_zxx(v):
    return np.asarray([v[2], v[0], v[0]])
@njit
def swizzle_n3_zxy(v):
    return np.asarray([v[2], v[0], v[1]])
@njit
def swizzle_n3_zxz(v):
    return np.asarray([v[2], v[0], v[2]])
@njit
def swizzle_n3_zyx(v):
    return np.asarray([v[2], v[1], v[0]])
@njit
def swizzle_n3_zyy(v):
    return np.asarray([v[2], v[1], v[1]])
@njit
def swizzle_n3_zyz(v):
    return np.asarray([v[2], v[1], v[2]])
@njit
def swizzle_n3_zzx(v):
    return np.asarray([v[2], v[2], v[0]])
@njit
def swizzle_n3_zzy(v):
    return np.asarray([v[2], v[2], v[1]])
@njit
def swizzle_n3_zzz(v):
    return np.asarray([v[2], v[2], v[2]])
@njit
def swizzle_n3_xxxx(v):
    return np.asarray([v[0], v[0], v[0], v[0]])
@njit
def swizzle_n3_xxxy(v):
    return np.asarray([v[0], v[0], v[0], v[1]])
@njit
def swizzle_n3_xxxz(v):
    return np.asarray([v[0], v[0], v[0], v[2]])
@njit
def swizzle_n3_xxyx(v):
    return np.asarray([v[0], v[0], v[1], v[0]])
@njit
def swizzle_n3_xxyy(v):
    return np.asarray([v[0], v[0], v[1], v[1]])
@njit
def swizzle_n3_xxyz(v):
    return np.asarray([v[0], v[0], v[1], v[2]])
@njit
def swizzle_n3_xxzx(v):
    return np.asarray([v[0], v[0], v[2], v[0]])
@njit
def swizzle_n3_xxzy(v):
    return np.asarray([v[0], v[0], v[2], v[1]])
@njit
def swizzle_n3_xxzz(v):
    return np.asarray([v[0], v[0], v[2], v[2]])
@njit
def swizzle_n3_xyxx(v):
    return np.asarray([v[0], v[1], v[0], v[0]])
@njit
def swizzle_n3_xyxy(v):
    return np.asarray([v[0], v[1], v[0], v[1]])
@njit
def swizzle_n3_xyxz(v):
    return np.asarray([v[0], v[1], v[0], v[2]])
@njit
def swizzle_n3_xyyx(v):
    return np.asarray([v[0], v[1], v[1], v[0]])
@njit
def swizzle_n3_xyyy(v):
    return np.asarray([v[0], v[1], v[1], v[1]])
@njit
def swizzle_n3_xyyz(v):
    return np.asarray([v[0], v[1], v[1], v[2]])
@njit
def swizzle_n3_xyzx(v):
    return np.asarray([v[0], v[1], v[2], v[0]])
@njit
def swizzle_n3_xyzy(v):
    return np.asarray([v[0], v[1], v[2], v[1]])
@njit
def swizzle_n3_xyzz(v):
    return np.asarray([v[0], v[1], v[2], v[2]])
@njit
def swizzle_n3_xzxx(v):
    return np.asarray([v[0], v[2], v[0], v[0]])
@njit
def swizzle_n3_xzxy(v):
    return np.asarray([v[0], v[2], v[0], v[1]])
@njit
def swizzle_n3_xzxz(v):
    return np.asarray([v[0], v[2], v[0], v[2]])
@njit
def swizzle_n3_xzyx(v):
    return np.asarray([v[0], v[2], v[1], v[0]])
@njit
def swizzle_n3_xzyy(v):
    return np.asarray([v[0], v[2], v[1], v[1]])
@njit
def swizzle_n3_xzyz(v):
    return np.asarray([v[0], v[2], v[1], v[2]])
@njit
def swizzle_n3_xzzx(v):
    return np.asarray([v[0], v[2], v[2], v[0]])
@njit
def swizzle_n3_xzzy(v):
    return np.asarray([v[0], v[2], v[2], v[1]])
@njit
def swizzle_n3_xzzz(v):
    return np.asarray([v[0], v[2], v[2], v[2]])
@njit
def swizzle_n3_yxxx(v):
    return np.asarray([v[1], v[0], v[0], v[0]])
@njit
def swizzle_n3_yxxy(v):
    return np.asarray([v[1], v[0], v[0], v[1]])
@njit
def swizzle_n3_yxxz(v):
    return np.asarray([v[1], v[0], v[0], v[2]])
@njit
def swizzle_n3_yxyx(v):
    return np.asarray([v[1], v[0], v[1], v[0]])
@njit
def swizzle_n3_yxyy(v):
    return np.asarray([v[1], v[0], v[1], v[1]])
@njit
def swizzle_n3_yxyz(v):
    return np.asarray([v[1], v[0], v[1], v[2]])
@njit
def swizzle_n3_yxzx(v):
    return np.asarray([v[1], v[0], v[2], v[0]])
@njit
def swizzle_n3_yxzy(v):
    return np.asarray([v[1], v[0], v[2], v[1]])
@njit
def swizzle_n3_yxzz(v):
    return np.asarray([v[1], v[0], v[2], v[2]])
@njit
def swizzle_n3_yyxx(v):
    return np.asarray([v[1], v[1], v[0], v[0]])
@njit
def swizzle_n3_yyxy(v):
    return np.asarray([v[1], v[1], v[0], v[1]])
@njit
def swizzle_n3_yyxz(v):
    return np.asarray([v[1], v[1], v[0], v[2]])
@njit
def swizzle_n3_yyyx(v):
    return np.asarray([v[1], v[1], v[1], v[0]])
@njit
def swizzle_n3_yyyy(v):
    return np.asarray([v[1], v[1], v[1], v[1]])
@njit
def swizzle_n3_yyyz(v):
    return np.asarray([v[1], v[1], v[1], v[2]])
@njit
def swizzle_n3_yyzx(v):
    return np.asarray([v[1], v[1], v[2], v[0]])
@njit
def swizzle_n3_yyzy(v):
    return np.asarray([v[1], v[1], v[2], v[1]])
@njit
def swizzle_n3_yyzz(v):
    return np.asarray([v[1], v[1], v[2], v[2]])
@njit
def swizzle_n3_yzxx(v):
    return np.asarray([v[1], v[2], v[0], v[0]])
@njit
def swizzle_n3_yzxy(v):
    return np.asarray([v[1], v[2], v[0], v[1]])
@njit
def swizzle_n3_yzxz(v):
    return np.asarray([v[1], v[2], v[0], v[2]])
@njit
def swizzle_n3_yzyx(v):
    return np.asarray([v[1], v[2], v[1], v[0]])
@njit
def swizzle_n3_yzyy(v):
    return np.asarray([v[1], v[2], v[1], v[1]])
@njit
def swizzle_n3_yzyz(v):
    return np.asarray([v[1], v[2], v[1], v[2]])
@njit
def swizzle_n3_yzzx(v):
    return np.asarray([v[1], v[2], v[2], v[0]])
@njit
def swizzle_n3_yzzy(v):
    return np.asarray([v[1], v[2], v[2], v[1]])
@njit
def swizzle_n3_yzzz(v):
    return np.asarray([v[1], v[2], v[2], v[2]])
@njit
def swizzle_n3_zxxx(v):
    return np.asarray([v[2], v[0], v[0], v[0]])
@njit
def swizzle_n3_zxxy(v):
    return np.asarray([v[2], v[0], v[0], v[1]])
@njit
def swizzle_n3_zxxz(v):
    return np.asarray([v[2], v[0], v[0], v[2]])
@njit
def swizzle_n3_zxyx(v):
    return np.asarray([v[2], v[0], v[1], v[0]])
@njit
def swizzle_n3_zxyy(v):
    return np.asarray([v[2], v[0], v[1], v[1]])
@njit
def swizzle_n3_zxyz(v):
    return np.asarray([v[2], v[0], v[1], v[2]])
@njit
def swizzle_n3_zxzx(v):
    return np.asarray([v[2], v[0], v[2], v[0]])
@njit
def swizzle_n3_zxzy(v):
    return np.asarray([v[2], v[0], v[2], v[1]])
@njit
def swizzle_n3_zxzz(v):
    return np.asarray([v[2], v[0], v[2], v[2]])
@njit
def swizzle_n3_zyxx(v):
    return np.asarray([v[2], v[1], v[0], v[0]])
@njit
def swizzle_n3_zyxy(v):
    return np.asarray([v[2], v[1], v[0], v[1]])
@njit
def swizzle_n3_zyxz(v):
    return np.asarray([v[2], v[1], v[0], v[2]])
@njit
def swizzle_n3_zyyx(v):
    return np.asarray([v[2], v[1], v[1], v[0]])
@njit
def swizzle_n3_zyyy(v):
    return np.asarray([v[2], v[1], v[1], v[1]])
@njit
def swizzle_n3_zyyz(v):
    return np.asarray([v[2], v[1], v[1], v[2]])
@njit
def swizzle_n3_zyzx(v):
    return np.asarray([v[2], v[1], v[2], v[0]])
@njit
def swizzle_n3_zyzy(v):
    return np.asarray([v[2], v[1], v[2], v[1]])
@njit
def swizzle_n3_zyzz(v):
    return np.asarray([v[2], v[1], v[2], v[2]])
@njit
def swizzle_n3_zzxx(v):
    return np.asarray([v[2], v[2], v[0], v[0]])
@njit
def swizzle_n3_zzxy(v):
    return np.asarray([v[2], v[2], v[0], v[1]])
@njit
def swizzle_n3_zzxz(v):
    return np.asarray([v[2], v[2], v[0], v[2]])
@njit
def swizzle_n3_zzyx(v):
    return np.asarray([v[2], v[2], v[1], v[0]])
@njit
def swizzle_n3_zzyy(v):
    return np.asarray([v[2], v[2], v[1], v[1]])
@njit
def swizzle_n3_zzyz(v):
    return np.asarray([v[2], v[2], v[1], v[2]])
@njit
def swizzle_n3_zzzx(v):
    return np.asarray([v[2], v[2], v[2], v[0]])
@njit
def swizzle_n3_zzzy(v):
    return np.asarray([v[2], v[2], v[2], v[1]])
@njit
def swizzle_n3_zzzz(v):
    return np.asarray([v[2], v[2], v[2], v[2]])
@njit
def swizzle_n4_x(v):
    return v[0]
@njit
def swizzle_n4_y(v):
    return v[1]
@njit
def swizzle_n4_z(v):
    return v[2]
@njit
def swizzle_n4_w(v):
    return v[3]
@njit
def swizzle_n4_xx(v):
    return np.asarray([v[0], v[0]])
@njit
def swizzle_n4_xy(v):
    return np.asarray([v[0], v[1]])
@njit
def swizzle_n4_xz(v):
    return np.asarray([v[0], v[2]])
@njit
def swizzle_n4_xw(v):
    return np.asarray([v[0], v[3]])
@njit
def swizzle_n4_yx(v):
    return np.asarray([v[1], v[0]])
@njit
def swizzle_n4_yy(v):
    return np.asarray([v[1], v[1]])
@njit
def swizzle_n4_yz(v):
    return np.asarray([v[1], v[2]])
@njit
def swizzle_n4_yw(v):
    return np.asarray([v[1], v[3]])
@njit
def swizzle_n4_zx(v):
    return np.asarray([v[2], v[0]])
@njit
def swizzle_n4_zy(v):
    return np.asarray([v[2], v[1]])
@njit
def swizzle_n4_zz(v):
    return np.asarray([v[2], v[2]])
@njit
def swizzle_n4_zw(v):
    return np.asarray([v[2], v[3]])
@njit
def swizzle_n4_wx(v):
    return np.asarray([v[3], v[0]])
@njit
def swizzle_n4_wy(v):
    return np.asarray([v[3], v[1]])
@njit
def swizzle_n4_wz(v):
    return np.asarray([v[3], v[2]])
@njit
def swizzle_n4_ww(v):
    return np.asarray([v[3], v[3]])
@njit
def swizzle_n4_xxx(v):
    return np.asarray([v[0], v[0], v[0]])
@njit
def swizzle_n4_xxy(v):
    return np.asarray([v[0], v[0], v[1]])
@njit
def swizzle_n4_xxz(v):
    return np.asarray([v[0], v[0], v[2]])
@njit
def swizzle_n4_xxw(v):
    return np.asarray([v[0], v[0], v[3]])
@njit
def swizzle_n4_xyx(v):
    return np.asarray([v[0], v[1], v[0]])
@njit
def swizzle_n4_xyy(v):
    return np.asarray([v[0], v[1], v[1]])
@njit
def swizzle_n4_xyz(v):
    return np.asarray([v[0], v[1], v[2]])
@njit
def swizzle_n4_xyw(v):
    return np.asarray([v[0], v[1], v[3]])
@njit
def swizzle_n4_xzx(v):
    return np.asarray([v[0], v[2], v[0]])
@njit
def swizzle_n4_xzy(v):
    return np.asarray([v[0], v[2], v[1]])
@njit
def swizzle_n4_xzz(v):
    return np.asarray([v[0], v[2], v[2]])
@njit
def swizzle_n4_xzw(v):
    return np.asarray([v[0], v[2], v[3]])
@njit
def swizzle_n4_xwx(v):
    return np.asarray([v[0], v[3], v[0]])
@njit
def swizzle_n4_xwy(v):
    return np.asarray([v[0], v[3], v[1]])
@njit
def swizzle_n4_xwz(v):
    return np.asarray([v[0], v[3], v[2]])
@njit
def swizzle_n4_xww(v):
    return np.asarray([v[0], v[3], v[3]])
@njit
def swizzle_n4_yxx(v):
    return np.asarray([v[1], v[0], v[0]])
@njit
def swizzle_n4_yxy(v):
    return np.asarray([v[1], v[0], v[1]])
@njit
def swizzle_n4_yxz(v):
    return np.asarray([v[1], v[0], v[2]])
@njit
def swizzle_n4_yxw(v):
    return np.asarray([v[1], v[0], v[3]])
@njit
def swizzle_n4_yyx(v):
    return np.asarray([v[1], v[1], v[0]])
@njit
def swizzle_n4_yyy(v):
    return np.asarray([v[1], v[1], v[1]])
@njit
def swizzle_n4_yyz(v):
    return np.asarray([v[1], v[1], v[2]])
@njit
def swizzle_n4_yyw(v):
    return np.asarray([v[1], v[1], v[3]])
@njit
def swizzle_n4_yzx(v):
    return np.asarray([v[1], v[2], v[0]])
@njit
def swizzle_n4_yzy(v):
    return np.asarray([v[1], v[2], v[1]])
@njit
def swizzle_n4_yzz(v):
    return np.asarray([v[1], v[2], v[2]])
@njit
def swizzle_n4_yzw(v):
    return np.asarray([v[1], v[2], v[3]])
@njit
def swizzle_n4_ywx(v):
    return np.asarray([v[1], v[3], v[0]])
@njit
def swizzle_n4_ywy(v):
    return np.asarray([v[1], v[3], v[1]])
@njit
def swizzle_n4_ywz(v):
    return np.asarray([v[1], v[3], v[2]])
@njit
def swizzle_n4_yww(v):
    return np.asarray([v[1], v[3], v[3]])
@njit
def swizzle_n4_zxx(v):
    return np.asarray([v[2], v[0], v[0]])
@njit
def swizzle_n4_zxy(v):
    return np.asarray([v[2], v[0], v[1]])
@njit
def swizzle_n4_zxz(v):
    return np.asarray([v[2], v[0], v[2]])
@njit
def swizzle_n4_zxw(v):
    return np.asarray([v[2], v[0], v[3]])
@njit
def swizzle_n4_zyx(v):
    return np.asarray([v[2], v[1], v[0]])
@njit
def swizzle_n4_zyy(v):
    return np.asarray([v[2], v[1], v[1]])
@njit
def swizzle_n4_zyz(v):
    return np.asarray([v[2], v[1], v[2]])
@njit
def swizzle_n4_zyw(v):
    return np.asarray([v[2], v[1], v[3]])
@njit
def swizzle_n4_zzx(v):
    return np.asarray([v[2], v[2], v[0]])
@njit
def swizzle_n4_zzy(v):
    return np.asarray([v[2], v[2], v[1]])
@njit
def swizzle_n4_zzz(v):
    return np.asarray([v[2], v[2], v[2]])
@njit
def swizzle_n4_zzw(v):
    return np.asarray([v[2], v[2], v[3]])
@njit
def swizzle_n4_zwx(v):
    return np.asarray([v[2], v[3], v[0]])
@njit
def swizzle_n4_zwy(v):
    return np.asarray([v[2], v[3], v[1]])
@njit
def swizzle_n4_zwz(v):
    return np.asarray([v[2], v[3], v[2]])
@njit
def swizzle_n4_zww(v):
    return np.asarray([v[2], v[3], v[3]])
@njit
def swizzle_n4_wxx(v):
    return np.asarray([v[3], v[0], v[0]])
@njit
def swizzle_n4_wxy(v):
    return np.asarray([v[3], v[0], v[1]])
@njit
def swizzle_n4_wxz(v):
    return np.asarray([v[3], v[0], v[2]])
@njit
def swizzle_n4_wxw(v):
    return np.asarray([v[3], v[0], v[3]])
@njit
def swizzle_n4_wyx(v):
    return np.asarray([v[3], v[1], v[0]])
@njit
def swizzle_n4_wyy(v):
    return np.asarray([v[3], v[1], v[1]])
@njit
def swizzle_n4_wyz(v):
    return np.asarray([v[3], v[1], v[2]])
@njit
def swizzle_n4_wyw(v):
    return np.asarray([v[3], v[1], v[3]])
@njit
def swizzle_n4_wzx(v):
    return np.asarray([v[3], v[2], v[0]])
@njit
def swizzle_n4_wzy(v):
    return np.asarray([v[3], v[2], v[1]])
@njit
def swizzle_n4_wzz(v):
    return np.asarray([v[3], v[2], v[2]])
@njit
def swizzle_n4_wzw(v):
    return np.asarray([v[3], v[2], v[3]])
@njit
def swizzle_n4_wwx(v):
    return np.asarray([v[3], v[3], v[0]])
@njit
def swizzle_n4_wwy(v):
    return np.asarray([v[3], v[3], v[1]])
@njit
def swizzle_n4_wwz(v):
    return np.asarray([v[3], v[3], v[2]])
@njit
def swizzle_n4_www(v):
    return np.asarray([v[3], v[3], v[3]])
@njit
def swizzle_n4_xxxx(v):
    return np.asarray([v[0], v[0], v[0], v[0]])
@njit
def swizzle_n4_xxxy(v):
    return np.asarray([v[0], v[0], v[0], v[1]])
@njit
def swizzle_n4_xxxz(v):
    return np.asarray([v[0], v[0], v[0], v[2]])
@njit
def swizzle_n4_xxxw(v):
    return np.asarray([v[0], v[0], v[0], v[3]])
@njit
def swizzle_n4_xxyx(v):
    return np.asarray([v[0], v[0], v[1], v[0]])
@njit
def swizzle_n4_xxyy(v):
    return np.asarray([v[0], v[0], v[1], v[1]])
@njit
def swizzle_n4_xxyz(v):
    return np.asarray([v[0], v[0], v[1], v[2]])
@njit
def swizzle_n4_xxyw(v):
    return np.asarray([v[0], v[0], v[1], v[3]])
@njit
def swizzle_n4_xxzx(v):
    return np.asarray([v[0], v[0], v[2], v[0]])
@njit
def swizzle_n4_xxzy(v):
    return np.asarray([v[0], v[0], v[2], v[1]])
@njit
def swizzle_n4_xxzz(v):
    return np.asarray([v[0], v[0], v[2], v[2]])
@njit
def swizzle_n4_xxzw(v):
    return np.asarray([v[0], v[0], v[2], v[3]])
@njit
def swizzle_n4_xxwx(v):
    return np.asarray([v[0], v[0], v[3], v[0]])
@njit
def swizzle_n4_xxwy(v):
    return np.asarray([v[0], v[0], v[3], v[1]])
@njit
def swizzle_n4_xxwz(v):
    return np.asarray([v[0], v[0], v[3], v[2]])
@njit
def swizzle_n4_xxww(v):
    return np.asarray([v[0], v[0], v[3], v[3]])
@njit
def swizzle_n4_xyxx(v):
    return np.asarray([v[0], v[1], v[0], v[0]])
@njit
def swizzle_n4_xyxy(v):
    return np.asarray([v[0], v[1], v[0], v[1]])
@njit
def swizzle_n4_xyxz(v):
    return np.asarray([v[0], v[1], v[0], v[2]])
@njit
def swizzle_n4_xyxw(v):
    return np.asarray([v[0], v[1], v[0], v[3]])
@njit
def swizzle_n4_xyyx(v):
    return np.asarray([v[0], v[1], v[1], v[0]])
@njit
def swizzle_n4_xyyy(v):
    return np.asarray([v[0], v[1], v[1], v[1]])
@njit
def swizzle_n4_xyyz(v):
    return np.asarray([v[0], v[1], v[1], v[2]])
@njit
def swizzle_n4_xyyw(v):
    return np.asarray([v[0], v[1], v[1], v[3]])
@njit
def swizzle_n4_xyzx(v):
    return np.asarray([v[0], v[1], v[2], v[0]])
@njit
def swizzle_n4_xyzy(v):
    return np.asarray([v[0], v[1], v[2], v[1]])
@njit
def swizzle_n4_xyzz(v):
    return np.asarray([v[0], v[1], v[2], v[2]])
@njit
def swizzle_n4_xyzw(v):
    return np.asarray([v[0], v[1], v[2], v[3]])
@njit
def swizzle_n4_xywx(v):
    return np.asarray([v[0], v[1], v[3], v[0]])
@njit
def swizzle_n4_xywy(v):
    return np.asarray([v[0], v[1], v[3], v[1]])
@njit
def swizzle_n4_xywz(v):
    return np.asarray([v[0], v[1], v[3], v[2]])
@njit
def swizzle_n4_xyww(v):
    return np.asarray([v[0], v[1], v[3], v[3]])
@njit
def swizzle_n4_xzxx(v):
    return np.asarray([v[0], v[2], v[0], v[0]])
@njit
def swizzle_n4_xzxy(v):
    return np.asarray([v[0], v[2], v[0], v[1]])
@njit
def swizzle_n4_xzxz(v):
    return np.asarray([v[0], v[2], v[0], v[2]])
@njit
def swizzle_n4_xzxw(v):
    return np.asarray([v[0], v[2], v[0], v[3]])
@njit
def swizzle_n4_xzyx(v):
    return np.asarray([v[0], v[2], v[1], v[0]])
@njit
def swizzle_n4_xzyy(v):
    return np.asarray([v[0], v[2], v[1], v[1]])
@njit
def swizzle_n4_xzyz(v):
    return np.asarray([v[0], v[2], v[1], v[2]])
@njit
def swizzle_n4_xzyw(v):
    return np.asarray([v[0], v[2], v[1], v[3]])
@njit
def swizzle_n4_xzzx(v):
    return np.asarray([v[0], v[2], v[2], v[0]])
@njit
def swizzle_n4_xzzy(v):
    return np.asarray([v[0], v[2], v[2], v[1]])
@njit
def swizzle_n4_xzzz(v):
    return np.asarray([v[0], v[2], v[2], v[2]])
@njit
def swizzle_n4_xzzw(v):
    return np.asarray([v[0], v[2], v[2], v[3]])
@njit
def swizzle_n4_xzwx(v):
    return np.asarray([v[0], v[2], v[3], v[0]])
@njit
def swizzle_n4_xzwy(v):
    return np.asarray([v[0], v[2], v[3], v[1]])
@njit
def swizzle_n4_xzwz(v):
    return np.asarray([v[0], v[2], v[3], v[2]])
@njit
def swizzle_n4_xzww(v):
    return np.asarray([v[0], v[2], v[3], v[3]])
@njit
def swizzle_n4_xwxx(v):
    return np.asarray([v[0], v[3], v[0], v[0]])
@njit
def swizzle_n4_xwxy(v):
    return np.asarray([v[0], v[3], v[0], v[1]])
@njit
def swizzle_n4_xwxz(v):
    return np.asarray([v[0], v[3], v[0], v[2]])
@njit
def swizzle_n4_xwxw(v):
    return np.asarray([v[0], v[3], v[0], v[3]])
@njit
def swizzle_n4_xwyx(v):
    return np.asarray([v[0], v[3], v[1], v[0]])
@njit
def swizzle_n4_xwyy(v):
    return np.asarray([v[0], v[3], v[1], v[1]])
@njit
def swizzle_n4_xwyz(v):
    return np.asarray([v[0], v[3], v[1], v[2]])
@njit
def swizzle_n4_xwyw(v):
    return np.asarray([v[0], v[3], v[1], v[3]])
@njit
def swizzle_n4_xwzx(v):
    return np.asarray([v[0], v[3], v[2], v[0]])
@njit
def swizzle_n4_xwzy(v):
    return np.asarray([v[0], v[3], v[2], v[1]])
@njit
def swizzle_n4_xwzz(v):
    return np.asarray([v[0], v[3], v[2], v[2]])
@njit
def swizzle_n4_xwzw(v):
    return np.asarray([v[0], v[3], v[2], v[3]])
@njit
def swizzle_n4_xwwx(v):
    return np.asarray([v[0], v[3], v[3], v[0]])
@njit
def swizzle_n4_xwwy(v):
    return np.asarray([v[0], v[3], v[3], v[1]])
@njit
def swizzle_n4_xwwz(v):
    return np.asarray([v[0], v[3], v[3], v[2]])
@njit
def swizzle_n4_xwww(v):
    return np.asarray([v[0], v[3], v[3], v[3]])
@njit
def swizzle_n4_yxxx(v):
    return np.asarray([v[1], v[0], v[0], v[0]])
@njit
def swizzle_n4_yxxy(v):
    return np.asarray([v[1], v[0], v[0], v[1]])
@njit
def swizzle_n4_yxxz(v):
    return np.asarray([v[1], v[0], v[0], v[2]])
@njit
def swizzle_n4_yxxw(v):
    return np.asarray([v[1], v[0], v[0], v[3]])
@njit
def swizzle_n4_yxyx(v):
    return np.asarray([v[1], v[0], v[1], v[0]])
@njit
def swizzle_n4_yxyy(v):
    return np.asarray([v[1], v[0], v[1], v[1]])
@njit
def swizzle_n4_yxyz(v):
    return np.asarray([v[1], v[0], v[1], v[2]])
@njit
def swizzle_n4_yxyw(v):
    return np.asarray([v[1], v[0], v[1], v[3]])
@njit
def swizzle_n4_yxzx(v):
    return np.asarray([v[1], v[0], v[2], v[0]])
@njit
def swizzle_n4_yxzy(v):
    return np.asarray([v[1], v[0], v[2], v[1]])
@njit
def swizzle_n4_yxzz(v):
    return np.asarray([v[1], v[0], v[2], v[2]])
@njit
def swizzle_n4_yxzw(v):
    return np.asarray([v[1], v[0], v[2], v[3]])
@njit
def swizzle_n4_yxwx(v):
    return np.asarray([v[1], v[0], v[3], v[0]])
@njit
def swizzle_n4_yxwy(v):
    return np.asarray([v[1], v[0], v[3], v[1]])
@njit
def swizzle_n4_yxwz(v):
    return np.asarray([v[1], v[0], v[3], v[2]])
@njit
def swizzle_n4_yxww(v):
    return np.asarray([v[1], v[0], v[3], v[3]])
@njit
def swizzle_n4_yyxx(v):
    return np.asarray([v[1], v[1], v[0], v[0]])
@njit
def swizzle_n4_yyxy(v):
    return np.asarray([v[1], v[1], v[0], v[1]])
@njit
def swizzle_n4_yyxz(v):
    return np.asarray([v[1], v[1], v[0], v[2]])
@njit
def swizzle_n4_yyxw(v):
    return np.asarray([v[1], v[1], v[0], v[3]])
@njit
def swizzle_n4_yyyx(v):
    return np.asarray([v[1], v[1], v[1], v[0]])
@njit
def swizzle_n4_yyyy(v):
    return np.asarray([v[1], v[1], v[1], v[1]])
@njit
def swizzle_n4_yyyz(v):
    return np.asarray([v[1], v[1], v[1], v[2]])
@njit
def swizzle_n4_yyyw(v):
    return np.asarray([v[1], v[1], v[1], v[3]])
@njit
def swizzle_n4_yyzx(v):
    return np.asarray([v[1], v[1], v[2], v[0]])
@njit
def swizzle_n4_yyzy(v):
    return np.asarray([v[1], v[1], v[2], v[1]])
@njit
def swizzle_n4_yyzz(v):
    return np.asarray([v[1], v[1], v[2], v[2]])
@njit
def swizzle_n4_yyzw(v):
    return np.asarray([v[1], v[1], v[2], v[3]])
@njit
def swizzle_n4_yywx(v):
    return np.asarray([v[1], v[1], v[3], v[0]])
@njit
def swizzle_n4_yywy(v):
    return np.asarray([v[1], v[1], v[3], v[1]])
@njit
def swizzle_n4_yywz(v):
    return np.asarray([v[1], v[1], v[3], v[2]])
@njit
def swizzle_n4_yyww(v):
    return np.asarray([v[1], v[1], v[3], v[3]])
@njit
def swizzle_n4_yzxx(v):
    return np.asarray([v[1], v[2], v[0], v[0]])
@njit
def swizzle_n4_yzxy(v):
    return np.asarray([v[1], v[2], v[0], v[1]])
@njit
def swizzle_n4_yzxz(v):
    return np.asarray([v[1], v[2], v[0], v[2]])
@njit
def swizzle_n4_yzxw(v):
    return np.asarray([v[1], v[2], v[0], v[3]])
@njit
def swizzle_n4_yzyx(v):
    return np.asarray([v[1], v[2], v[1], v[0]])
@njit
def swizzle_n4_yzyy(v):
    return np.asarray([v[1], v[2], v[1], v[1]])
@njit
def swizzle_n4_yzyz(v):
    return np.asarray([v[1], v[2], v[1], v[2]])
@njit
def swizzle_n4_yzyw(v):
    return np.asarray([v[1], v[2], v[1], v[3]])
@njit
def swizzle_n4_yzzx(v):
    return np.asarray([v[1], v[2], v[2], v[0]])
@njit
def swizzle_n4_yzzy(v):
    return np.asarray([v[1], v[2], v[2], v[1]])
@njit
def swizzle_n4_yzzz(v):
    return np.asarray([v[1], v[2], v[2], v[2]])
@njit
def swizzle_n4_yzzw(v):
    return np.asarray([v[1], v[2], v[2], v[3]])
@njit
def swizzle_n4_yzwx(v):
    return np.asarray([v[1], v[2], v[3], v[0]])
@njit
def swizzle_n4_yzwy(v):
    return np.asarray([v[1], v[2], v[3], v[1]])
@njit
def swizzle_n4_yzwz(v):
    return np.asarray([v[1], v[2], v[3], v[2]])
@njit
def swizzle_n4_yzww(v):
    return np.asarray([v[1], v[2], v[3], v[3]])
@njit
def swizzle_n4_ywxx(v):
    return np.asarray([v[1], v[3], v[0], v[0]])
@njit
def swizzle_n4_ywxy(v):
    return np.asarray([v[1], v[3], v[0], v[1]])
@njit
def swizzle_n4_ywxz(v):
    return np.asarray([v[1], v[3], v[0], v[2]])
@njit
def swizzle_n4_ywxw(v):
    return np.asarray([v[1], v[3], v[0], v[3]])
@njit
def swizzle_n4_ywyx(v):
    return np.asarray([v[1], v[3], v[1], v[0]])
@njit
def swizzle_n4_ywyy(v):
    return np.asarray([v[1], v[3], v[1], v[1]])
@njit
def swizzle_n4_ywyz(v):
    return np.asarray([v[1], v[3], v[1], v[2]])
@njit
def swizzle_n4_ywyw(v):
    return np.asarray([v[1], v[3], v[1], v[3]])
@njit
def swizzle_n4_ywzx(v):
    return np.asarray([v[1], v[3], v[2], v[0]])
@njit
def swizzle_n4_ywzy(v):
    return np.asarray([v[1], v[3], v[2], v[1]])
@njit
def swizzle_n4_ywzz(v):
    return np.asarray([v[1], v[3], v[2], v[2]])
@njit
def swizzle_n4_ywzw(v):
    return np.asarray([v[1], v[3], v[2], v[3]])
@njit
def swizzle_n4_ywwx(v):
    return np.asarray([v[1], v[3], v[3], v[0]])
@njit
def swizzle_n4_ywwy(v):
    return np.asarray([v[1], v[3], v[3], v[1]])
@njit
def swizzle_n4_ywwz(v):
    return np.asarray([v[1], v[3], v[3], v[2]])
@njit
def swizzle_n4_ywww(v):
    return np.asarray([v[1], v[3], v[3], v[3]])
@njit
def swizzle_n4_zxxx(v):
    return np.asarray([v[2], v[0], v[0], v[0]])
@njit
def swizzle_n4_zxxy(v):
    return np.asarray([v[2], v[0], v[0], v[1]])
@njit
def swizzle_n4_zxxz(v):
    return np.asarray([v[2], v[0], v[0], v[2]])
@njit
def swizzle_n4_zxxw(v):
    return np.asarray([v[2], v[0], v[0], v[3]])
@njit
def swizzle_n4_zxyx(v):
    return np.asarray([v[2], v[0], v[1], v[0]])
@njit
def swizzle_n4_zxyy(v):
    return np.asarray([v[2], v[0], v[1], v[1]])
@njit
def swizzle_n4_zxyz(v):
    return np.asarray([v[2], v[0], v[1], v[2]])
@njit
def swizzle_n4_zxyw(v):
    return np.asarray([v[2], v[0], v[1], v[3]])
@njit
def swizzle_n4_zxzx(v):
    return np.asarray([v[2], v[0], v[2], v[0]])
@njit
def swizzle_n4_zxzy(v):
    return np.asarray([v[2], v[0], v[2], v[1]])
@njit
def swizzle_n4_zxzz(v):
    return np.asarray([v[2], v[0], v[2], v[2]])
@njit
def swizzle_n4_zxzw(v):
    return np.asarray([v[2], v[0], v[2], v[3]])
@njit
def swizzle_n4_zxwx(v):
    return np.asarray([v[2], v[0], v[3], v[0]])
@njit
def swizzle_n4_zxwy(v):
    return np.asarray([v[2], v[0], v[3], v[1]])
@njit
def swizzle_n4_zxwz(v):
    return np.asarray([v[2], v[0], v[3], v[2]])
@njit
def swizzle_n4_zxww(v):
    return np.asarray([v[2], v[0], v[3], v[3]])
@njit
def swizzle_n4_zyxx(v):
    return np.asarray([v[2], v[1], v[0], v[0]])
@njit
def swizzle_n4_zyxy(v):
    return np.asarray([v[2], v[1], v[0], v[1]])
@njit
def swizzle_n4_zyxz(v):
    return np.asarray([v[2], v[1], v[0], v[2]])
@njit
def swizzle_n4_zyxw(v):
    return np.asarray([v[2], v[1], v[0], v[3]])
@njit
def swizzle_n4_zyyx(v):
    return np.asarray([v[2], v[1], v[1], v[0]])
@njit
def swizzle_n4_zyyy(v):
    return np.asarray([v[2], v[1], v[1], v[1]])
@njit
def swizzle_n4_zyyz(v):
    return np.asarray([v[2], v[1], v[1], v[2]])
@njit
def swizzle_n4_zyyw(v):
    return np.asarray([v[2], v[1], v[1], v[3]])
@njit
def swizzle_n4_zyzx(v):
    return np.asarray([v[2], v[1], v[2], v[0]])
@njit
def swizzle_n4_zyzy(v):
    return np.asarray([v[2], v[1], v[2], v[1]])
@njit
def swizzle_n4_zyzz(v):
    return np.asarray([v[2], v[1], v[2], v[2]])
@njit
def swizzle_n4_zyzw(v):
    return np.asarray([v[2], v[1], v[2], v[3]])
@njit
def swizzle_n4_zywx(v):
    return np.asarray([v[2], v[1], v[3], v[0]])
@njit
def swizzle_n4_zywy(v):
    return np.asarray([v[2], v[1], v[3], v[1]])
@njit
def swizzle_n4_zywz(v):
    return np.asarray([v[2], v[1], v[3], v[2]])
@njit
def swizzle_n4_zyww(v):
    return np.asarray([v[2], v[1], v[3], v[3]])
@njit
def swizzle_n4_zzxx(v):
    return np.asarray([v[2], v[2], v[0], v[0]])
@njit
def swizzle_n4_zzxy(v):
    return np.asarray([v[2], v[2], v[0], v[1]])
@njit
def swizzle_n4_zzxz(v):
    return np.asarray([v[2], v[2], v[0], v[2]])
@njit
def swizzle_n4_zzxw(v):
    return np.asarray([v[2], v[2], v[0], v[3]])
@njit
def swizzle_n4_zzyx(v):
    return np.asarray([v[2], v[2], v[1], v[0]])
@njit
def swizzle_n4_zzyy(v):
    return np.asarray([v[2], v[2], v[1], v[1]])
@njit
def swizzle_n4_zzyz(v):
    return np.asarray([v[2], v[2], v[1], v[2]])
@njit
def swizzle_n4_zzyw(v):
    return np.asarray([v[2], v[2], v[1], v[3]])
@njit
def swizzle_n4_zzzx(v):
    return np.asarray([v[2], v[2], v[2], v[0]])
@njit
def swizzle_n4_zzzy(v):
    return np.asarray([v[2], v[2], v[2], v[1]])
@njit
def swizzle_n4_zzzz(v):
    return np.asarray([v[2], v[2], v[2], v[2]])
@njit
def swizzle_n4_zzzw(v):
    return np.asarray([v[2], v[2], v[2], v[3]])
@njit
def swizzle_n4_zzwx(v):
    return np.asarray([v[2], v[2], v[3], v[0]])
@njit
def swizzle_n4_zzwy(v):
    return np.asarray([v[2], v[2], v[3], v[1]])
@njit
def swizzle_n4_zzwz(v):
    return np.asarray([v[2], v[2], v[3], v[2]])
@njit
def swizzle_n4_zzww(v):
    return np.asarray([v[2], v[2], v[3], v[3]])
@njit
def swizzle_n4_zwxx(v):
    return np.asarray([v[2], v[3], v[0], v[0]])
@njit
def swizzle_n4_zwxy(v):
    return np.asarray([v[2], v[3], v[0], v[1]])
@njit
def swizzle_n4_zwxz(v):
    return np.asarray([v[2], v[3], v[0], v[2]])
@njit
def swizzle_n4_zwxw(v):
    return np.asarray([v[2], v[3], v[0], v[3]])
@njit
def swizzle_n4_zwyx(v):
    return np.asarray([v[2], v[3], v[1], v[0]])
@njit
def swizzle_n4_zwyy(v):
    return np.asarray([v[2], v[3], v[1], v[1]])
@njit
def swizzle_n4_zwyz(v):
    return np.asarray([v[2], v[3], v[1], v[2]])
@njit
def swizzle_n4_zwyw(v):
    return np.asarray([v[2], v[3], v[1], v[3]])
@njit
def swizzle_n4_zwzx(v):
    return np.asarray([v[2], v[3], v[2], v[0]])
@njit
def swizzle_n4_zwzy(v):
    return np.asarray([v[2], v[3], v[2], v[1]])
@njit
def swizzle_n4_zwzz(v):
    return np.asarray([v[2], v[3], v[2], v[2]])
@njit
def swizzle_n4_zwzw(v):
    return np.asarray([v[2], v[3], v[2], v[3]])
@njit
def swizzle_n4_zwwx(v):
    return np.asarray([v[2], v[3], v[3], v[0]])
@njit
def swizzle_n4_zwwy(v):
    return np.asarray([v[2], v[3], v[3], v[1]])
@njit
def swizzle_n4_zwwz(v):
    return np.asarray([v[2], v[3], v[3], v[2]])
@njit
def swizzle_n4_zwww(v):
    return np.asarray([v[2], v[3], v[3], v[3]])
@njit
def swizzle_n4_wxxx(v):
    return np.asarray([v[3], v[0], v[0], v[0]])
@njit
def swizzle_n4_wxxy(v):
    return np.asarray([v[3], v[0], v[0], v[1]])
@njit
def swizzle_n4_wxxz(v):
    return np.asarray([v[3], v[0], v[0], v[2]])
@njit
def swizzle_n4_wxxw(v):
    return np.asarray([v[3], v[0], v[0], v[3]])
@njit
def swizzle_n4_wxyx(v):
    return np.asarray([v[3], v[0], v[1], v[0]])
@njit
def swizzle_n4_wxyy(v):
    return np.asarray([v[3], v[0], v[1], v[1]])
@njit
def swizzle_n4_wxyz(v):
    return np.asarray([v[3], v[0], v[1], v[2]])
@njit
def swizzle_n4_wxyw(v):
    return np.asarray([v[3], v[0], v[1], v[3]])
@njit
def swizzle_n4_wxzx(v):
    return np.asarray([v[3], v[0], v[2], v[0]])
@njit
def swizzle_n4_wxzy(v):
    return np.asarray([v[3], v[0], v[2], v[1]])
@njit
def swizzle_n4_wxzz(v):
    return np.asarray([v[3], v[0], v[2], v[2]])
@njit
def swizzle_n4_wxzw(v):
    return np.asarray([v[3], v[0], v[2], v[3]])
@njit
def swizzle_n4_wxwx(v):
    return np.asarray([v[3], v[0], v[3], v[0]])
@njit
def swizzle_n4_wxwy(v):
    return np.asarray([v[3], v[0], v[3], v[1]])
@njit
def swizzle_n4_wxwz(v):
    return np.asarray([v[3], v[0], v[3], v[2]])
@njit
def swizzle_n4_wxww(v):
    return np.asarray([v[3], v[0], v[3], v[3]])
@njit
def swizzle_n4_wyxx(v):
    return np.asarray([v[3], v[1], v[0], v[0]])
@njit
def swizzle_n4_wyxy(v):
    return np.asarray([v[3], v[1], v[0], v[1]])
@njit
def swizzle_n4_wyxz(v):
    return np.asarray([v[3], v[1], v[0], v[2]])
@njit
def swizzle_n4_wyxw(v):
    return np.asarray([v[3], v[1], v[0], v[3]])
@njit
def swizzle_n4_wyyx(v):
    return np.asarray([v[3], v[1], v[1], v[0]])
@njit
def swizzle_n4_wyyy(v):
    return np.asarray([v[3], v[1], v[1], v[1]])
@njit
def swizzle_n4_wyyz(v):
    return np.asarray([v[3], v[1], v[1], v[2]])
@njit
def swizzle_n4_wyyw(v):
    return np.asarray([v[3], v[1], v[1], v[3]])
@njit
def swizzle_n4_wyzx(v):
    return np.asarray([v[3], v[1], v[2], v[0]])
@njit
def swizzle_n4_wyzy(v):
    return np.asarray([v[3], v[1], v[2], v[1]])
@njit
def swizzle_n4_wyzz(v):
    return np.asarray([v[3], v[1], v[2], v[2]])
@njit
def swizzle_n4_wyzw(v):
    return np.asarray([v[3], v[1], v[2], v[3]])
@njit
def swizzle_n4_wywx(v):
    return np.asarray([v[3], v[1], v[3], v[0]])
@njit
def swizzle_n4_wywy(v):
    return np.asarray([v[3], v[1], v[3], v[1]])
@njit
def swizzle_n4_wywz(v):
    return np.asarray([v[3], v[1], v[3], v[2]])
@njit
def swizzle_n4_wyww(v):
    return np.asarray([v[3], v[1], v[3], v[3]])
@njit
def swizzle_n4_wzxx(v):
    return np.asarray([v[3], v[2], v[0], v[0]])
@njit
def swizzle_n4_wzxy(v):
    return np.asarray([v[3], v[2], v[0], v[1]])
@njit
def swizzle_n4_wzxz(v):
    return np.asarray([v[3], v[2], v[0], v[2]])
@njit
def swizzle_n4_wzxw(v):
    return np.asarray([v[3], v[2], v[0], v[3]])
@njit
def swizzle_n4_wzyx(v):
    return np.asarray([v[3], v[2], v[1], v[0]])
@njit
def swizzle_n4_wzyy(v):
    return np.asarray([v[3], v[2], v[1], v[1]])
@njit
def swizzle_n4_wzyz(v):
    return np.asarray([v[3], v[2], v[1], v[2]])
@njit
def swizzle_n4_wzyw(v):
    return np.asarray([v[3], v[2], v[1], v[3]])
@njit
def swizzle_n4_wzzx(v):
    return np.asarray([v[3], v[2], v[2], v[0]])
@njit
def swizzle_n4_wzzy(v):
    return np.asarray([v[3], v[2], v[2], v[1]])
@njit
def swizzle_n4_wzzz(v):
    return np.asarray([v[3], v[2], v[2], v[2]])
@njit
def swizzle_n4_wzzw(v):
    return np.asarray([v[3], v[2], v[2], v[3]])
@njit
def swizzle_n4_wzwx(v):
    return np.asarray([v[3], v[2], v[3], v[0]])
@njit
def swizzle_n4_wzwy(v):
    return np.asarray([v[3], v[2], v[3], v[1]])
@njit
def swizzle_n4_wzwz(v):
    return np.asarray([v[3], v[2], v[3], v[2]])
@njit
def swizzle_n4_wzww(v):
    return np.asarray([v[3], v[2], v[3], v[3]])
@njit
def swizzle_n4_wwxx(v):
    return np.asarray([v[3], v[3], v[0], v[0]])
@njit
def swizzle_n4_wwxy(v):
    return np.asarray([v[3], v[3], v[0], v[1]])
@njit
def swizzle_n4_wwxz(v):
    return np.asarray([v[3], v[3], v[0], v[2]])
@njit
def swizzle_n4_wwxw(v):
    return np.asarray([v[3], v[3], v[0], v[3]])
@njit
def swizzle_n4_wwyx(v):
    return np.asarray([v[3], v[3], v[1], v[0]])
@njit
def swizzle_n4_wwyy(v):
    return np.asarray([v[3], v[3], v[1], v[1]])
@njit
def swizzle_n4_wwyz(v):
    return np.asarray([v[3], v[3], v[1], v[2]])
@njit
def swizzle_n4_wwyw(v):
    return np.asarray([v[3], v[3], v[1], v[3]])
@njit
def swizzle_n4_wwzx(v):
    return np.asarray([v[3], v[3], v[2], v[0]])
@njit
def swizzle_n4_wwzy(v):
    return np.asarray([v[3], v[3], v[2], v[1]])
@njit
def swizzle_n4_wwzz(v):
    return np.asarray([v[3], v[3], v[2], v[2]])
@njit
def swizzle_n4_wwzw(v):
    return np.asarray([v[3], v[3], v[2], v[3]])
@njit
def swizzle_n4_wwwx(v):
    return np.asarray([v[3], v[3], v[3], v[0]])
@njit
def swizzle_n4_wwwy(v):
    return np.asarray([v[3], v[3], v[3], v[1]])
@njit
def swizzle_n4_wwwz(v):
    return np.asarray([v[3], v[3], v[3], v[2]])
@njit
def swizzle_n4_wwww(v):
    return np.asarray([v[3], v[3], v[3], v[3]])
@njit
def swizzle_set_n_x(v, val):
    raise
@njit
def swizzle_set_n2_x(v, val):
    v[0] = val
    return val
@njit
def swizzle_set_n2_y(v, val):
    v[1] = val
    return val
@njit
def swizzle_set_n2_xy(v, val):
    v[0] = val[0]
    v[1] = val[1]
    return val
@njit
def swizzle_set_n2_yx(v, val):
    v[1] = val[0]
    v[0] = val[1]
    return val
@njit
def swizzle_set_n3_x(v, val):
    v[0] = val
    return val
@njit
def swizzle_set_n3_y(v, val):
    v[1] = val
    return val
@njit
def swizzle_set_n3_z(v, val):
    v[2] = val
    return val
@njit
def swizzle_set_n3_xy(v, val):
    v[0] = val[0]
    v[1] = val[1]
    return val
@njit
def swizzle_set_n3_xz(v, val):
    v[0] = val[0]
    v[2] = val[1]
    return val
@njit
def swizzle_set_n3_yx(v, val):
    v[1] = val[0]
    v[0] = val[1]
    return val
@njit
def swizzle_set_n3_yz(v, val):
    v[1] = val[0]
    v[2] = val[1]
    return val
@njit
def swizzle_set_n3_zx(v, val):
    v[2] = val[0]
    v[0] = val[1]
    return val
@njit
def swizzle_set_n3_zy(v, val):
    v[2] = val[0]
    v[1] = val[1]
    return val
@njit
def swizzle_set_n3_xyz(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n3_xzy(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n3_yxz(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n3_yzx(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n3_zxy(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n3_zyx(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_x(v, val):
    v[0] = val
    return val
@njit
def swizzle_set_n4_y(v, val):
    v[1] = val
    return val
@njit
def swizzle_set_n4_z(v, val):
    v[2] = val
    return val
@njit
def swizzle_set_n4_w(v, val):
    v[3] = val
    return val
@njit
def swizzle_set_n4_xy(v, val):
    v[0] = val[0]
    v[1] = val[1]
    return val
@njit
def swizzle_set_n4_xz(v, val):
    v[0] = val[0]
    v[2] = val[1]
    return val
@njit
def swizzle_set_n4_xw(v, val):
    v[0] = val[0]
    v[3] = val[1]
    return val
@njit
def swizzle_set_n4_yx(v, val):
    v[1] = val[0]
    v[0] = val[1]
    return val
@njit
def swizzle_set_n4_yz(v, val):
    v[1] = val[0]
    v[2] = val[1]
    return val
@njit
def swizzle_set_n4_yw(v, val):
    v[1] = val[0]
    v[3] = val[1]
    return val
@njit
def swizzle_set_n4_zx(v, val):
    v[2] = val[0]
    v[0] = val[1]
    return val
@njit
def swizzle_set_n4_zy(v, val):
    v[2] = val[0]
    v[1] = val[1]
    return val
@njit
def swizzle_set_n4_zw(v, val):
    v[2] = val[0]
    v[3] = val[1]
    return val
@njit
def swizzle_set_n4_wx(v, val):
    v[3] = val[0]
    v[0] = val[1]
    return val
@njit
def swizzle_set_n4_wy(v, val):
    v[3] = val[0]
    v[1] = val[1]
    return val
@njit
def swizzle_set_n4_wz(v, val):
    v[3] = val[0]
    v[2] = val[1]
    return val
@njit
def swizzle_set_n4_xyz(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n4_xyw(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    return val
@njit
def swizzle_set_n4_xzy(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n4_xzw(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    return val
@njit
def swizzle_set_n4_xwy(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n4_xwz(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n4_yxz(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n4_yxw(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    return val
@njit
def swizzle_set_n4_yzx(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_yzw(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    return val
@njit
def swizzle_set_n4_ywx(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_ywz(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n4_zxy(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n4_zxw(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    return val
@njit
def swizzle_set_n4_zyx(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_zyw(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    return val
@njit
def swizzle_set_n4_zwx(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_zwy(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n4_wxy(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n4_wxz(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n4_wyx(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_wyz(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    return val
@njit
def swizzle_set_n4_wzx(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    return val
@njit
def swizzle_set_n4_wzy(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    return val
@njit
def swizzle_set_n4_xyzw(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    v[3] = val[3]
    return val
@njit
def swizzle_set_n4_xywz(v, val):
    v[0] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    v[2] = val[3]
    return val
@njit
def swizzle_set_n4_xzyw(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    v[3] = val[3]
    return val
@njit
def swizzle_set_n4_xzwy(v, val):
    v[0] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    v[1] = val[3]
    return val
@njit
def swizzle_set_n4_xwyz(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    v[2] = val[3]
    return val
@njit
def swizzle_set_n4_xwzy(v, val):
    v[0] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    v[1] = val[3]
    return val
@njit
def swizzle_set_n4_yxzw(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    v[3] = val[3]
    return val
@njit
def swizzle_set_n4_yxwz(v, val):
    v[1] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    v[2] = val[3]
    return val
@njit
def swizzle_set_n4_yzxw(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    v[3] = val[3]
    return val
@njit
def swizzle_set_n4_yzwx(v, val):
    v[1] = val[0]
    v[2] = val[1]
    v[3] = val[2]
    v[0] = val[3]
    return val
@njit
def swizzle_set_n4_ywxz(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    v[2] = val[3]
    return val
@njit
def swizzle_set_n4_ywzx(v, val):
    v[1] = val[0]
    v[3] = val[1]
    v[2] = val[2]
    v[0] = val[3]
    return val
@njit
def swizzle_set_n4_zxyw(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    v[3] = val[3]
    return val
@njit
def swizzle_set_n4_zxwy(v, val):
    v[2] = val[0]
    v[0] = val[1]
    v[3] = val[2]
    v[1] = val[3]
    return val
@njit
def swizzle_set_n4_zyxw(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    v[3] = val[3]
    return val
@njit
def swizzle_set_n4_zywx(v, val):
    v[2] = val[0]
    v[1] = val[1]
    v[3] = val[2]
    v[0] = val[3]
    return val
@njit
def swizzle_set_n4_zwxy(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[0] = val[2]
    v[1] = val[3]
    return val
@njit
def swizzle_set_n4_zwyx(v, val):
    v[2] = val[0]
    v[3] = val[1]
    v[1] = val[2]
    v[0] = val[3]
    return val
@njit
def swizzle_set_n4_wxyz(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[1] = val[2]
    v[2] = val[3]
    return val
@njit
def swizzle_set_n4_wxzy(v, val):
    v[3] = val[0]
    v[0] = val[1]
    v[2] = val[2]
    v[1] = val[3]
    return val
@njit
def swizzle_set_n4_wyxz(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[0] = val[2]
    v[2] = val[3]
    return val
@njit
def swizzle_set_n4_wyzx(v, val):
    v[3] = val[0]
    v[1] = val[1]
    v[2] = val[2]
    v[0] = val[3]
    return val
@njit
def swizzle_set_n4_wzxy(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[0] = val[2]
    v[1] = val[3]
    return val
@njit
def swizzle_set_n4_wzyx(v, val):
    v[3] = val[0]
    v[2] = val[1]
    v[1] = val[2]
    v[0] = val[3]
    return val
@njit
def swizzle_t_n_x(v):
    return v
@njit
def swizzle_t_n_xx(v):
    return np.column_stack((v, v))
@njit
def swizzle_t_n_xxx(v):
    return np.column_stack((v, v, v))
@njit
def swizzle_t_n_xxxx(v):
    return np.column_stack((v, v, v, v))
@njit
def swizzle_t_n2_x(v):
    return np.copy(v[..., 0])
@njit
def swizzle_t_n2_y(v):
    return np.copy(v[..., 1])
@njit
def swizzle_t_n2_xx(v):
    return np.column_stack((v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_xy(v):
    return np.column_stack((v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_yx(v):
    return np.column_stack((v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_yy(v):
    return np.column_stack((v[..., 1], v[..., 1]))
@njit
def swizzle_t_n2_xxx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_xxy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_xyx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_xyy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n2_yxx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_yxy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_yyx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_yyy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n2_xxxx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_xxxy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_xxyx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_xxyy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n2_xyxx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_xyxy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_xyyx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_xyyy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n2_yxxx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_yxxy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_yxyx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_yxyy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n2_yyxx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n2_yyxy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n2_yyyx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n2_yyyy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_x(v):
    return np.copy(v[..., 0])
@njit
def swizzle_t_n3_y(v):
    return np.copy(v[..., 1])
@njit
def swizzle_t_n3_z(v):
    return np.copy(v[..., 2])
@njit
def swizzle_t_n3_xx(v):
    return np.column_stack((v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_xy(v):
    return np.column_stack((v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_xz(v):
    return np.column_stack((v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_yx(v):
    return np.column_stack((v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_yy(v):
    return np.column_stack((v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_yz(v):
    return np.column_stack((v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_zx(v):
    return np.column_stack((v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_zy(v):
    return np.column_stack((v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_zz(v):
    return np.column_stack((v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_xxx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_xxy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_xxz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_xyx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_xyy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_xyz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_xzx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_xzy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_xzz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_yxx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_yxy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_yxz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_yyx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_yyy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_yyz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_yzx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_yzy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_yzz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_zxx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_zxy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_zxz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_zyx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_zyy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_zyz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_zzx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_zzy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_zzz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_xxxx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_xxxy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_xxxz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_xxyx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_xxyy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_xxyz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_xxzx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_xxzy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_xxzz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_xyxx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_xyxy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_xyxz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_xyyx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_xyyy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_xyyz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_xyzx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_xyzy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_xyzz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_xzxx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_xzxy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_xzxz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_xzyx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_xzyy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_xzyz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_xzzx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_xzzy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_xzzz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_yxxx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_yxxy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_yxxz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_yxyx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_yxyy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_yxyz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_yxzx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_yxzy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_yxzz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_yyxx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_yyxy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_yyxz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_yyyx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_yyyy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_yyyz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_yyzx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_yyzy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_yyzz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_yzxx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_yzxy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_yzxz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_yzyx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_yzyy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_yzyz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_yzzx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_yzzy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_yzzz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_zxxx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_zxxy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_zxxz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_zxyx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_zxyy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_zxyz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_zxzx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_zxzy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_zxzz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_zyxx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_zyxy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_zyxz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_zyyx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_zyyy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_zyyz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_zyzx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_zyzy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_zyzz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n3_zzxx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n3_zzxy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n3_zzxz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n3_zzyx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n3_zzyy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n3_zzyz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n3_zzzx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n3_zzzy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n3_zzzz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_x(v):
    return np.copy(v[..., 0])
@njit
def swizzle_t_n4_y(v):
    return np.copy(v[..., 1])
@njit
def swizzle_t_n4_z(v):
    return np.copy(v[..., 2])
@njit
def swizzle_t_n4_w(v):
    return np.copy(v[..., 3])
@njit
def swizzle_t_n4_xx(v):
    return np.column_stack((v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_xy(v):
    return np.column_stack((v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_xz(v):
    return np.column_stack((v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_xw(v):
    return np.column_stack((v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_yx(v):
    return np.column_stack((v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_yy(v):
    return np.column_stack((v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_yz(v):
    return np.column_stack((v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_yw(v):
    return np.column_stack((v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_zx(v):
    return np.column_stack((v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_zy(v):
    return np.column_stack((v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_zz(v):
    return np.column_stack((v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_zw(v):
    return np.column_stack((v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_wx(v):
    return np.column_stack((v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_wy(v):
    return np.column_stack((v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_wz(v):
    return np.column_stack((v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_ww(v):
    return np.column_stack((v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_xxx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_xxy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_xxz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_xxw(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_xyx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_xyy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_xyz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_xyw(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_xzx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_xzy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_xzz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_xzw(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_xwx(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_xwy(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_xwz(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_xww(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_yxx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_yxy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_yxz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_yxw(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_yyx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_yyy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_yyz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_yyw(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_yzx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_yzy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_yzz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_yzw(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_ywx(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_ywy(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_ywz(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_yww(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_zxx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_zxy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_zxz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_zxw(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_zyx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_zyy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_zyz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_zyw(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_zzx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_zzy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_zzz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_zzw(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_zwx(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_zwy(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_zwz(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_zww(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_wxx(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_wxy(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_wxz(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_wxw(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_wyx(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_wyy(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_wyz(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_wyw(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_wzx(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_wzy(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_wzz(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_wzw(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_wwx(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_wwy(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_wwz(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_www(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_xxxx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_xxxy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_xxxz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_xxxw(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_xxyx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_xxyy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_xxyz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_xxyw(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_xxzx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_xxzy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_xxzz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_xxzw(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_xxwx(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_xxwy(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_xxwz(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_xxww(v):
    return np.column_stack((v[..., 0], v[..., 0], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_xyxx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_xyxy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_xyxz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_xyxw(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_xyyx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_xyyy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_xyyz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_xyyw(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_xyzx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_xyzy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_xyzz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_xyzw(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_xywx(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_xywy(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_xywz(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_xyww(v):
    return np.column_stack((v[..., 0], v[..., 1], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_xzxx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_xzxy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_xzxz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_xzxw(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_xzyx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_xzyy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_xzyz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_xzyw(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_xzzx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_xzzy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_xzzz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_xzzw(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_xzwx(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_xzwy(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_xzwz(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_xzww(v):
    return np.column_stack((v[..., 0], v[..., 2], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_xwxx(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_xwxy(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_xwxz(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_xwxw(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_xwyx(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_xwyy(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_xwyz(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_xwyw(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_xwzx(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_xwzy(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_xwzz(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_xwzw(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_xwwx(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_xwwy(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_xwwz(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_xwww(v):
    return np.column_stack((v[..., 0], v[..., 3], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_yxxx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_yxxy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_yxxz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_yxxw(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_yxyx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_yxyy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_yxyz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_yxyw(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_yxzx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_yxzy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_yxzz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_yxzw(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_yxwx(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_yxwy(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_yxwz(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_yxww(v):
    return np.column_stack((v[..., 1], v[..., 0], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_yyxx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_yyxy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_yyxz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_yyxw(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_yyyx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_yyyy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_yyyz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_yyyw(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_yyzx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_yyzy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_yyzz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_yyzw(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_yywx(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_yywy(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_yywz(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_yyww(v):
    return np.column_stack((v[..., 1], v[..., 1], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_yzxx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_yzxy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_yzxz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_yzxw(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_yzyx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_yzyy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_yzyz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_yzyw(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_yzzx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_yzzy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_yzzz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_yzzw(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_yzwx(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_yzwy(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_yzwz(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_yzww(v):
    return np.column_stack((v[..., 1], v[..., 2], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_ywxx(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_ywxy(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_ywxz(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_ywxw(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_ywyx(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_ywyy(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_ywyz(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_ywyw(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_ywzx(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_ywzy(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_ywzz(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_ywzw(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_ywwx(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_ywwy(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_ywwz(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_ywww(v):
    return np.column_stack((v[..., 1], v[..., 3], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_zxxx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_zxxy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_zxxz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_zxxw(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_zxyx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_zxyy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_zxyz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_zxyw(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_zxzx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_zxzy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_zxzz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_zxzw(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_zxwx(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_zxwy(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_zxwz(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_zxww(v):
    return np.column_stack((v[..., 2], v[..., 0], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_zyxx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_zyxy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_zyxz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_zyxw(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_zyyx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_zyyy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_zyyz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_zyyw(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_zyzx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_zyzy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_zyzz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_zyzw(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_zywx(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_zywy(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_zywz(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_zyww(v):
    return np.column_stack((v[..., 2], v[..., 1], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_zzxx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_zzxy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_zzxz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_zzxw(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_zzyx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_zzyy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_zzyz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_zzyw(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_zzzx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_zzzy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_zzzz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_zzzw(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_zzwx(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_zzwy(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_zzwz(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_zzww(v):
    return np.column_stack((v[..., 2], v[..., 2], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_zwxx(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_zwxy(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_zwxz(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_zwxw(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_zwyx(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_zwyy(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_zwyz(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_zwyw(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_zwzx(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_zwzy(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_zwzz(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_zwzw(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_zwwx(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_zwwy(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_zwwz(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_zwww(v):
    return np.column_stack((v[..., 2], v[..., 3], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_wxxx(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_wxxy(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_wxxz(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_wxxw(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_wxyx(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_wxyy(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_wxyz(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_wxyw(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_wxzx(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_wxzy(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_wxzz(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_wxzw(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_wxwx(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_wxwy(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_wxwz(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_wxww(v):
    return np.column_stack((v[..., 3], v[..., 0], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_wyxx(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_wyxy(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_wyxz(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_wyxw(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_wyyx(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_wyyy(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_wyyz(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_wyyw(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_wyzx(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_wyzy(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_wyzz(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_wyzw(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_wywx(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_wywy(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_wywz(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_wyww(v):
    return np.column_stack((v[..., 3], v[..., 1], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_wzxx(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_wzxy(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_wzxz(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_wzxw(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_wzyx(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_wzyy(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_wzyz(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_wzyw(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_wzzx(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_wzzy(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_wzzz(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_wzzw(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_wzwx(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_wzwy(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_wzwz(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_wzww(v):
    return np.column_stack((v[..., 3], v[..., 2], v[..., 3], v[..., 3]))
@njit
def swizzle_t_n4_wwxx(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 0], v[..., 0]))
@njit
def swizzle_t_n4_wwxy(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 0], v[..., 1]))
@njit
def swizzle_t_n4_wwxz(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 0], v[..., 2]))
@njit
def swizzle_t_n4_wwxw(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 0], v[..., 3]))
@njit
def swizzle_t_n4_wwyx(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 1], v[..., 0]))
@njit
def swizzle_t_n4_wwyy(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 1], v[..., 1]))
@njit
def swizzle_t_n4_wwyz(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 1], v[..., 2]))
@njit
def swizzle_t_n4_wwyw(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 1], v[..., 3]))
@njit
def swizzle_t_n4_wwzx(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 2], v[..., 0]))
@njit
def swizzle_t_n4_wwzy(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 2], v[..., 1]))
@njit
def swizzle_t_n4_wwzz(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 2], v[..., 2]))
@njit
def swizzle_t_n4_wwzw(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 2], v[..., 3]))
@njit
def swizzle_t_n4_wwwx(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 3], v[..., 0]))
@njit
def swizzle_t_n4_wwwy(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 3], v[..., 1]))
@njit
def swizzle_t_n4_wwwz(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 3], v[..., 2]))
@njit
def swizzle_t_n4_wwww(v):
    return np.column_stack((v[..., 3], v[..., 3], v[..., 3], v[..., 3]))
@njit
def swizzle_set_t_n_x(v, val):
    np.copyto(v, val)
    return val
@njit
def swizzle_set_t_n2_x(v, val):
    v[..., 0] = val
    return val
@njit
def swizzle_set_t_n2_y(v, val):
    v[..., 1] = val
    return val
@njit
def swizzle_set_t_n2_xy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
@njit
def swizzle_set_t_n2_yx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_x(v, val):
    v[..., 0] = val
    return val
@njit
def swizzle_set_t_n3_y(v, val):
    v[..., 1] = val
    return val
@njit
def swizzle_set_t_n3_z(v, val):
    v[..., 2] = val
    return val
@njit
def swizzle_set_t_n3_xy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_xz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_yx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_yz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_zx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_zy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
@njit
def swizzle_set_t_n3_xyz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n3_xzy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n3_yxz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n3_yzx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n3_zxy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n3_zyx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_x(v, val):
    v[..., 0] = val
    return val
@njit
def swizzle_set_t_n4_y(v, val):
    v[..., 1] = val
    return val
@njit
def swizzle_set_t_n4_z(v, val):
    v[..., 2] = val
    return val
@njit
def swizzle_set_t_n4_w(v, val):
    v[..., 3] = val
    return val
@njit
def swizzle_set_t_n4_xy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_xz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_xw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_yx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_yz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_yw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_zx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_zy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_zw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_wx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_wy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_wz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    return val
@njit
def swizzle_set_t_n4_xyz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_xyw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_xzy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_xzw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_xwy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_xwz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_yxz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_yxw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_yzx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_yzw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_ywx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_ywz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_zxy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_zxw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_zyx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_zyw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_zwx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_zwy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_wxy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_wxz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_wyx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_wyz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_wzx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_wzy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    return val
@njit
def swizzle_set_t_n4_xyzw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_xywz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_xzyw(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_xzwy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_xwyz(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_xwzy(v, val):
    v[..., 0] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_yxzw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_yxwz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_yzxw(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_yzwx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_ywxz(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_ywzx(v, val):
    v[..., 1] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_zxyw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_zxwy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_zyxw(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 3] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_zywx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 3] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_zwxy(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_zwyx(v, val):
    v[..., 2] = val[..., 0]
    v[..., 3] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_wxyz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_wxzy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 0] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_wyxz(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 2] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_wyzx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 1] = val[..., 1]
    v[..., 2] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_wzxy(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 0] = val[..., 2]
    v[..., 1] = val[..., 3]
    return val
@njit
def swizzle_set_t_n4_wzyx(v, val):
    v[..., 3] = val[..., 0]
    v[..., 2] = val[..., 1]
    v[..., 1] = val[..., 2]
    v[..., 0] = val[..., 3]
    return val
##@njit
def swizzle_set_and_broadcast_n_x(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n_x(v, val), v
##@njit
def swizzle_set_and_broadcast_n2_x(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n2_x(v, val), v
##@njit
def swizzle_set_and_broadcast_n2_y(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n2_y(v, val), v
##@njit
def swizzle_set_and_broadcast_n2_xy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n2_xy(v, val), v
##@njit
def swizzle_set_and_broadcast_n2_yx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n2_yx(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_x(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_x(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_y(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_y(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_z(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_z(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_xy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xy(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_xz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xz(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_yx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yx(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_yz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yz(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_zx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zx(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_zy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zy(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_xyz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xyz(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_xzy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_xzy(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_yxz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yxz(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_yzx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_yzx(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_zxy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zxy(v, val), v
##@njit
def swizzle_set_and_broadcast_n3_zyx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n3_zyx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_x(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_x(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_y(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_y(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_z(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_z(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_w(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_w(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xyz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xyz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xyw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xyw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xzy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xzw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xwy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xwz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yxz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yxw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yzx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yzw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_ywx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_ywz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zxy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zxw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zyx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zyx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zyw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zyw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zwx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zwy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wxy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wxz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wyx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wyz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wzx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wzy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xyzw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xyzw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xywz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xywz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xzyw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzyw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xzwy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xzwy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xwyz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwyz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_xwzy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_xwzy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yxzw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxzw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yxwz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yxwz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yzxw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzxw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_yzwx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_yzwx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_ywxz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywxz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_ywzx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_ywzx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zxyw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxyw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zxwy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zxwy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zyxw(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zyxw(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zywx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zywx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zwxy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwxy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_zwyx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_zwyx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wxyz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxyz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wxzy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wxzy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wyxz(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyxz(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wyzx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wyzx(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wzxy(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzxy(v, val), v
##@njit
def swizzle_set_and_broadcast_n4_wzyx(v, val):
    v = np.tile(v, (len(val), 1))
    return swizzle_set_t_n4_wzyx(v, val), v
#---end---
#----------------------------------------


#pip install matplotlib numpy numba cupy-cuda11x imageio PyOpenGL glfw

import matplotlib.pyplot as plt
import matplotlib.animation as mpanim
import imageio.v3 as iio
import OpenGL.GL as gl
import glfw
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
    return np.any(v)
def not_all_ifexp_true_t_n(v):
    return not np.all(v)

def array_copy(v):
    return v.copy()

def init_buffer():
    global iResolution
    return np.broadcast_to(np.asarray([[0.0, 0.0, 0.0, 1.0]]), (iResolution[0] * iResolution[1], 4))
def buffer_to_tex(v):
    global iResolution
    img = np.asarray(np.array_split(v, iResolution[1]))
    img = img[::-1, :, :] # Flip vertically.
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
    return np.flip(np.transpose(img, (1, 0, 2)), 1)

def load_tex_2d(file):
    data = iio.imread(file)
    if len(data.shape) == 2:
        return np.flip(np.transpose(data), 1)
    else:
        return np.flip(np.transpose(data, (1, 0, 2)), 1)

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
    return np.moveaxis(np.stack(datas), 0, 2)

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
    return data
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
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
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
    V = np.array_split(V, iResolution[1])
    ax.clear()
    cameraz = 0.0
    if hasattr(sys.modules[__name__], "get_camera_z"):
        cameraz = get_camera_z()
    elif hasattr(sys.modules[__name__], "get_camera_pos"):
        cameraz = get_camera_pos()[2]
    else:
        cameraz = 0.0
    info = "time:{0:.3f} frame time:{1:.3f} camera z:{2:.2f}".format(iTime, iTimeDelta, cameraz)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(info)
    ax.text(0.0, 1.0, info)
    im = ax.imshow(V, interpolation='bilinear',
                   origin='lower', extent=[0, iResolution[0], 0, iResolution[1]],
                   vmax=maxv, vmin=-minv)

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
    np.random.seed(19680801)
    iTimeDelta = 0
    iTime = 0
    iFrame = 0

    coordx = np.arange(0.0, iResolution[0])
    coordy = np.arange(0.0, iResolution[1])
    X, Y = np.meshgrid(coordx, coordy)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    fcd = np.column_stack((X, Y))
    #fc = np.broadcast_to(hlsl_float4_n_n_n_n(0.5, 0.5, 0.5, 1.0), (iResolution[0], iResolution[1], 4), axis=0)
    fc = np.asarray([0.5,0.5,0.5,1.0])

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

            img = np.asarray(np.array_split(V, iResolution[1]))
            img = img[::-1, :, :] # Flip vertically.
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

            wtitle = "time:{0:.3f} frame time:{1:.3f} iter:{2}".format(iTime, iTimeDelta, ct)
            display_image(img, g_win_zoom, g_win_size, wtitle)
            #tensor_pools.RecycleAll()
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
        '''#
        iterCount = 1 if g_is_profiling else 1000
        for ct in range(iterCount):
            curTime = time.time()
            iTimeDelta = curTime - lastTime
            iTime += iTimeDelta
            iFrame += 1
            if iTimeDelta > 0.01:
                iFrameRate = iFrameRate * 0.7 + 0.3 / iTimeDelta
            lastTime = curTime
            V = shader_main(fc, fcd)
            maxv = abs(V).max()
            minv = -abs(V).max()
            V = np.array_split(V, iResolution[1])
            ax.clear()
            cameraz = 0.0
            if hasattr(sys.modules[__name__], "get_camera_z"):
                cameraz = get_camera_z()
            elif hasattr(sys.modules[__name__], "get_camera_pos"):
                cameraz = get_camera_pos()[2]
            else:
                cameraz = 0.0
            ax.text(0.0, 1.0, "time:{0:.3f} frame time:{1:.3f} camera z:{2:.2f}".format(iTime, iTimeDelta, cameraz))
            im = ax.imshow(V, interpolation='bilinear',
                           origin='lower', extent=[0, iResolution[0], 0, iResolution[1]],
                           vmax=maxv, vmin=-minv)
            plt.pause(0.1)
        #'''
        ani = mpanim.FuncAnimation(fig, functools.partial(update, ax = ax, fc = fc, fcd = fcd), interval = 100.0, repeat = not g_is_profiling)
        plt.show()

def main_entry_autodiff():
    pass

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

_ = None
g_last_time = 0.0
g_glfw_window = None
g_show_with_opengl = False
g_is_profiling = False
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

def compute_dispatch_templ():
    global _FogTexSize, _CameraDepthTexture, _WorldSpaceCameraPos, _UNITY_MATRIX_I_VP, _LAST_UNITY_MATRIX_VP
    _FogTexSize = np.asarray([128.0, 128.0])
    _CameraDepthTexture = iio.imread("shaderlib/noise4.jpg")
    _WorldSpaceCameraPos = np.asarray([0.0, 0.0, 0.0])
    _UNITY_MATRIX_I_VP = np.asarray([[-0.72308,-0.02593,1711.43000,-0.13753],[-0.00007,0.57599,23.63344,-0.06164],[0.62276,-0.03004,1719.61400,-0.23995],[0.00000,0.00000,3.33233,0.00100]])
    _LAST_UNITY_MATRIX_VP = np.asarray([[-0.79400,-0.00008,0.68384,54.89616],[-0.07779,1.72796,-0.09012,74.19912],[0.00020,0.00002,0.00023,0.08257],[-0.65104,-0.06873,-0.75592,724.93730]])

    groupId = np.asarray([[0,0],[0,1],[1,0],[1,1]])
    xs, ys = np.meshgrid([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    groupThreadId = np.column_stack((xs, ys))
    groupCt = len(groupId)
    groupThreadCt = len(groupThreadId)
    ct = groupCt * groupThreadCt
    groupIds = groupId.repeat(groupThreadCt, axis=0)
    groupThreadIds = groupThreadId.reshape(1, groupThreadCt, 2).repeat(len(groupId), axis=0).reshape(ct, 2)
    dispThreadIds = groupIds * np.asarray([8,8]) + groupThreadIds
    shader_main(dispThreadIds, groupIds, groupThreadIds)

def shader_dispatch_templ():
    pass

def compute_dispatch(fc, fcd, entry):
    pass

def shader_dispatch(fc, fcd, entry):
    pass

def glsl_vec3_f(arg):
	return h_f3_n_n_n(arg, arg, arg)

def glsl_vec4_f(arg):
	return h_f4_n_n_n_n(arg, arg, arg, arg)

def rot_f(a):
	return h_f2x2_n_n_n_n(h_cos_n(a), h_sin_n(a), h_sub_f(h_sin_n(a)), h_cos_n(a))

def glsl_float_x4_ctor_f_f_f_f(v0, v1, v2, v3):
	__arr_tmp = array_init_an([v0, v1, v2, v3])
	return __arr_tmp

def glsl_vec3_x4_ctor_f3_f3_f3_f3(v0, v1, v2, v3):
	__arr_tmp = array_init_an3([v0, v1, v2, v3])
	return __arr_tmp


def noise_f2_arr(x):
	global iChannel0, s_linear_clamp_sampler
	p = h_floor_t_v(x)
	f = h_frac_t_v(x)
	f = h_mul_t_vf_t_vf(h_mul_t_vf_t_vf(f, f), h_sub_f_t_vf(3.0, h_mul_f_t_vf(2.0, f)))
	uv = h_add_t_vf_t_vf(swizzle_t_n2_xy(p), swizzle_t_n2_xy(f))
	return swizzle_t_n4_x(Texture2D_SampleLevel_n_t_v_n(iChannel0, s_linear_clamp_sampler, h_div_t_vf_f(h_add_t_vf_f(uv, 0.5), 256.0), 0.0))

def noise_f3_arr(x):
	global iChannel0, s_linear_clamp_sampler
	p = h_floor_t_v(x)
	f = h_frac_t_v(x)
	f = h_mul_t_vf_t_vf(h_mul_t_vf_t_vf(f, f), h_sub_f_t_vf(3.0, h_mul_f_t_vf(2.0, f)))
	uv = h_add_t_vf_t_vf(h_add_t_vf_t_vf(swizzle_t_n3_xy(p), h_mul_vf_t_f(h_f2_n_n(37.0, 17.0), swizzle_t_n3_z(p))), swizzle_t_n3_xy(f))
	rg = swizzle_t_n4_yx(Texture2D_SampleLevel_n_t_v_n(iChannel0, s_linear_clamp_sampler, h_div_t_vf_f(h_add_t_vf_f(uv, 0.5), 256.0), 0.0))
	return h_lerp_t_n_t_n_t_n(swizzle_t_n2_x(rg), swizzle_t_n2_y(rg), swizzle_t_n3_z(f))

def cloudMap_f3_f_arr(p, ani):
	r = h_div_t_vf_f(p, 260.4167)
	den = h_add_f_t_f(-1.8, h_cos_t_n(h_sub_t_f_f(h_mul_t_f_f(swizzle_t_n3_y(r), 5.0), 4.2999999999999998)))
	f = 0.0
	q = h_add_t_vf_vf(h_mul_t_vf_vf(h_mul_f_t_vf(2.5, r), h_f3_n_n_n(0.75, 1.0, 0.75)), h_mul_vf_f(h_mul_vf_f(h_f3_n_n_n(1.0, 2.0, 1.0), ani), 0.14999999999999999))
	f = h_mul_f_t_f(0.5, noise_f3_arr(q))
	q = h_sub_t_vf_vf(h_mul_t_vf_f(q, 2.02), h_mul_vf_f(h_mul_vf_f(h_f3_n_n_n(-1.0, 1.0, -1.0), ani), 0.14999999999999999))
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.25, noise_f3_arr(q)))
	q = h_add_t_vf_vf(h_mul_t_vf_f(q, 2.0299999999999998), h_mul_vf_f(h_mul_vf_f(h_f3_n_n_n(1.0, -1.0, 1.0), ani), 0.14999999999999999))
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.125, noise_f3_arr(q)))
	q = h_sub_t_vf_vf(h_mul_t_vf_f(q, 2.0099999999999998), h_mul_vf_f(h_mul_vf_f(h_f3_n_n_n(1.0, 1.0, -1.0), ani), 0.14999999999999999))
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.0625, noise_f3_arr(q)))
	q = h_add_t_vf_vf(h_mul_t_vf_f(q, 2.02), h_mul_vf_f(h_mul_vf_f(h_f3_n_n_n(1.0, 1.0, 1.0), ani), 0.14999999999999999))
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.03125, noise_f3_arr(q)))
	return h_mul_f_t_f(0.065000000000000002, h_clamp_t_n_n_n(h_add_t_f_t_f(den, h_mul_f_t_f(4.4000000000000004, f)), 0.0, 1.0))

def hash_f_arr(n):
	return h_frac_t_n(h_mul_t_f_f(h_sin_t_n(n), 43758.545299999998))

def terrainMap_f3_arr(p):
	global iChannel1, s_linear_clamp_sampler, m2
	return h_add_t_f_t_f(h_sub_t_f_f(h_mul_t_f_t_f(h_mul_t_f_f(swizzle_t_n4_x(Texture2D_SampleLevel_n_t_v_n(iChannel1, s_linear_clamp_sampler, h_mul_t_vf_f(h_matmul_f2x2_t_f2(m2, h_sub_t_vf(swizzle_t_n3_zx(p))), 4.6E-5), 0.0)), 600.0), h_smoothstep_n_n_t_n(820.0, 1000.0, h_length_t_v(swizzle_t_n3_xz(p)))), 2.0), h_mul_t_f_f(noise_f2_arr(h_mul_t_vf_f(swizzle_t_n3_xz(p), 0.5)), 15.0))

def fbm_f3_arr(p):
	global m3
	f = 0.0
	f = h_add_f_t_f(f, h_mul_f_t_f(0.5, noise_f3_arr(p)))
	p = h_mul_t_vf_f(h_matmul_t_f3_f3x3(p, m3), 2.02)
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.25, noise_f3_arr(p)))
	p = h_mul_t_vf_f(h_matmul_t_f3_f3x3(p, m3), 2.0299999999999998)
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.125, noise_f3_arr(p)))
	p = h_mul_t_vf_f(h_matmul_t_f3_f3x3(p, m3), 2.0099999999999998)
	f = h_add_t_f_t_f(f, h_mul_f_t_f(0.0625, noise_f3_arr(p)))
	return h_div_t_f_f(f, 0.9375)

def glsl_vec3_f_arr(arg):
	return h_t_f3_t_n_t_n_t_n(arg, arg, arg)

def intersectPlane_f3_f3_f_f_arr1(ro, rd, height, dist):
	_func_ret_val_183 = False
	_func_ret_flag_183 = False
	_vecif_354_exp = h_equal_t_n_n(swizzle_t_n3_y(rd), 0.0)
	if any_ifexp_true_t_n(_vecif_354_exp):
		_vecif_354__func_ret_flag_183 = _func_ret_flag_183
		_vecif_354__func_ret_val_183 = _func_ret_val_183
		_vecif_354__func_ret_flag_183 = True
		_vecif_354__func_ret_val_183 = False
		_func_ret_flag_183 = h_where_t_n_n_n(_vecif_354_exp, _vecif_354__func_ret_flag_183, _func_ret_flag_183)
		_func_ret_val_183 = h_where_t_n_n_n(_vecif_354_exp, _vecif_354__func_ret_val_183, _func_ret_val_183)
	else:
		_func_ret_flag_183 = h_broadcast_t_b_b(ro, _func_ret_flag_183)
		_func_ret_val_183 = h_broadcast_t_b_b(ro, _func_ret_val_183)
	_vecif_355_exp = h_not_t_n(_func_ret_flag_183)
	if any_ifexp_true_t_n(_vecif_355_exp):
		_vecif_355_dist = dist
		_vecif_355__func_ret_flag_183 = _func_ret_flag_183
		_vecif_355__func_ret_val_183 = _func_ret_val_183
		d = h_div_t_f_t_f(h_sub_t_f(h_sub_t_f_f(swizzle_t_n3_y(ro), height)), swizzle_t_n3_y(rd))
		d = h_min_n_t_n(1.0E+5, d)
		_vecif_356_exp_0 = h_and_t_n_t_n(h_greater_than_t_n_n(d, 0.0), h_less_than_t_n_t_n(d, _vecif_355_dist))
		if any_ifexp_true_t_n(_vecif_356_exp_0):
			_vecif_356_dist = _vecif_355_dist
			_vecif_356__func_ret_flag_183 = _vecif_355__func_ret_flag_183
			_vecif_356__func_ret_val_183 = _vecif_355__func_ret_val_183
			_vecif_356_dist = d
			_vecif_356__func_ret_flag_183 = h_broadcast_t_b_b(_vecif_356__func_ret_flag_183, True)
			_vecif_356__func_ret_val_183 = h_broadcast_t_b_b(_vecif_356__func_ret_val_183, True)
			_vecif_355_dist = h_where_t_n_t_n_t_n(_vecif_356_exp_0, _vecif_356_dist, _vecif_355_dist)
			_vecif_355__func_ret_flag_183 = h_where_t_n_t_n_t_n(_vecif_356_exp_0, _vecif_356__func_ret_flag_183, _vecif_355__func_ret_flag_183)
			_vecif_355__func_ret_val_183 = h_where_t_n_t_n_t_n(_vecif_356_exp_0, _vecif_356__func_ret_val_183, _vecif_355__func_ret_val_183)
		if not_all_ifexp_true_t_n(_vecif_356_exp_0):
			_vecif_356__func_ret_flag_183 = _vecif_355__func_ret_flag_183
			_vecif_356__func_ret_val_183 = _vecif_355__func_ret_val_183
			_vecif_356__func_ret_flag_183 = h_broadcast_t_b_b(_vecif_356__func_ret_flag_183, True)
			_vecif_356__func_ret_val_183 = h_broadcast_t_b_b(_vecif_356__func_ret_val_183, False)
			#condition: not _vecif_356_exp_0
			_vecif_355__func_ret_flag_183 = h_where_t_n_t_n_t_n(_vecif_356_exp_0, _vecif_355__func_ret_flag_183, _vecif_356__func_ret_flag_183)
			#condition: not _vecif_356_exp_0
			_vecif_355__func_ret_val_183 = h_where_t_n_t_n_t_n(_vecif_356_exp_0, _vecif_355__func_ret_val_183, _vecif_356__func_ret_val_183)
		dist = h_where_t_n_t_n_t_n(_vecif_355_exp, _vecif_355_dist, dist)
		_func_ret_flag_183 = h_where_t_n_t_n_t_n(_vecif_355_exp, _vecif_355__func_ret_flag_183, _func_ret_flag_183)
		_func_ret_val_183 = h_where_t_n_t_n_t_n(_vecif_355_exp, _vecif_355__func_ret_val_183, _func_ret_val_183)
	return _func_ret_val_183, dist

def raymarchClouds_f3_f3_f3_f3_f_f_f_arr(ro, rd, bgc, fgc, startdist, maxdist, ani):
	global iTime, lig
	t = h_add_t_f_t_f(startdist, h_mul_f_t_f(5.208333, hash_f_arr(h_add_t_f_f(h_add_t_f_t_f(swizzle_t_n3_x(rd), h_mul_f_t_f(35.698722099999998, swizzle_t_n3_y(rd))), h_add_f_f(iTime, 285.0)))))
	sum = glsl_vec4_f(0.0)
	_cont_flag_187 = False
	if True:
		i = 0
		_cont_flag_187 = False
		_vecif_290_exp = h_or_n_t_n(h_greater_than_n_n(swizzle_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_290_exp):
			_vecif_290__cont_flag_187 = _cont_flag_187
			_vecif_290__cont_flag_187 = True
			_cont_flag_187 = h_where_t_n_n_n(_vecif_290_exp, _vecif_290__cont_flag_187, _cont_flag_187)
		else:
			_cont_flag_187 = h_broadcast_t_b_b(ro, _cont_flag_187)
		_vecif_291_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_291_exp):
			_vecif_291_sum = sum
			_vecif_291_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_291_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_291_sum = h_add_vf_t_vf(_vecif_291_sum, h_mul_t_vf_f(col, h_sub_f_f(1.0, swizzle_n4_w(_vecif_291_sum))))
			_vecif_291_t = h_add_t_f_t_f(_vecif_291_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_291_t, 0.012)))
			sum = h_where_t_n_t_v_v(_vecif_291_exp, _vecif_291_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_291_exp, _vecif_291_t, t)
		else:
			sum = h_broadcast_t_f4_f4(ro, sum)
		i = 1
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_292_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_292_exp):
			_vecif_292__cont_flag_187 = _cont_flag_187
			_vecif_292__cont_flag_187 = h_broadcast_t_b_b(_vecif_292__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_292_exp, _vecif_292__cont_flag_187, _cont_flag_187)
		_vecif_293_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_293_exp):
			_vecif_293_sum = sum
			_vecif_293_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_293_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_293_sum = h_add_t_vf_t_vf(_vecif_293_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_293_sum))))
			_vecif_293_t = h_add_t_f_t_f(_vecif_293_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_293_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_293_exp, _vecif_293_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_293_exp, _vecif_293_t, t)
		i = 2
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_294_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_294_exp):
			_vecif_294__cont_flag_187 = _cont_flag_187
			_vecif_294__cont_flag_187 = h_broadcast_t_b_b(_vecif_294__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_294_exp, _vecif_294__cont_flag_187, _cont_flag_187)
		_vecif_295_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_295_exp):
			_vecif_295_sum = sum
			_vecif_295_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_295_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_295_sum = h_add_t_vf_t_vf(_vecif_295_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_295_sum))))
			_vecif_295_t = h_add_t_f_t_f(_vecif_295_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_295_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_295_exp, _vecif_295_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_295_exp, _vecif_295_t, t)
		i = 3
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_296_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_296_exp):
			_vecif_296__cont_flag_187 = _cont_flag_187
			_vecif_296__cont_flag_187 = h_broadcast_t_b_b(_vecif_296__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_296_exp, _vecif_296__cont_flag_187, _cont_flag_187)
		_vecif_297_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_297_exp):
			_vecif_297_sum = sum
			_vecif_297_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_297_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_297_sum = h_add_t_vf_t_vf(_vecif_297_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_297_sum))))
			_vecif_297_t = h_add_t_f_t_f(_vecif_297_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_297_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_297_exp, _vecif_297_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_297_exp, _vecif_297_t, t)
		i = 4
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_298_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_298_exp):
			_vecif_298__cont_flag_187 = _cont_flag_187
			_vecif_298__cont_flag_187 = h_broadcast_t_b_b(_vecif_298__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_298_exp, _vecif_298__cont_flag_187, _cont_flag_187)
		_vecif_299_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_299_exp):
			_vecif_299_sum = sum
			_vecif_299_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_299_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_299_sum = h_add_t_vf_t_vf(_vecif_299_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_299_sum))))
			_vecif_299_t = h_add_t_f_t_f(_vecif_299_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_299_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_299_exp, _vecif_299_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_299_exp, _vecif_299_t, t)
		i = 5
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_300_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_300_exp):
			_vecif_300__cont_flag_187 = _cont_flag_187
			_vecif_300__cont_flag_187 = h_broadcast_t_b_b(_vecif_300__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_300_exp, _vecif_300__cont_flag_187, _cont_flag_187)
		_vecif_301_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_301_exp):
			_vecif_301_sum = sum
			_vecif_301_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_301_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_301_sum = h_add_t_vf_t_vf(_vecif_301_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_301_sum))))
			_vecif_301_t = h_add_t_f_t_f(_vecif_301_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_301_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_301_exp, _vecif_301_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_301_exp, _vecif_301_t, t)
		i = 6
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_302_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_302_exp):
			_vecif_302__cont_flag_187 = _cont_flag_187
			_vecif_302__cont_flag_187 = h_broadcast_t_b_b(_vecif_302__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_302_exp, _vecif_302__cont_flag_187, _cont_flag_187)
		_vecif_303_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_303_exp):
			_vecif_303_sum = sum
			_vecif_303_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_303_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_303_sum = h_add_t_vf_t_vf(_vecif_303_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_303_sum))))
			_vecif_303_t = h_add_t_f_t_f(_vecif_303_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_303_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_303_exp, _vecif_303_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_303_exp, _vecif_303_t, t)
		i = 7
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_304_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_304_exp):
			_vecif_304__cont_flag_187 = _cont_flag_187
			_vecif_304__cont_flag_187 = h_broadcast_t_b_b(_vecif_304__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_304_exp, _vecif_304__cont_flag_187, _cont_flag_187)
		_vecif_305_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_305_exp):
			_vecif_305_sum = sum
			_vecif_305_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_305_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_305_sum = h_add_t_vf_t_vf(_vecif_305_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_305_sum))))
			_vecif_305_t = h_add_t_f_t_f(_vecif_305_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_305_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_305_exp, _vecif_305_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_305_exp, _vecif_305_t, t)
		i = 8
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_306_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_306_exp):
			_vecif_306__cont_flag_187 = _cont_flag_187
			_vecif_306__cont_flag_187 = h_broadcast_t_b_b(_vecif_306__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_306_exp, _vecif_306__cont_flag_187, _cont_flag_187)
		_vecif_307_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_307_exp):
			_vecif_307_sum = sum
			_vecif_307_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_307_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_307_sum = h_add_t_vf_t_vf(_vecif_307_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_307_sum))))
			_vecif_307_t = h_add_t_f_t_f(_vecif_307_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_307_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_307_exp, _vecif_307_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_307_exp, _vecif_307_t, t)
		i = 9
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_308_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_308_exp):
			_vecif_308__cont_flag_187 = _cont_flag_187
			_vecif_308__cont_flag_187 = h_broadcast_t_b_b(_vecif_308__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_308_exp, _vecif_308__cont_flag_187, _cont_flag_187)
		_vecif_309_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_309_exp):
			_vecif_309_sum = sum
			_vecif_309_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_309_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_309_sum = h_add_t_vf_t_vf(_vecif_309_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_309_sum))))
			_vecif_309_t = h_add_t_f_t_f(_vecif_309_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_309_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_309_exp, _vecif_309_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_309_exp, _vecif_309_t, t)
		i = 10
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_310_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_310_exp):
			_vecif_310__cont_flag_187 = _cont_flag_187
			_vecif_310__cont_flag_187 = h_broadcast_t_b_b(_vecif_310__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_310_exp, _vecif_310__cont_flag_187, _cont_flag_187)
		_vecif_311_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_311_exp):
			_vecif_311_sum = sum
			_vecif_311_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_311_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_311_sum = h_add_t_vf_t_vf(_vecif_311_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_311_sum))))
			_vecif_311_t = h_add_t_f_t_f(_vecif_311_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_311_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_311_exp, _vecif_311_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_311_exp, _vecif_311_t, t)
		i = 11
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_312_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_312_exp):
			_vecif_312__cont_flag_187 = _cont_flag_187
			_vecif_312__cont_flag_187 = h_broadcast_t_b_b(_vecif_312__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_312_exp, _vecif_312__cont_flag_187, _cont_flag_187)
		_vecif_313_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_313_exp):
			_vecif_313_sum = sum
			_vecif_313_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_313_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_313_sum = h_add_t_vf_t_vf(_vecif_313_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_313_sum))))
			_vecif_313_t = h_add_t_f_t_f(_vecif_313_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_313_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_313_exp, _vecif_313_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_313_exp, _vecif_313_t, t)
		i = 12
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_314_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_314_exp):
			_vecif_314__cont_flag_187 = _cont_flag_187
			_vecif_314__cont_flag_187 = h_broadcast_t_b_b(_vecif_314__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_314_exp, _vecif_314__cont_flag_187, _cont_flag_187)
		_vecif_315_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_315_exp):
			_vecif_315_sum = sum
			_vecif_315_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_315_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_315_sum = h_add_t_vf_t_vf(_vecif_315_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_315_sum))))
			_vecif_315_t = h_add_t_f_t_f(_vecif_315_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_315_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_315_exp, _vecif_315_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_315_exp, _vecif_315_t, t)
		i = 13
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_316_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_316_exp):
			_vecif_316__cont_flag_187 = _cont_flag_187
			_vecif_316__cont_flag_187 = h_broadcast_t_b_b(_vecif_316__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_316_exp, _vecif_316__cont_flag_187, _cont_flag_187)
		_vecif_317_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_317_exp):
			_vecif_317_sum = sum
			_vecif_317_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_317_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_317_sum = h_add_t_vf_t_vf(_vecif_317_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_317_sum))))
			_vecif_317_t = h_add_t_f_t_f(_vecif_317_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_317_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_317_exp, _vecif_317_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_317_exp, _vecif_317_t, t)
		i = 14
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_318_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_318_exp):
			_vecif_318__cont_flag_187 = _cont_flag_187
			_vecif_318__cont_flag_187 = h_broadcast_t_b_b(_vecif_318__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_318_exp, _vecif_318__cont_flag_187, _cont_flag_187)
		_vecif_319_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_319_exp):
			_vecif_319_sum = sum
			_vecif_319_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_319_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_319_sum = h_add_t_vf_t_vf(_vecif_319_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_319_sum))))
			_vecif_319_t = h_add_t_f_t_f(_vecif_319_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_319_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_319_exp, _vecif_319_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_319_exp, _vecif_319_t, t)
		i = 15
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_320_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_320_exp):
			_vecif_320__cont_flag_187 = _cont_flag_187
			_vecif_320__cont_flag_187 = h_broadcast_t_b_b(_vecif_320__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_320_exp, _vecif_320__cont_flag_187, _cont_flag_187)
		_vecif_321_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_321_exp):
			_vecif_321_sum = sum
			_vecif_321_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_321_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_321_sum = h_add_t_vf_t_vf(_vecif_321_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_321_sum))))
			_vecif_321_t = h_add_t_f_t_f(_vecif_321_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_321_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_321_exp, _vecif_321_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_321_exp, _vecif_321_t, t)
		i = 16
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_322_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_322_exp):
			_vecif_322__cont_flag_187 = _cont_flag_187
			_vecif_322__cont_flag_187 = h_broadcast_t_b_b(_vecif_322__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_322_exp, _vecif_322__cont_flag_187, _cont_flag_187)
		_vecif_323_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_323_exp):
			_vecif_323_sum = sum
			_vecif_323_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_323_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_323_sum = h_add_t_vf_t_vf(_vecif_323_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_323_sum))))
			_vecif_323_t = h_add_t_f_t_f(_vecif_323_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_323_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_323_exp, _vecif_323_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_323_exp, _vecif_323_t, t)
		i = 17
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_324_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_324_exp):
			_vecif_324__cont_flag_187 = _cont_flag_187
			_vecif_324__cont_flag_187 = h_broadcast_t_b_b(_vecif_324__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_324_exp, _vecif_324__cont_flag_187, _cont_flag_187)
		_vecif_325_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_325_exp):
			_vecif_325_sum = sum
			_vecif_325_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_325_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_325_sum = h_add_t_vf_t_vf(_vecif_325_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_325_sum))))
			_vecif_325_t = h_add_t_f_t_f(_vecif_325_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_325_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_325_exp, _vecif_325_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_325_exp, _vecif_325_t, t)
		i = 18
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_326_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_326_exp):
			_vecif_326__cont_flag_187 = _cont_flag_187
			_vecif_326__cont_flag_187 = h_broadcast_t_b_b(_vecif_326__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_326_exp, _vecif_326__cont_flag_187, _cont_flag_187)
		_vecif_327_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_327_exp):
			_vecif_327_sum = sum
			_vecif_327_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_327_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_327_sum = h_add_t_vf_t_vf(_vecif_327_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_327_sum))))
			_vecif_327_t = h_add_t_f_t_f(_vecif_327_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_327_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_327_exp, _vecif_327_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_327_exp, _vecif_327_t, t)
		i = 19
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_328_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_328_exp):
			_vecif_328__cont_flag_187 = _cont_flag_187
			_vecif_328__cont_flag_187 = h_broadcast_t_b_b(_vecif_328__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_328_exp, _vecif_328__cont_flag_187, _cont_flag_187)
		_vecif_329_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_329_exp):
			_vecif_329_sum = sum
			_vecif_329_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_329_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_329_sum = h_add_t_vf_t_vf(_vecif_329_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_329_sum))))
			_vecif_329_t = h_add_t_f_t_f(_vecif_329_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_329_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_329_exp, _vecif_329_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_329_exp, _vecif_329_t, t)
		i = 20
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_330_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_330_exp):
			_vecif_330__cont_flag_187 = _cont_flag_187
			_vecif_330__cont_flag_187 = h_broadcast_t_b_b(_vecif_330__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_330_exp, _vecif_330__cont_flag_187, _cont_flag_187)
		_vecif_331_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_331_exp):
			_vecif_331_sum = sum
			_vecif_331_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_331_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_331_sum = h_add_t_vf_t_vf(_vecif_331_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_331_sum))))
			_vecif_331_t = h_add_t_f_t_f(_vecif_331_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_331_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_331_exp, _vecif_331_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_331_exp, _vecif_331_t, t)
		i = 21
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_332_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_332_exp):
			_vecif_332__cont_flag_187 = _cont_flag_187
			_vecif_332__cont_flag_187 = h_broadcast_t_b_b(_vecif_332__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_332_exp, _vecif_332__cont_flag_187, _cont_flag_187)
		_vecif_333_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_333_exp):
			_vecif_333_sum = sum
			_vecif_333_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_333_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_333_sum = h_add_t_vf_t_vf(_vecif_333_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_333_sum))))
			_vecif_333_t = h_add_t_f_t_f(_vecif_333_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_333_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_333_exp, _vecif_333_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_333_exp, _vecif_333_t, t)
		i = 22
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_334_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_334_exp):
			_vecif_334__cont_flag_187 = _cont_flag_187
			_vecif_334__cont_flag_187 = h_broadcast_t_b_b(_vecif_334__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_334_exp, _vecif_334__cont_flag_187, _cont_flag_187)
		_vecif_335_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_335_exp):
			_vecif_335_sum = sum
			_vecif_335_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_335_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_335_sum = h_add_t_vf_t_vf(_vecif_335_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_335_sum))))
			_vecif_335_t = h_add_t_f_t_f(_vecif_335_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_335_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_335_exp, _vecif_335_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_335_exp, _vecif_335_t, t)
		i = 23
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_336_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_336_exp):
			_vecif_336__cont_flag_187 = _cont_flag_187
			_vecif_336__cont_flag_187 = h_broadcast_t_b_b(_vecif_336__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_336_exp, _vecif_336__cont_flag_187, _cont_flag_187)
		_vecif_337_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_337_exp):
			_vecif_337_sum = sum
			_vecif_337_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_337_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_337_sum = h_add_t_vf_t_vf(_vecif_337_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_337_sum))))
			_vecif_337_t = h_add_t_f_t_f(_vecif_337_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_337_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_337_exp, _vecif_337_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_337_exp, _vecif_337_t, t)
		i = 24
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_338_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_338_exp):
			_vecif_338__cont_flag_187 = _cont_flag_187
			_vecif_338__cont_flag_187 = h_broadcast_t_b_b(_vecif_338__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_338_exp, _vecif_338__cont_flag_187, _cont_flag_187)
		_vecif_339_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_339_exp):
			_vecif_339_sum = sum
			_vecif_339_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_339_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_339_sum = h_add_t_vf_t_vf(_vecif_339_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_339_sum))))
			_vecif_339_t = h_add_t_f_t_f(_vecif_339_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_339_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_339_exp, _vecif_339_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_339_exp, _vecif_339_t, t)
		i = 25
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_340_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_340_exp):
			_vecif_340__cont_flag_187 = _cont_flag_187
			_vecif_340__cont_flag_187 = h_broadcast_t_b_b(_vecif_340__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_340_exp, _vecif_340__cont_flag_187, _cont_flag_187)
		_vecif_341_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_341_exp):
			_vecif_341_sum = sum
			_vecif_341_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_341_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_341_sum = h_add_t_vf_t_vf(_vecif_341_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_341_sum))))
			_vecif_341_t = h_add_t_f_t_f(_vecif_341_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_341_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_341_exp, _vecif_341_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_341_exp, _vecif_341_t, t)
		i = 26
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_342_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_342_exp):
			_vecif_342__cont_flag_187 = _cont_flag_187
			_vecif_342__cont_flag_187 = h_broadcast_t_b_b(_vecif_342__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_342_exp, _vecif_342__cont_flag_187, _cont_flag_187)
		_vecif_343_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_343_exp):
			_vecif_343_sum = sum
			_vecif_343_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_343_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_343_sum = h_add_t_vf_t_vf(_vecif_343_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_343_sum))))
			_vecif_343_t = h_add_t_f_t_f(_vecif_343_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_343_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_343_exp, _vecif_343_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_343_exp, _vecif_343_t, t)
		i = 27
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_344_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_344_exp):
			_vecif_344__cont_flag_187 = _cont_flag_187
			_vecif_344__cont_flag_187 = h_broadcast_t_b_b(_vecif_344__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_344_exp, _vecif_344__cont_flag_187, _cont_flag_187)
		_vecif_345_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_345_exp):
			_vecif_345_sum = sum
			_vecif_345_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_345_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_345_sum = h_add_t_vf_t_vf(_vecif_345_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_345_sum))))
			_vecif_345_t = h_add_t_f_t_f(_vecif_345_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_345_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_345_exp, _vecif_345_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_345_exp, _vecif_345_t, t)
		i = 28
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_346_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_346_exp):
			_vecif_346__cont_flag_187 = _cont_flag_187
			_vecif_346__cont_flag_187 = h_broadcast_t_b_b(_vecif_346__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_346_exp, _vecif_346__cont_flag_187, _cont_flag_187)
		_vecif_347_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_347_exp):
			_vecif_347_sum = sum
			_vecif_347_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_347_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_347_sum = h_add_t_vf_t_vf(_vecif_347_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_347_sum))))
			_vecif_347_t = h_add_t_f_t_f(_vecif_347_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_347_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_347_exp, _vecif_347_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_347_exp, _vecif_347_t, t)
		i = 29
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_348_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_348_exp):
			_vecif_348__cont_flag_187 = _cont_flag_187
			_vecif_348__cont_flag_187 = h_broadcast_t_b_b(_vecif_348__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_348_exp, _vecif_348__cont_flag_187, _cont_flag_187)
		_vecif_349_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_349_exp):
			_vecif_349_sum = sum
			_vecif_349_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_349_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_349_sum = h_add_t_vf_t_vf(_vecif_349_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_349_sum))))
			_vecif_349_t = h_add_t_f_t_f(_vecif_349_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_349_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_349_exp, _vecif_349_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_349_exp, _vecif_349_t, t)
		i = 30
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_350_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_350_exp):
			_vecif_350__cont_flag_187 = _cont_flag_187
			_vecif_350__cont_flag_187 = h_broadcast_t_b_b(_vecif_350__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_350_exp, _vecif_350__cont_flag_187, _cont_flag_187)
		_vecif_351_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_351_exp):
			_vecif_351_sum = sum
			_vecif_351_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_351_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_351_sum = h_add_t_vf_t_vf(_vecif_351_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_351_sum))))
			_vecif_351_t = h_add_t_f_t_f(_vecif_351_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_351_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_351_exp, _vecif_351_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_351_exp, _vecif_351_t, t)
		i = 31
		_cont_flag_187 = h_broadcast_t_b_b(_cont_flag_187, False)
		_vecif_352_exp = h_or_t_n_t_n(h_greater_than_t_n_n(swizzle_t_n4_w(sum), 0.98999999999999999), h_greater_than_t_n_t_n(t, maxdist))
		if any_ifexp_true_t_n(_vecif_352_exp):
			_vecif_352__cont_flag_187 = _cont_flag_187
			_vecif_352__cont_flag_187 = h_broadcast_t_b_b(_vecif_352__cont_flag_187, True)
			_cont_flag_187 = h_where_t_n_t_n_t_n(_vecif_352_exp, _vecif_352__cont_flag_187, _cont_flag_187)
		_vecif_353_exp = h_not_t_n(_cont_flag_187)
		if any_ifexp_true_t_n(_vecif_353_exp):
			_vecif_353_sum = sum
			_vecif_353_t = t
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_353_t, rd))
			a = cloudMap_f3_f_arr(pos, ani)
			dif = h_clamp_t_n_n_n(h_add_f_t_f(0.10000000000000001, h_mul_f_t_f(0.80000000000000004, h_sub_t_f_t_f(a, cloudMap_f3_f_arr(h_add_t_vf_vf(pos, h_mul_vf_f(h_mul_vf_f(lig, 0.14999999999999999), 260.4167)), ani)))), 0.0, 0.5)
			col = h_t_f4_t_n3_t_n(h_mul_t_f_t_vf(h_add_f_t_f(1.0, dif), fgc), a)
			swizzle_set_t_n4_xyz(col, h_mul_t_vf_t_f(swizzle_t_n4_xyz(col), swizzle_t_n4_w(col)))
			_vecif_353_sum = h_add_t_vf_t_vf(_vecif_353_sum, h_mul_t_vf_t_f(col, h_sub_f_t_f(1.0, swizzle_t_n4_w(_vecif_353_sum))))
			_vecif_353_t = h_add_t_f_t_f(_vecif_353_t, h_add_f_t_f(7.8125, h_mul_t_f_f(_vecif_353_t, 0.012)))
			sum = h_where_t_n_t_v_t_v(_vecif_353_exp, _vecif_353_sum, sum)
			t = h_where_t_n_t_n_t_n(_vecif_353_exp, _vecif_353_t, t)
	swizzle_set_t_n4_xyz(sum, h_lerp_t_v_t_v_t_n(bgc, h_div_t_vf_t_f(swizzle_t_n4_xyz(sum), h_add_t_f_f(swizzle_t_n4_w(sum), 1.0E-4)), swizzle_t_n4_w(sum)))
	return h_clamp_t_v_n_n(swizzle_t_n4_xyz(sum), 0.0, 1.0)

def raymarchTerrain_f3_f3_f3_f_f_arr(ro, rd, bgc, startdist, dist):
	global iChannel2, s_linear_clamp_sampler, lig
	t = startdist
	sum = glsl_vec4_f(0.0)
	hit = False
	col = bgc
	_br_flag_190 = False
	if True:
		i = 0
		_vecif_208_exp_0 = h_not_n(_br_flag_190)
		if any_ifexp_true_n(_vecif_208_exp_0):
			_vecif_208__br_flag_190 = _br_flag_190
			_vecif_208_t = t
			_vecif_208_hit = hit
			_vecif_209_exp = _vecif_208_hit
			if any_ifexp_true_n(_vecif_209_exp):
				_vecif_209__br_flag_190 = _vecif_208__br_flag_190
				_vecif_209__br_flag_190 = True
				_vecif_208__br_flag_190 = h_where_n_n_n(_vecif_209_exp, _vecif_209__br_flag_190, _vecif_208__br_flag_190)
			_vecif_210_exp = h_not_n(_vecif_208__br_flag_190)
			if any_ifexp_true_n(_vecif_210_exp):
				_vecif_210_t = _vecif_208_t
				_vecif_210_hit = _vecif_208_hit
				_vecif_210_t = h_add_t_f_t_f(_vecif_210_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_210_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_210_t, rd))
				_vecif_211_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_211_exp):
					_vecif_211_hit = _vecif_210_hit
					_vecif_211_hit = True
					_vecif_210_hit = h_where_t_n_n_n(_vecif_211_exp, _vecif_211_hit, _vecif_210_hit)
				else:
					_vecif_210_hit = h_broadcast_t_b_b(ro, _vecif_210_hit)
				_vecif_208_t = h_where_n_t_n_t_n(_vecif_210_exp, _vecif_210_t, _vecif_208_t)
				_vecif_208_hit = h_where_n_t_n_n(_vecif_210_exp, _vecif_210_hit, _vecif_208_hit)
			else:
				_vecif_208_hit = h_broadcast_t_b_b(ro, _vecif_208_hit)
			_br_flag_190 = h_where_n_n_n(_vecif_208_exp_0, _vecif_208__br_flag_190, _br_flag_190)
			t = h_where_n_t_n_t_n(_vecif_208_exp_0, _vecif_208_t, t)
			hit = h_where_n_t_n_n(_vecif_208_exp_0, _vecif_208_hit, hit)
		else:
			hit = h_broadcast_t_b_b(ro, hit)
		i = 1
		_vecif_212_exp_0 = h_not_n(_br_flag_190)
		if any_ifexp_true_n(_vecif_212_exp_0):
			_vecif_212__br_flag_190 = _br_flag_190
			_vecif_212_t = t
			_vecif_212_hit = hit
			_vecif_213_exp = _vecif_212_hit
			if any_ifexp_true_t_n(_vecif_213_exp):
				_vecif_213__br_flag_190 = _vecif_212__br_flag_190
				_vecif_213__br_flag_190 = True
				_vecif_212__br_flag_190 = h_where_t_n_n_n(_vecif_213_exp, _vecif_213__br_flag_190, _vecif_212__br_flag_190)
			else:
				_vecif_212__br_flag_190 = h_broadcast_t_b_b(ro, _vecif_212__br_flag_190)
			_vecif_214_exp = h_not_t_n(_vecif_212__br_flag_190)
			if any_ifexp_true_t_n(_vecif_214_exp):
				_vecif_214_t = _vecif_212_t
				_vecif_214_hit = _vecif_212_hit
				_vecif_214_t = h_add_t_f_t_f(_vecif_214_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_214_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_214_t, rd))
				_vecif_215_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_215_exp):
					_vecif_215_hit = _vecif_214_hit
					_vecif_215_hit = h_broadcast_t_b_b(_vecif_215_hit, True)
					_vecif_214_hit = h_where_t_n_t_n_t_n(_vecif_215_exp, _vecif_215_hit, _vecif_214_hit)
				_vecif_212_t = h_where_t_n_t_n_t_n(_vecif_214_exp, _vecif_214_t, _vecif_212_t)
				_vecif_212_hit = h_where_t_n_t_n_t_n(_vecif_214_exp, _vecif_214_hit, _vecif_212_hit)
			_br_flag_190 = h_where_n_t_n_n(_vecif_212_exp_0, _vecif_212__br_flag_190, _br_flag_190)
			t = h_where_n_t_n_t_n(_vecif_212_exp_0, _vecif_212_t, t)
			hit = h_where_n_t_n_t_n(_vecif_212_exp_0, _vecif_212_hit, hit)
		else:
			_br_flag_190 = h_broadcast_t_b_b(ro, _br_flag_190)
		i = 2
		_vecif_216_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_216_exp_0):
			_vecif_216__br_flag_190 = _br_flag_190
			_vecif_216_t = t
			_vecif_216_hit = hit
			_vecif_217_exp = _vecif_216_hit
			if any_ifexp_true_t_n(_vecif_217_exp):
				_vecif_217__br_flag_190 = _vecif_216__br_flag_190
				_vecif_217__br_flag_190 = h_broadcast_t_b_b(_vecif_217__br_flag_190, True)
				_vecif_216__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_217_exp, _vecif_217__br_flag_190, _vecif_216__br_flag_190)
			_vecif_218_exp = h_not_t_n(_vecif_216__br_flag_190)
			if any_ifexp_true_t_n(_vecif_218_exp):
				_vecif_218_t = _vecif_216_t
				_vecif_218_hit = _vecif_216_hit
				_vecif_218_t = h_add_t_f_t_f(_vecif_218_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_218_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_218_t, rd))
				_vecif_219_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_219_exp):
					_vecif_219_hit = _vecif_218_hit
					_vecif_219_hit = h_broadcast_t_b_b(_vecif_219_hit, True)
					_vecif_218_hit = h_where_t_n_t_n_t_n(_vecif_219_exp, _vecif_219_hit, _vecif_218_hit)
				_vecif_216_t = h_where_t_n_t_n_t_n(_vecif_218_exp, _vecif_218_t, _vecif_216_t)
				_vecif_216_hit = h_where_t_n_t_n_t_n(_vecif_218_exp, _vecif_218_hit, _vecif_216_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_216_exp_0, _vecif_216__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_216_exp_0, _vecif_216_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_216_exp_0, _vecif_216_hit, hit)
		i = 3
		_vecif_220_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_220_exp_0):
			_vecif_220__br_flag_190 = _br_flag_190
			_vecif_220_t = t
			_vecif_220_hit = hit
			_vecif_221_exp = _vecif_220_hit
			if any_ifexp_true_t_n(_vecif_221_exp):
				_vecif_221__br_flag_190 = _vecif_220__br_flag_190
				_vecif_221__br_flag_190 = h_broadcast_t_b_b(_vecif_221__br_flag_190, True)
				_vecif_220__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_221_exp, _vecif_221__br_flag_190, _vecif_220__br_flag_190)
			_vecif_222_exp = h_not_t_n(_vecif_220__br_flag_190)
			if any_ifexp_true_t_n(_vecif_222_exp):
				_vecif_222_t = _vecif_220_t
				_vecif_222_hit = _vecif_220_hit
				_vecif_222_t = h_add_t_f_t_f(_vecif_222_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_222_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_222_t, rd))
				_vecif_223_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_223_exp):
					_vecif_223_hit = _vecif_222_hit
					_vecif_223_hit = h_broadcast_t_b_b(_vecif_223_hit, True)
					_vecif_222_hit = h_where_t_n_t_n_t_n(_vecif_223_exp, _vecif_223_hit, _vecif_222_hit)
				_vecif_220_t = h_where_t_n_t_n_t_n(_vecif_222_exp, _vecif_222_t, _vecif_220_t)
				_vecif_220_hit = h_where_t_n_t_n_t_n(_vecif_222_exp, _vecif_222_hit, _vecif_220_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_220_exp_0, _vecif_220__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_220_exp_0, _vecif_220_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_220_exp_0, _vecif_220_hit, hit)
		i = 4
		_vecif_224_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_224_exp_0):
			_vecif_224__br_flag_190 = _br_flag_190
			_vecif_224_t = t
			_vecif_224_hit = hit
			_vecif_225_exp = _vecif_224_hit
			if any_ifexp_true_t_n(_vecif_225_exp):
				_vecif_225__br_flag_190 = _vecif_224__br_flag_190
				_vecif_225__br_flag_190 = h_broadcast_t_b_b(_vecif_225__br_flag_190, True)
				_vecif_224__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_225_exp, _vecif_225__br_flag_190, _vecif_224__br_flag_190)
			_vecif_226_exp = h_not_t_n(_vecif_224__br_flag_190)
			if any_ifexp_true_t_n(_vecif_226_exp):
				_vecif_226_t = _vecif_224_t
				_vecif_226_hit = _vecif_224_hit
				_vecif_226_t = h_add_t_f_t_f(_vecif_226_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_226_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_226_t, rd))
				_vecif_227_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_227_exp):
					_vecif_227_hit = _vecif_226_hit
					_vecif_227_hit = h_broadcast_t_b_b(_vecif_227_hit, True)
					_vecif_226_hit = h_where_t_n_t_n_t_n(_vecif_227_exp, _vecif_227_hit, _vecif_226_hit)
				_vecif_224_t = h_where_t_n_t_n_t_n(_vecif_226_exp, _vecif_226_t, _vecif_224_t)
				_vecif_224_hit = h_where_t_n_t_n_t_n(_vecif_226_exp, _vecif_226_hit, _vecif_224_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_224_exp_0, _vecif_224__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_224_exp_0, _vecif_224_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_224_exp_0, _vecif_224_hit, hit)
		i = 5
		_vecif_228_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_228_exp_0):
			_vecif_228__br_flag_190 = _br_flag_190
			_vecif_228_t = t
			_vecif_228_hit = hit
			_vecif_229_exp = _vecif_228_hit
			if any_ifexp_true_t_n(_vecif_229_exp):
				_vecif_229__br_flag_190 = _vecif_228__br_flag_190
				_vecif_229__br_flag_190 = h_broadcast_t_b_b(_vecif_229__br_flag_190, True)
				_vecif_228__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_229_exp, _vecif_229__br_flag_190, _vecif_228__br_flag_190)
			_vecif_230_exp = h_not_t_n(_vecif_228__br_flag_190)
			if any_ifexp_true_t_n(_vecif_230_exp):
				_vecif_230_t = _vecif_228_t
				_vecif_230_hit = _vecif_228_hit
				_vecif_230_t = h_add_t_f_t_f(_vecif_230_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_230_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_230_t, rd))
				_vecif_231_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_231_exp):
					_vecif_231_hit = _vecif_230_hit
					_vecif_231_hit = h_broadcast_t_b_b(_vecif_231_hit, True)
					_vecif_230_hit = h_where_t_n_t_n_t_n(_vecif_231_exp, _vecif_231_hit, _vecif_230_hit)
				_vecif_228_t = h_where_t_n_t_n_t_n(_vecif_230_exp, _vecif_230_t, _vecif_228_t)
				_vecif_228_hit = h_where_t_n_t_n_t_n(_vecif_230_exp, _vecif_230_hit, _vecif_228_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_228_exp_0, _vecif_228__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_228_exp_0, _vecif_228_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_228_exp_0, _vecif_228_hit, hit)
		i = 6
		_vecif_232_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_232_exp_0):
			_vecif_232__br_flag_190 = _br_flag_190
			_vecif_232_t = t
			_vecif_232_hit = hit
			_vecif_233_exp = _vecif_232_hit
			if any_ifexp_true_t_n(_vecif_233_exp):
				_vecif_233__br_flag_190 = _vecif_232__br_flag_190
				_vecif_233__br_flag_190 = h_broadcast_t_b_b(_vecif_233__br_flag_190, True)
				_vecif_232__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_233_exp, _vecif_233__br_flag_190, _vecif_232__br_flag_190)
			_vecif_234_exp = h_not_t_n(_vecif_232__br_flag_190)
			if any_ifexp_true_t_n(_vecif_234_exp):
				_vecif_234_t = _vecif_232_t
				_vecif_234_hit = _vecif_232_hit
				_vecif_234_t = h_add_t_f_t_f(_vecif_234_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_234_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_234_t, rd))
				_vecif_235_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_235_exp):
					_vecif_235_hit = _vecif_234_hit
					_vecif_235_hit = h_broadcast_t_b_b(_vecif_235_hit, True)
					_vecif_234_hit = h_where_t_n_t_n_t_n(_vecif_235_exp, _vecif_235_hit, _vecif_234_hit)
				_vecif_232_t = h_where_t_n_t_n_t_n(_vecif_234_exp, _vecif_234_t, _vecif_232_t)
				_vecif_232_hit = h_where_t_n_t_n_t_n(_vecif_234_exp, _vecif_234_hit, _vecif_232_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_232_exp_0, _vecif_232__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_232_exp_0, _vecif_232_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_232_exp_0, _vecif_232_hit, hit)
		i = 7
		_vecif_236_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_236_exp_0):
			_vecif_236__br_flag_190 = _br_flag_190
			_vecif_236_t = t
			_vecif_236_hit = hit
			_vecif_237_exp = _vecif_236_hit
			if any_ifexp_true_t_n(_vecif_237_exp):
				_vecif_237__br_flag_190 = _vecif_236__br_flag_190
				_vecif_237__br_flag_190 = h_broadcast_t_b_b(_vecif_237__br_flag_190, True)
				_vecif_236__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_237_exp, _vecif_237__br_flag_190, _vecif_236__br_flag_190)
			_vecif_238_exp = h_not_t_n(_vecif_236__br_flag_190)
			if any_ifexp_true_t_n(_vecif_238_exp):
				_vecif_238_t = _vecif_236_t
				_vecif_238_hit = _vecif_236_hit
				_vecif_238_t = h_add_t_f_t_f(_vecif_238_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_238_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_238_t, rd))
				_vecif_239_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_239_exp):
					_vecif_239_hit = _vecif_238_hit
					_vecif_239_hit = h_broadcast_t_b_b(_vecif_239_hit, True)
					_vecif_238_hit = h_where_t_n_t_n_t_n(_vecif_239_exp, _vecif_239_hit, _vecif_238_hit)
				_vecif_236_t = h_where_t_n_t_n_t_n(_vecif_238_exp, _vecif_238_t, _vecif_236_t)
				_vecif_236_hit = h_where_t_n_t_n_t_n(_vecif_238_exp, _vecif_238_hit, _vecif_236_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_236_exp_0, _vecif_236__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_236_exp_0, _vecif_236_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_236_exp_0, _vecif_236_hit, hit)
		i = 8
		_vecif_240_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_240_exp_0):
			_vecif_240__br_flag_190 = _br_flag_190
			_vecif_240_t = t
			_vecif_240_hit = hit
			_vecif_241_exp = _vecif_240_hit
			if any_ifexp_true_t_n(_vecif_241_exp):
				_vecif_241__br_flag_190 = _vecif_240__br_flag_190
				_vecif_241__br_flag_190 = h_broadcast_t_b_b(_vecif_241__br_flag_190, True)
				_vecif_240__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_241_exp, _vecif_241__br_flag_190, _vecif_240__br_flag_190)
			_vecif_242_exp = h_not_t_n(_vecif_240__br_flag_190)
			if any_ifexp_true_t_n(_vecif_242_exp):
				_vecif_242_t = _vecif_240_t
				_vecif_242_hit = _vecif_240_hit
				_vecif_242_t = h_add_t_f_t_f(_vecif_242_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_242_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_242_t, rd))
				_vecif_243_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_243_exp):
					_vecif_243_hit = _vecif_242_hit
					_vecif_243_hit = h_broadcast_t_b_b(_vecif_243_hit, True)
					_vecif_242_hit = h_where_t_n_t_n_t_n(_vecif_243_exp, _vecif_243_hit, _vecif_242_hit)
				_vecif_240_t = h_where_t_n_t_n_t_n(_vecif_242_exp, _vecif_242_t, _vecif_240_t)
				_vecif_240_hit = h_where_t_n_t_n_t_n(_vecif_242_exp, _vecif_242_hit, _vecif_240_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_240_exp_0, _vecif_240__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_240_exp_0, _vecif_240_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_240_exp_0, _vecif_240_hit, hit)
		i = 9
		_vecif_244_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_244_exp_0):
			_vecif_244__br_flag_190 = _br_flag_190
			_vecif_244_t = t
			_vecif_244_hit = hit
			_vecif_245_exp = _vecif_244_hit
			if any_ifexp_true_t_n(_vecif_245_exp):
				_vecif_245__br_flag_190 = _vecif_244__br_flag_190
				_vecif_245__br_flag_190 = h_broadcast_t_b_b(_vecif_245__br_flag_190, True)
				_vecif_244__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_245_exp, _vecif_245__br_flag_190, _vecif_244__br_flag_190)
			_vecif_246_exp = h_not_t_n(_vecif_244__br_flag_190)
			if any_ifexp_true_t_n(_vecif_246_exp):
				_vecif_246_t = _vecif_244_t
				_vecif_246_hit = _vecif_244_hit
				_vecif_246_t = h_add_t_f_t_f(_vecif_246_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_246_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_246_t, rd))
				_vecif_247_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_247_exp):
					_vecif_247_hit = _vecif_246_hit
					_vecif_247_hit = h_broadcast_t_b_b(_vecif_247_hit, True)
					_vecif_246_hit = h_where_t_n_t_n_t_n(_vecif_247_exp, _vecif_247_hit, _vecif_246_hit)
				_vecif_244_t = h_where_t_n_t_n_t_n(_vecif_246_exp, _vecif_246_t, _vecif_244_t)
				_vecif_244_hit = h_where_t_n_t_n_t_n(_vecif_246_exp, _vecif_246_hit, _vecif_244_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_244_exp_0, _vecif_244__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_244_exp_0, _vecif_244_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_244_exp_0, _vecif_244_hit, hit)
		i = 10
		_vecif_248_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_248_exp_0):
			_vecif_248__br_flag_190 = _br_flag_190
			_vecif_248_t = t
			_vecif_248_hit = hit
			_vecif_249_exp = _vecif_248_hit
			if any_ifexp_true_t_n(_vecif_249_exp):
				_vecif_249__br_flag_190 = _vecif_248__br_flag_190
				_vecif_249__br_flag_190 = h_broadcast_t_b_b(_vecif_249__br_flag_190, True)
				_vecif_248__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_249_exp, _vecif_249__br_flag_190, _vecif_248__br_flag_190)
			_vecif_250_exp = h_not_t_n(_vecif_248__br_flag_190)
			if any_ifexp_true_t_n(_vecif_250_exp):
				_vecif_250_t = _vecif_248_t
				_vecif_250_hit = _vecif_248_hit
				_vecif_250_t = h_add_t_f_t_f(_vecif_250_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_250_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_250_t, rd))
				_vecif_251_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_251_exp):
					_vecif_251_hit = _vecif_250_hit
					_vecif_251_hit = h_broadcast_t_b_b(_vecif_251_hit, True)
					_vecif_250_hit = h_where_t_n_t_n_t_n(_vecif_251_exp, _vecif_251_hit, _vecif_250_hit)
				_vecif_248_t = h_where_t_n_t_n_t_n(_vecif_250_exp, _vecif_250_t, _vecif_248_t)
				_vecif_248_hit = h_where_t_n_t_n_t_n(_vecif_250_exp, _vecif_250_hit, _vecif_248_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_248_exp_0, _vecif_248__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_248_exp_0, _vecif_248_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_248_exp_0, _vecif_248_hit, hit)
		i = 11
		_vecif_252_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_252_exp_0):
			_vecif_252__br_flag_190 = _br_flag_190
			_vecif_252_t = t
			_vecif_252_hit = hit
			_vecif_253_exp = _vecif_252_hit
			if any_ifexp_true_t_n(_vecif_253_exp):
				_vecif_253__br_flag_190 = _vecif_252__br_flag_190
				_vecif_253__br_flag_190 = h_broadcast_t_b_b(_vecif_253__br_flag_190, True)
				_vecif_252__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_253_exp, _vecif_253__br_flag_190, _vecif_252__br_flag_190)
			_vecif_254_exp = h_not_t_n(_vecif_252__br_flag_190)
			if any_ifexp_true_t_n(_vecif_254_exp):
				_vecif_254_t = _vecif_252_t
				_vecif_254_hit = _vecif_252_hit
				_vecif_254_t = h_add_t_f_t_f(_vecif_254_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_254_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_254_t, rd))
				_vecif_255_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_255_exp):
					_vecif_255_hit = _vecif_254_hit
					_vecif_255_hit = h_broadcast_t_b_b(_vecif_255_hit, True)
					_vecif_254_hit = h_where_t_n_t_n_t_n(_vecif_255_exp, _vecif_255_hit, _vecif_254_hit)
				_vecif_252_t = h_where_t_n_t_n_t_n(_vecif_254_exp, _vecif_254_t, _vecif_252_t)
				_vecif_252_hit = h_where_t_n_t_n_t_n(_vecif_254_exp, _vecif_254_hit, _vecif_252_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_252_exp_0, _vecif_252__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_252_exp_0, _vecif_252_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_252_exp_0, _vecif_252_hit, hit)
		i = 12
		_vecif_256_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_256_exp_0):
			_vecif_256__br_flag_190 = _br_flag_190
			_vecif_256_t = t
			_vecif_256_hit = hit
			_vecif_257_exp = _vecif_256_hit
			if any_ifexp_true_t_n(_vecif_257_exp):
				_vecif_257__br_flag_190 = _vecif_256__br_flag_190
				_vecif_257__br_flag_190 = h_broadcast_t_b_b(_vecif_257__br_flag_190, True)
				_vecif_256__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_257_exp, _vecif_257__br_flag_190, _vecif_256__br_flag_190)
			_vecif_258_exp = h_not_t_n(_vecif_256__br_flag_190)
			if any_ifexp_true_t_n(_vecif_258_exp):
				_vecif_258_t = _vecif_256_t
				_vecif_258_hit = _vecif_256_hit
				_vecif_258_t = h_add_t_f_t_f(_vecif_258_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_258_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_258_t, rd))
				_vecif_259_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_259_exp):
					_vecif_259_hit = _vecif_258_hit
					_vecif_259_hit = h_broadcast_t_b_b(_vecif_259_hit, True)
					_vecif_258_hit = h_where_t_n_t_n_t_n(_vecif_259_exp, _vecif_259_hit, _vecif_258_hit)
				_vecif_256_t = h_where_t_n_t_n_t_n(_vecif_258_exp, _vecif_258_t, _vecif_256_t)
				_vecif_256_hit = h_where_t_n_t_n_t_n(_vecif_258_exp, _vecif_258_hit, _vecif_256_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_256_exp_0, _vecif_256__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_256_exp_0, _vecif_256_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_256_exp_0, _vecif_256_hit, hit)
		i = 13
		_vecif_260_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_260_exp_0):
			_vecif_260__br_flag_190 = _br_flag_190
			_vecif_260_t = t
			_vecif_260_hit = hit
			_vecif_261_exp = _vecif_260_hit
			if any_ifexp_true_t_n(_vecif_261_exp):
				_vecif_261__br_flag_190 = _vecif_260__br_flag_190
				_vecif_261__br_flag_190 = h_broadcast_t_b_b(_vecif_261__br_flag_190, True)
				_vecif_260__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_261_exp, _vecif_261__br_flag_190, _vecif_260__br_flag_190)
			_vecif_262_exp = h_not_t_n(_vecif_260__br_flag_190)
			if any_ifexp_true_t_n(_vecif_262_exp):
				_vecif_262_t = _vecif_260_t
				_vecif_262_hit = _vecif_260_hit
				_vecif_262_t = h_add_t_f_t_f(_vecif_262_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_262_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_262_t, rd))
				_vecif_263_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_263_exp):
					_vecif_263_hit = _vecif_262_hit
					_vecif_263_hit = h_broadcast_t_b_b(_vecif_263_hit, True)
					_vecif_262_hit = h_where_t_n_t_n_t_n(_vecif_263_exp, _vecif_263_hit, _vecif_262_hit)
				_vecif_260_t = h_where_t_n_t_n_t_n(_vecif_262_exp, _vecif_262_t, _vecif_260_t)
				_vecif_260_hit = h_where_t_n_t_n_t_n(_vecif_262_exp, _vecif_262_hit, _vecif_260_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_260_exp_0, _vecif_260__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_260_exp_0, _vecif_260_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_260_exp_0, _vecif_260_hit, hit)
		i = 14
		_vecif_264_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_264_exp_0):
			_vecif_264__br_flag_190 = _br_flag_190
			_vecif_264_t = t
			_vecif_264_hit = hit
			_vecif_265_exp = _vecif_264_hit
			if any_ifexp_true_t_n(_vecif_265_exp):
				_vecif_265__br_flag_190 = _vecif_264__br_flag_190
				_vecif_265__br_flag_190 = h_broadcast_t_b_b(_vecif_265__br_flag_190, True)
				_vecif_264__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_265_exp, _vecif_265__br_flag_190, _vecif_264__br_flag_190)
			_vecif_266_exp = h_not_t_n(_vecif_264__br_flag_190)
			if any_ifexp_true_t_n(_vecif_266_exp):
				_vecif_266_t = _vecif_264_t
				_vecif_266_hit = _vecif_264_hit
				_vecif_266_t = h_add_t_f_t_f(_vecif_266_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_266_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_266_t, rd))
				_vecif_267_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_267_exp):
					_vecif_267_hit = _vecif_266_hit
					_vecif_267_hit = h_broadcast_t_b_b(_vecif_267_hit, True)
					_vecif_266_hit = h_where_t_n_t_n_t_n(_vecif_267_exp, _vecif_267_hit, _vecif_266_hit)
				_vecif_264_t = h_where_t_n_t_n_t_n(_vecif_266_exp, _vecif_266_t, _vecif_264_t)
				_vecif_264_hit = h_where_t_n_t_n_t_n(_vecif_266_exp, _vecif_266_hit, _vecif_264_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_264_exp_0, _vecif_264__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_264_exp_0, _vecif_264_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_264_exp_0, _vecif_264_hit, hit)
		i = 15
		_vecif_268_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_268_exp_0):
			_vecif_268__br_flag_190 = _br_flag_190
			_vecif_268_t = t
			_vecif_268_hit = hit
			_vecif_269_exp = _vecif_268_hit
			if any_ifexp_true_t_n(_vecif_269_exp):
				_vecif_269__br_flag_190 = _vecif_268__br_flag_190
				_vecif_269__br_flag_190 = h_broadcast_t_b_b(_vecif_269__br_flag_190, True)
				_vecif_268__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_269_exp, _vecif_269__br_flag_190, _vecif_268__br_flag_190)
			_vecif_270_exp = h_not_t_n(_vecif_268__br_flag_190)
			if any_ifexp_true_t_n(_vecif_270_exp):
				_vecif_270_t = _vecif_268_t
				_vecif_270_hit = _vecif_268_hit
				_vecif_270_t = h_add_t_f_t_f(_vecif_270_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_270_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_270_t, rd))
				_vecif_271_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_271_exp):
					_vecif_271_hit = _vecif_270_hit
					_vecif_271_hit = h_broadcast_t_b_b(_vecif_271_hit, True)
					_vecif_270_hit = h_where_t_n_t_n_t_n(_vecif_271_exp, _vecif_271_hit, _vecif_270_hit)
				_vecif_268_t = h_where_t_n_t_n_t_n(_vecif_270_exp, _vecif_270_t, _vecif_268_t)
				_vecif_268_hit = h_where_t_n_t_n_t_n(_vecif_270_exp, _vecif_270_hit, _vecif_268_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_268_exp_0, _vecif_268__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_268_exp_0, _vecif_268_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_268_exp_0, _vecif_268_hit, hit)
		i = 16
		_vecif_272_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_272_exp_0):
			_vecif_272__br_flag_190 = _br_flag_190
			_vecif_272_t = t
			_vecif_272_hit = hit
			_vecif_273_exp = _vecif_272_hit
			if any_ifexp_true_t_n(_vecif_273_exp):
				_vecif_273__br_flag_190 = _vecif_272__br_flag_190
				_vecif_273__br_flag_190 = h_broadcast_t_b_b(_vecif_273__br_flag_190, True)
				_vecif_272__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_273_exp, _vecif_273__br_flag_190, _vecif_272__br_flag_190)
			_vecif_274_exp = h_not_t_n(_vecif_272__br_flag_190)
			if any_ifexp_true_t_n(_vecif_274_exp):
				_vecif_274_t = _vecif_272_t
				_vecif_274_hit = _vecif_272_hit
				_vecif_274_t = h_add_t_f_t_f(_vecif_274_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_274_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_274_t, rd))
				_vecif_275_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_275_exp):
					_vecif_275_hit = _vecif_274_hit
					_vecif_275_hit = h_broadcast_t_b_b(_vecif_275_hit, True)
					_vecif_274_hit = h_where_t_n_t_n_t_n(_vecif_275_exp, _vecif_275_hit, _vecif_274_hit)
				_vecif_272_t = h_where_t_n_t_n_t_n(_vecif_274_exp, _vecif_274_t, _vecif_272_t)
				_vecif_272_hit = h_where_t_n_t_n_t_n(_vecif_274_exp, _vecif_274_hit, _vecif_272_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_272_exp_0, _vecif_272__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_272_exp_0, _vecif_272_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_272_exp_0, _vecif_272_hit, hit)
		i = 17
		_vecif_276_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_276_exp_0):
			_vecif_276__br_flag_190 = _br_flag_190
			_vecif_276_t = t
			_vecif_276_hit = hit
			_vecif_277_exp = _vecif_276_hit
			if any_ifexp_true_t_n(_vecif_277_exp):
				_vecif_277__br_flag_190 = _vecif_276__br_flag_190
				_vecif_277__br_flag_190 = h_broadcast_t_b_b(_vecif_277__br_flag_190, True)
				_vecif_276__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_277_exp, _vecif_277__br_flag_190, _vecif_276__br_flag_190)
			_vecif_278_exp = h_not_t_n(_vecif_276__br_flag_190)
			if any_ifexp_true_t_n(_vecif_278_exp):
				_vecif_278_t = _vecif_276_t
				_vecif_278_hit = _vecif_276_hit
				_vecif_278_t = h_add_t_f_t_f(_vecif_278_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_278_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_278_t, rd))
				_vecif_279_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_279_exp):
					_vecif_279_hit = _vecif_278_hit
					_vecif_279_hit = h_broadcast_t_b_b(_vecif_279_hit, True)
					_vecif_278_hit = h_where_t_n_t_n_t_n(_vecif_279_exp, _vecif_279_hit, _vecif_278_hit)
				_vecif_276_t = h_where_t_n_t_n_t_n(_vecif_278_exp, _vecif_278_t, _vecif_276_t)
				_vecif_276_hit = h_where_t_n_t_n_t_n(_vecif_278_exp, _vecif_278_hit, _vecif_276_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_276_exp_0, _vecif_276__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_276_exp_0, _vecif_276_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_276_exp_0, _vecif_276_hit, hit)
		i = 18
		_vecif_280_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_280_exp_0):
			_vecif_280__br_flag_190 = _br_flag_190
			_vecif_280_t = t
			_vecif_280_hit = hit
			_vecif_281_exp = _vecif_280_hit
			if any_ifexp_true_t_n(_vecif_281_exp):
				_vecif_281__br_flag_190 = _vecif_280__br_flag_190
				_vecif_281__br_flag_190 = h_broadcast_t_b_b(_vecif_281__br_flag_190, True)
				_vecif_280__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_281_exp, _vecif_281__br_flag_190, _vecif_280__br_flag_190)
			_vecif_282_exp = h_not_t_n(_vecif_280__br_flag_190)
			if any_ifexp_true_t_n(_vecif_282_exp):
				_vecif_282_t = _vecif_280_t
				_vecif_282_hit = _vecif_280_hit
				_vecif_282_t = h_add_t_f_t_f(_vecif_282_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_282_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_282_t, rd))
				_vecif_283_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_283_exp):
					_vecif_283_hit = _vecif_282_hit
					_vecif_283_hit = h_broadcast_t_b_b(_vecif_283_hit, True)
					_vecif_282_hit = h_where_t_n_t_n_t_n(_vecif_283_exp, _vecif_283_hit, _vecif_282_hit)
				_vecif_280_t = h_where_t_n_t_n_t_n(_vecif_282_exp, _vecif_282_t, _vecif_280_t)
				_vecif_280_hit = h_where_t_n_t_n_t_n(_vecif_282_exp, _vecif_282_hit, _vecif_280_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_280_exp_0, _vecif_280__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_280_exp_0, _vecif_280_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_280_exp_0, _vecif_280_hit, hit)
		i = 19
		_vecif_284_exp_0 = h_not_t_n(_br_flag_190)
		if any_ifexp_true_t_n(_vecif_284_exp_0):
			_vecif_284__br_flag_190 = _br_flag_190
			_vecif_284_t = t
			_vecif_284_hit = hit
			_vecif_285_exp = _vecif_284_hit
			if any_ifexp_true_t_n(_vecif_285_exp):
				_vecif_285__br_flag_190 = _vecif_284__br_flag_190
				_vecif_285__br_flag_190 = h_broadcast_t_b_b(_vecif_285__br_flag_190, True)
				_vecif_284__br_flag_190 = h_where_t_n_t_n_t_n(_vecif_285_exp, _vecif_285__br_flag_190, _vecif_284__br_flag_190)
			_vecif_286_exp = h_not_t_n(_vecif_284__br_flag_190)
			if any_ifexp_true_t_n(_vecif_286_exp):
				_vecif_286_t = _vecif_284_t
				_vecif_286_hit = _vecif_284_hit
				_vecif_286_t = h_add_t_f_t_f(_vecif_286_t, h_add_f_t_f(8.0, h_div_t_f_f(_vecif_286_t, 300.0)))
				pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_286_t, rd))
				_vecif_287_exp = h_less_than_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))
				if any_ifexp_true_t_n(_vecif_287_exp):
					_vecif_287_hit = _vecif_286_hit
					_vecif_287_hit = h_broadcast_t_b_b(_vecif_287_hit, True)
					_vecif_286_hit = h_where_t_n_t_n_t_n(_vecif_287_exp, _vecif_287_hit, _vecif_286_hit)
				_vecif_284_t = h_where_t_n_t_n_t_n(_vecif_286_exp, _vecif_286_t, _vecif_284_t)
				_vecif_284_hit = h_where_t_n_t_n_t_n(_vecif_286_exp, _vecif_286_hit, _vecif_284_hit)
			_br_flag_190 = h_where_t_n_t_n_t_n(_vecif_284_exp_0, _vecif_284__br_flag_190, _br_flag_190)
			t = h_where_t_n_t_n_t_n(_vecif_284_exp_0, _vecif_284_t, t)
			hit = h_where_t_n_t_n_t_n(_vecif_284_exp_0, _vecif_284_hit, hit)
	_vecif_288_exp = hit
	if any_ifexp_true_t_n(_vecif_288_exp):
		_vecif_288_t = t
		_vecif_288_col = col
		_vecif_288_dist = dist
		dt = h_add_f_t_f(4.0, h_div_t_f_f(_vecif_288_t, 400.0))
		_vecif_288_t = h_sub_t_f_t_f(_vecif_288_t, dt)
		pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_288_t, rd))
		_vecif_288_t = h_add_t_f_t_f(_vecif_288_t, h_mul_t_f_t_f(h_sub_f_t_f(0.5, h_step_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))), dt))
		if True:
			j = 0
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_288_t, rd))
			dt = h_mul_t_f_f(dt, 0.5)
			_vecif_288_t = h_add_t_f_t_f(_vecif_288_t, h_mul_t_f_t_f(h_sub_f_t_f(0.5, h_step_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))), dt))
			j = 1
			pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_288_t, rd))
			dt = h_mul_t_f_f(dt, 0.5)
			_vecif_288_t = h_add_t_f_t_f(_vecif_288_t, h_mul_t_f_t_f(h_sub_f_t_f(0.5, h_step_t_n_t_n(swizzle_t_n3_y(pos), terrainMap_f3_arr(pos))), dt))
		pos = h_add_t_vf_t_vf(ro, h_mul_t_f_t_vf(_vecif_288_t, rd))
		dx = h_f3_n_n_n(10.00, 0.0, 0.0)
		dz = h_f3_n_n_n(0.0, 0.0, 10.00)
		normal = h_f3_n_n_n(0.0, 0.0, 0.0)
		_, normal = swizzle_set_and_broadcast_n3_x(normal, h_div_t_f_f(h_sub_t_f_t_f(terrainMap_f3_arr(h_add_t_vf_vf(pos, dx)), terrainMap_f3_arr(h_sub_t_vf_vf(pos, dx))), 20.00))
		swizzle_set_t_n3_z(normal, h_div_t_f_f(h_sub_t_f_t_f(terrainMap_f3_arr(h_add_t_vf_vf(pos, dz)), terrainMap_f3_arr(h_sub_t_vf_vf(pos, dz))), 20.00))
		swizzle_set_t_n3_y(normal, h_broadcast_t_f_f(swizzle_t_n3_y(normal), 1.0))
		normal = h_normalize_t_v(normal)
		_vecif_288_col = h_add_vf_t_vf(glsl_vec3_f(0.20000000000000001), h_mul_t_vf_vf(h_mul_f_t_vf(0.69999999999999996, swizzle_t_n4_xyz(Texture2D_Sample_n_t_v(iChannel2, s_linear_clamp_sampler, h_mul_t_vf_f(swizzle_t_n3_xz(pos), 0.01)))), h_f3_n_n_n(1.0, 0.90000000000000002, 0.59999999999999998)))
		veg = h_add_t_f_t_f(h_mul_f_t_f(0.29999999999999999, fbm_f3_arr(h_mul_t_vf_f(pos, 0.20000000000000001))), swizzle_t_n3_y(normal))
		_vecif_289_exp_0 = h_greater_than_t_n_n(veg, 0.75)
		if any_ifexp_true_t_n(_vecif_289_exp_0):
			_vecif_289_col = _vecif_288_col
			_vecif_289_col = h_mul_t_vf_f(h_mul_vf_t_f(h_f3_n_n_n(0.45000000000000001, 0.59999999999999998, 0.29999999999999999), h_add_f_t_f(0.5, h_mul_f_t_f(0.5, fbm_f3_arr(h_mul_t_vf_f(pos, 0.5))))), 0.59999999999999998)
			_vecif_288_col = h_where_t_n_t_v_t_v(_vecif_289_exp_0, _vecif_289_col, _vecif_288_col)
		_vecif_289_exp_1 = h_greater_than_t_n_n(veg, 0.66000000000000003)
		if any_ifexp_true_t_n(_vecif_289_exp_1):
			_vecif_289_col = _vecif_288_col
			_vecif_289_col = h_add_t_vf_t_vf(h_mul_t_vf_f(_vecif_289_col, 0.59999999999999998), h_mul_t_vf_f(h_mul_vf_t_f(h_f3_n_n_n(0.40000000000000002, 0.5, 0.29999999999999999), h_add_f_t_f(0.5, h_mul_f_t_f(0.5, fbm_f3_arr(h_mul_t_vf_f(pos, 0.25))))), 0.29999999999999999))
			_vecif_288_col = h_where_t_n_t_v_t_v(_vecif_289_exp_1, _vecif_289_col, _vecif_288_col)
		_vecif_288_col = h_mul_t_vf_vf(_vecif_288_col, h_mul_vf_vf(h_f3_n_n_n(0.5, 0.52000000000000002, 0.65000000000000002), h_f3_n_n_n(1.0, 0.90000000000000002, 0.80000000000000004)))
		brdf = _vecif_288_col
		diff = h_clamp_t_n_n_n(h_dot_t_v_v(normal, h_sub_vf(lig)), 0.0, 1.0)
		_vecif_288_col = h_mul_t_vf_vf(h_mul_t_vf_t_f(brdf, diff), h_f3_n_n_n(1.0, 0.59999999999999998, 0.10000000000000001))
		_vecif_288_col = h_add_t_vf_t_vf(_vecif_288_col, h_mul_t_vf_f(h_mul_t_vf_vf(h_mul_t_vf_t_f(brdf, h_clamp_t_n_n_n(h_dot_t_v_v(normal, lig), 0.0, 1.0)), h_f3_n_n_n(0.80000000000000004, 0.59999999999999998, 0.5)), 0.80000000000000004))
		_vecif_288_col = h_add_t_vf_t_vf(_vecif_288_col, h_mul_t_vf_f(h_mul_t_vf_vf(h_mul_t_vf_t_f(brdf, h_clamp_t_n_n_n(h_dot_t_v_v(normal, h_f3_n_n_n(0.0, 1.0, 0.0)), 0.0, 1.0)), h_f3_n_n_n(0.80000000000000004, 0.80000000000000004, 1.0)), 0.20000000000000001))
		_vecif_288_dist = _vecif_288_t
		_vecif_288_t = h_sub_t_f_t_f(_vecif_288_t, h_mul_t_f_f(swizzle_t_n3_y(pos), 3.5))
		_vecif_288_col = h_lerp_t_v_t_v_t_n(_vecif_288_col, bgc, h_sub_f_t_f(1.0, h_exp_t_n(h_mul_t_f_t_f(h_mul_f_t_f(-4.9999999999999998E-7, _vecif_288_t), _vecif_288_t))))
		t = h_where_t_n_t_n_t_n(_vecif_288_exp, _vecif_288_t, t)
		col = h_where_t_n_t_v_t_v(_vecif_288_exp, _vecif_288_col, col)
		dist = h_where_t_n_t_n_n(_vecif_288_exp, _vecif_288_dist, dist)
	else:
		dist = h_broadcast_t_f_f(ro, dist)
	return col, dist

def waterMap_f2_arr(pos):
	global m2, iTime
	posm = h_matmul_f2x2_t_f2(m2, pos)
	return h_mul_t_f_f(h_abs_t_n(h_sub_t_f_f(fbm_f3_arr(h_t_f3_t_n2_n(h_mul_f_t_vf(8.0, posm), h_add_f_f(iTime, 285.0))), 0.5)), 0.10000000000000001)

def intersectPlane_f3_f3_f_f_arr(ro, rd, height, dist):
	_func_ret_val_183 = False
	_func_ret_flag_183 = False
	_vecif_205_exp = h_equal_t_n_n(swizzle_t_n3_y(rd), 0.0)
	if any_ifexp_true_t_n(_vecif_205_exp):
		_vecif_205__func_ret_flag_183 = _func_ret_flag_183
		_vecif_205__func_ret_val_183 = _func_ret_val_183
		_vecif_205__func_ret_flag_183 = True
		_vecif_205__func_ret_val_183 = False
		_func_ret_flag_183 = h_where_t_n_n_n(_vecif_205_exp, _vecif_205__func_ret_flag_183, _func_ret_flag_183)
		_func_ret_val_183 = h_where_t_n_n_n(_vecif_205_exp, _vecif_205__func_ret_val_183, _func_ret_val_183)
	else:
		_func_ret_flag_183 = h_broadcast_t_b_b(rd, _func_ret_flag_183)
		_func_ret_val_183 = h_broadcast_t_b_b(rd, _func_ret_val_183)
	_vecif_206_exp = h_not_t_n(_func_ret_flag_183)
	if any_ifexp_true_t_n(_vecif_206_exp):
		_vecif_206_dist = dist
		_vecif_206__func_ret_flag_183 = _func_ret_flag_183
		_vecif_206__func_ret_val_183 = _func_ret_val_183
		d = h_div_f_t_f(h_sub_f(h_sub_f_f(swizzle_n3_y(ro), height)), swizzle_t_n3_y(rd))
		d = h_min_n_t_n(1.0E+5, d)
		_vecif_207_exp_0 = h_and_t_n_t_n(h_greater_than_t_n_n(d, 0.0), h_less_than_t_n_n(d, _vecif_206_dist))
		if any_ifexp_true_t_n(_vecif_207_exp_0):
			_vecif_207_dist = _vecif_206_dist
			_vecif_207__func_ret_flag_183 = _vecif_206__func_ret_flag_183
			_vecif_207__func_ret_val_183 = _vecif_206__func_ret_val_183
			_vecif_207_dist = d
			_vecif_207__func_ret_flag_183 = h_broadcast_t_b_b(_vecif_207__func_ret_flag_183, True)
			_vecif_207__func_ret_val_183 = h_broadcast_t_b_b(_vecif_207__func_ret_val_183, True)
			_vecif_206_dist = h_where_t_n_t_n_n(_vecif_207_exp_0, _vecif_207_dist, _vecif_206_dist)
			_vecif_206__func_ret_flag_183 = h_where_t_n_t_n_t_n(_vecif_207_exp_0, _vecif_207__func_ret_flag_183, _vecif_206__func_ret_flag_183)
			_vecif_206__func_ret_val_183 = h_where_t_n_t_n_t_n(_vecif_207_exp_0, _vecif_207__func_ret_val_183, _vecif_206__func_ret_val_183)
		else:
			_vecif_206_dist = h_broadcast_t_f_f(rd, _vecif_206_dist)
		if not_all_ifexp_true_t_n(_vecif_207_exp_0):
			_vecif_207__func_ret_flag_183 = _vecif_206__func_ret_flag_183
			_vecif_207__func_ret_val_183 = _vecif_206__func_ret_val_183
			_vecif_207__func_ret_flag_183 = h_broadcast_t_b_b(_vecif_207__func_ret_flag_183, True)
			_vecif_207__func_ret_val_183 = h_broadcast_t_b_b(_vecif_207__func_ret_val_183, False)
			#condition: not _vecif_207_exp_0
			_vecif_206__func_ret_flag_183 = h_where_t_n_t_n_t_n(_vecif_207_exp_0, _vecif_206__func_ret_flag_183, _vecif_207__func_ret_flag_183)
			#condition: not _vecif_207_exp_0
			_vecif_206__func_ret_val_183 = h_where_t_n_t_n_t_n(_vecif_207_exp_0, _vecif_206__func_ret_val_183, _vecif_207__func_ret_val_183)
		dist = h_where_t_n_t_n_n(_vecif_206_exp, _vecif_206_dist, dist)
		_func_ret_flag_183 = h_where_t_n_t_n_t_n(_vecif_206_exp, _vecif_206__func_ret_flag_183, _func_ret_flag_183)
		_func_ret_val_183 = h_where_t_n_t_n_t_n(_vecif_206_exp, _vecif_206__func_ret_val_183, _func_ret_val_183)
	else:
		dist = h_broadcast_t_f_f(rd, dist)
	return _func_ret_val_183, dist

def bgColor_f3_arr(rd):
	global lig
	sun = h_clamp_t_n_n_n(h_dot_v_t_v(lig, rd), 0.0, 1.0)
	col = h_add_t_vf_f(h_sub_vf_t_vf(h_f3_n_n_n(0.5, 0.52000000000000002, 0.55000000000000004), h_mul_t_f_vf(h_mul_t_f_f(swizzle_t_n3_y(rd), 0.20000000000000001), h_f3_n_n_n(1.0, 0.80000000000000004, 1.0))), 0.1125)
	col = h_add_t_vf_t_vf(col, h_mul_vf_t_f(h_f3_n_n_n(1.0, 0.59999999999999998, 0.10000000000000001), h_pow_t_n_n(sun, 8.0)))
	col = h_mul_t_vf_f(col, 0.94999999999999996)
	return col

def mainImage_f4_f2_arr(fragColor, fragCoord):
	global iResolution, iMouse, iTime, iChannel2, s_linear_clamp_sampler
	q = h_div_t_vf_vf(swizzle_t_n2_xy(fragCoord), swizzle_n3_xy(iResolution))
	p = h_add_f_t_vf(-1.0, h_mul_f_t_vf(2.0, q))
	swizzle_set_t_n2_x(p, h_mul_t_f_f(swizzle_t_n2_x(p), h_div_f_f(swizzle_n3_x(iResolution), swizzle_n3_y(iResolution))))
	ro = h_f3_n_n_n(0.0, 0.5, 0.0)
	ta = h_f3_n_n_n(0.0, 0.45000000000000001, 1.0)
	_vecif_199_exp = h_greater_equal_than_n_n(swizzle_n4_z(iMouse), 1.0)
	if any_ifexp_true_n(_vecif_199_exp):
		_vecif_199_ta = h_copy_f3(ta)
		swizzle_set_n3_xz(_vecif_199_ta, h_matmul_f2x2_f2(rot_f(h_mul_f_f(h_sub_f_f(h_div_f_f(swizzle_n4_x(iMouse), swizzle_n3_x(iResolution)), 0.5), 7.0)), swizzle_n3_xz(_vecif_199_ta)))
		ta = h_where_n_v_v(_vecif_199_exp, _vecif_199_ta, ta)
	swizzle_set_n3_xz(ta, h_matmul_f2x2_f2(rot_f(h_sub_f_f(h_mul_f_f(iTime, 0.050000000000000003), h_mul_f_f(h_floor_n(h_div_f_f(h_mul_f_f(iTime, 0.050000000000000003), 6.2831852000000001)), 6.2831852000000001))), swizzle_n3_xz(ta)))
	ww = h_normalize_v(h_sub_vf_vf(ta, ro))
	uu = h_normalize_v(h_cross_v_v(h_f3_n_n_n(0.0, 1.0, 0.0), ww))
	vv = h_normalize_v(h_cross_v_v(ww, uu))
	rd = h_normalize_t_v(h_add_t_vf_vf(h_add_t_vf_t_vf(h_mul_t_f_vf(swizzle_t_n2_x(p), uu), h_mul_t_f_vf(swizzle_t_n2_y(p), vv)), h_mul_f_vf(2.5, ww)))
	fresnel = 0.0
	refldist = 5000.0
	maxdist = 5000.0
	reflected = False
	normal = h_f3_defval()
	col = bgColor_f3_arr(rd)
	roo = ro
	rdo = rd
	bgc = col
	_vecif_200_exp = h_and_t_n_t_n(tuple_get_retval((_call_ret_201 := intersectPlane_f3_f3_f_f_arr(ro, rd, 0.0, refldist), (refldist := tuple_get_outparam(_call_ret_201, 1)))), h_less_than_t_n_n(refldist, 200.0))
	if any_ifexp_true_t_n(_vecif_200_exp):
		_vecif_200_normal = h_copy_f3(normal)
		_vecif_200_ro = ro
		_vecif_200_fresnel = fresnel
		_vecif_200_rd = rd
		_vecif_200_reflected = reflected
		_vecif_200_bgc = bgc
		_vecif_200_col = col
		_vecif_200_ro = h_add_vf_t_vf(_vecif_200_ro, h_mul_t_f_t_vf(refldist, _vecif_200_rd))
		coord = swizzle_t_n3_xz(_vecif_200_ro)
		bumpfactor = h_mul_f_t_f(0.10000000000000001, h_sub_f_t_f(1.0, h_smoothstep_n_n_t_n(0.0, 60.0, refldist)))
		dx = h_f2_n_n(0.10000000000000001, 0.0)
		dz = h_f2_n_n(0.0, 0.10000000000000001)
		_vecif_200_normal = h_f3_n_n_n(0.0, 1.0, 0.0)
		_, _vecif_200_normal = swizzle_set_and_broadcast_n3_x(_vecif_200_normal, h_div_t_f_f(h_mul_t_f_t_f(h_sub_t_f(bumpfactor), h_sub_t_f_t_f(waterMap_f2_arr(h_add_t_vf_vf(coord, dx)), waterMap_f2_arr(h_sub_t_vf_vf(coord, dx)))), 0.20))
		swizzle_set_t_n3_z(_vecif_200_normal, h_div_t_f_f(h_mul_t_f_t_f(h_sub_t_f(bumpfactor), h_sub_t_f_t_f(waterMap_f2_arr(h_add_t_vf_vf(coord, dz)), waterMap_f2_arr(h_sub_t_vf_vf(coord, dz)))), 0.20))
		_vecif_200_normal = h_normalize_t_v(_vecif_200_normal)
		ndotr = h_dot_t_v_t_v(_vecif_200_normal, _vecif_200_rd)
		_vecif_200_fresnel = h_pow_t_n_n(h_sub_f_t_f(1.0, h_abs_t_n(ndotr)), 5.0)
		_vecif_200_rd = h_reflect_t_v_t_v(_vecif_200_rd, _vecif_200_normal)
		_vecif_200_reflected = True
		_vecif_200_bgc = (_vecif_200_col := bgColor_f3_arr(_vecif_200_rd))
		normal = h_where_t_n_t_v_v(_vecif_200_exp, _vecif_200_normal, normal)
		ro = h_where_t_n_t_v_v(_vecif_200_exp, _vecif_200_ro, ro)
		fresnel = h_where_t_n_t_n_n(_vecif_200_exp, _vecif_200_fresnel, fresnel)
		rd = h_where_t_n_t_v_t_v(_vecif_200_exp, _vecif_200_rd, rd)
		reflected = h_where_t_n_n_n(_vecif_200_exp, _vecif_200_reflected, reflected)
		bgc = h_where_t_n_t_v_t_v(_vecif_200_exp, _vecif_200_bgc, bgc)
		col = h_where_t_n_t_v_t_v(_vecif_200_exp, _vecif_200_col, col)
	else:
		normal = h_broadcast_t_f3_f3(fragCoord, normal)
		ro = h_broadcast_t_f3_f3(fragCoord, ro)
		fresnel = h_broadcast_t_f_f(fragCoord, fresnel)
		reflected = h_broadcast_t_b_b(fragCoord, reflected)
	col = tuple_get_retval((_call_ret_202 := raymarchTerrain_f3_f3_f3_f_f_arr(ro, rd, col, h_where_t_n_t_n_n(reflected, h_sub_f_t_f(800.0, refldist), 800.0), maxdist), maxdist := tuple_get_outparam(_call_ret_202, 1)))
	col = raymarchClouds_f3_f3_f3_f3_f_f_f_arr(ro, rd, col, bgc, h_where_t_n_t_n_n(reflected, h_max_n_t_n(0.0, h_min_n_t_n(150.0, h_sub_f_t_f(150.0, refldist))), 150.0), maxdist, h_mul_f_f(h_add_f_f(iTime, 285.0), 0.050000000000000003))
	_vecif_203_exp = reflected
	if any_ifexp_true_t_n(_vecif_203_exp):
		_vecif_203_col = col
		_vecif_203_refldist = refldist
		_vecif_203_col = h_lerp_t_v_t_v_t_n(swizzle_t_n3_xyz(_vecif_203_col), bgc, h_sub_f_t_f(1.0, h_exp_t_n(h_mul_t_f_t_f(h_mul_f_t_f(-4.9999999999999998E-7, _vecif_203_refldist), _vecif_203_refldist))))
		_vecif_203_col = h_mul_t_vf_t_f(_vecif_203_col, h_mul_t_f_f(fresnel, 0.90000000000000002))
		refr = h_refract_t_v_t_v_n(rdo, normal, 0.7501876)
		tuple_get_retval((_call_ret_204 := intersectPlane_f3_f3_f_f_arr1(ro, refr, -2.0, _vecif_203_refldist), _vecif_203_refldist := tuple_get_outparam(_call_ret_204, 1)))
		_vecif_203_col = h_add_t_vf_t_vf(_vecif_203_col, h_mul_t_vf_f(h_mul_t_vf_t_f(h_lerp_t_v_v_t_n(h_mul_t_vf_vf(swizzle_t_n4_xyz(Texture2D_Sample_n_t_v(iChannel2, s_linear_clamp_sampler, h_mul_t_vf_f(swizzle_t_n3_xz(h_add_vf_t_vf(roo, h_mul_t_f_t_vf(_vecif_203_refldist, refr))), 1.3))), h_f3_n_n_n(1.0, 0.90000000000000002, 0.59999999999999998)), h_mul_vf_f(h_f3_n_n_n(1.0, 0.90000000000000002, 0.80000000000000004), 0.5), h_clamp_t_n_n_n(h_div_t_f_f(_vecif_203_refldist, 3.0), 0.0, 1.0)), h_sub_f_t_f(1.0, fresnel)), 0.125))
		col = h_where_t_n_t_v_t_v(_vecif_203_exp, _vecif_203_col, col)
		refldist = h_where_t_n_t_n_t_n(_vecif_203_exp, _vecif_203_refldist, refldist)
	col = h_pow_t_v_v(col, glsl_vec3_f(0.69999999999999996))
	col = h_mul_t_vf_t_vf(h_mul_t_vf_t_vf(col, col), h_sub_f_t_vf(3.0, h_mul_f_t_vf(2.0, col)))
	col = h_lerp_t_v_t_v_n(col, glsl_vec3_f_arr(h_dot_t_v_v(col, glsl_vec3_f(0.33000000000000002))), -0.5)
	col = h_mul_t_vf_t_f(col, h_add_f_t_f(0.25, h_mul_f_t_f(0.75, h_pow_t_n_n(h_mul_t_f_t_f(h_mul_t_f_t_f(h_mul_t_f_t_f(h_mul_f_t_f(16.0, swizzle_t_n2_x(q)), swizzle_t_n2_y(q)), h_sub_f_t_f(1.0, swizzle_t_n2_x(q))), h_sub_f_t_f(1.0, swizzle_t_n2_y(q))), 0.10000000000000001))))
	fragColor = h_t_f4_t_n3_n(col, 1.0)
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
m2 = h_f2x2_n_n_n_n(0.59999999999999998, -0.80000000000000004, 0.80000000000000004, 0.59999999999999998)
m3 = h_f3x3_n_n_n_n_n_n_n_n_n(0.0, 0.80000000000000004, 0.59999999999999998, -0.80000000000000004, 0.35999999999999999, -0.47999999999999998, -0.59999999999999998, -0.47999999999999998, 0.64000000000000001)
s_linear_clamp_sampler = None
iChannel0 = None
iChannel1 = None
iChannel2 = None
iChannel3 = None
lig = h_f3_defval()

def init_globals():
	global lig
	lig = h_normalize_v(h_f3_n_n_n(0.29999999999999999, 0.5, 0.59999999999999998))

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
	g_show_with_opengl = False
	g_is_autodiff = False
	g_is_profiling = False
	g_is_full_vectorized = True
	g_face_color = "gray"
	g_win_zoom = 1
	g_win_size = None
	iResolution = np.asarray([210, 118, 1])

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

