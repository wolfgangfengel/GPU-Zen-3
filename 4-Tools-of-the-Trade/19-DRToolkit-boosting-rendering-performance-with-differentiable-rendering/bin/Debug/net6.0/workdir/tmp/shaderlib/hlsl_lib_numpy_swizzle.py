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
