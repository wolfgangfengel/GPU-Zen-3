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
