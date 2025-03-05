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
