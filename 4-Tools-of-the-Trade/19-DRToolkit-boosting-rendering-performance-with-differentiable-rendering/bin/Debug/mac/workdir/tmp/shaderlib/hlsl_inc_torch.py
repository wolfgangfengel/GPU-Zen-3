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
            wtitle = "time:{0:.3f} frame time:{1:.3f} iter:{2}".format(iTime, iTimeDelta, ct)
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
