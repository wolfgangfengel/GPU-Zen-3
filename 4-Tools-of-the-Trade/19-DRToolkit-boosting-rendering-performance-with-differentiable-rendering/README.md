# hlsl2python

This is part of the content for drtoolkit

https://github.com/dreamanlan/Hlsl2Python

It mainly handles the translation of material shaders.
Updates related to drtoolkit will be updated here.

## English Version ##

## Command Line #

```
Hlsl2Python [-out outfile] [-args arg_dsl_file] [-entry main_func_name] [-tex2d0~3 tex_file] [-tex3d0~3 tex_file] [-texcube0~3 tex_file] [- shadertoy] [-gl] [-profiling] [-notorch] [-autodiff] [-gendsl] [-rewritedsl] [-printblocks] [-printfinalblocks] [-notvectorizebranch] [- novecterization] [-noconst] [-multiple] [-unroll max_loop] [-debug] [-src ] hlsl_file
 
[-out outfile] output file path and name
 
[-args arg_dsl_file] config file path and name, default is [hlsl_file_name]_args.dsl
 
[-entry main_func_name] shader entry function, default is mainImage for shadertoy, main for glsl/compute, frag for hlsl
 
[-tex2d0~3 tex_file] 2d channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer
 
[-tex3d0~3 tex_file] 3d channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer
 
[-texcube0~3 tex_file] cube channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer
 
[-shadertoy] shadertoy mode
 
[-gl] render with opengl [-ngl] render with matplotlib (default)
 
[-profiling] profiling mode [-notprofiling] normal mode (default)
 
[-notorch] dont use PyTorch lib [-torch] use PyTorch lib (default)
 
[-autodiff] autodiff mode, only valid with torch mode [-noautodiff] normal mode (default)
 
[-gendsl] generate prune dsl and final dsl [-notgendsl] dont generate prune dsl and final dsl (default)
 
[-rewritedsl] rewrite [hlsl_file_name].dsl to [hlsl_file_name].txt [-notrewritedsl] dont rewrite dsl (default)
 
[-printblocks] output dataflow structure built in scalar phase [-notprintblocks] dont output (default)
 
[-printfinalblocks] output dataflow structure built in vectorizing phase [-notprintfinalblocks] dont output (default)
 
[-notvectorizebranch] dont remove loop branches [-vectorizebranch] remove loop branches (default)
 
[-novecterization] dont do vectorization [-vecterization] do vectorization (default)
 
[-noconst] dont do const propagation [-const] do const propagation (default)
 
[-multiple] output standalone python lib files [-single] output one python file including lib contents (default)
 
[-unroll max_loop] max loop count while unroll loops, -1 default, means dont unroll an uncounted loops
 
[-debug] debug mode
 
[-src ] hlsl_file source hlsl/glsl file, -src can be omitted when file is the last argument
```
 
## Usage

 **There are two main types of usage**

- One is based on shadertoy, which has a relatively complete startup framework, so as long as there are no errors in the translation and no runtime library functions that need to be completed, you can run and see the results.

1. Debug the shader on Shadertoy and avoid using writable global variables, structures, and arrays.
2. Copy the shader content to the test.glsl file under bin/Debug/net6.0/workdir.
3. Run testglsl.bat.
4. Alternatively, if you are using a new file name, specify the file name as a command-line parameter for translation (you can refer to the command in testglsl.bat).
5. In the conda environment, go to the bin/Debug/net6.0/workdir/tmp directory and execute
	 
`python test.py`.


- The other category is the regular hlsl shader or compute shader, this category currently does not provide a startup framework, it will only generate an empty startup function, you need to modify the runtime (mainly because of the lack of relevant parameters, entry signatures, and runtime flow of the specification), you can generally refer to the logic of the startup function of the shadertoy startup function to modify the startup function
 
1. copy the contents of the shader into the hlsl_test.hlsl file under bin/Debug/net6.0/workdir.
2. Execute testhlsl.bat.
3. Modify the startup function of bin/Debug/net6.0/workdir/tmp/hlsl_test.py to provide shader global parameters and entry function parameters, call the shader entry function in a loop, and plot the tensor according to the shader result.
4. In the conda environment, go to the bin/Debug/net6.0/workdir/tmp directory, and run
 
`python hlsl_test.py`

**Conda environment installation:**
 
1. Install either Anaconda or Miniconda. Anaconda is more comprehensive and includes a Jupyter Notebook, which is convenient for graphical analysis.
2. [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution "Anaconda")
3. [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html "Miniconda")
4. If you need to do automatic differentiation or use PyTorch lib, install Cuda 11.7+ SDK [https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_ version=10&target_type=exe_local](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version= 10&target_type=exe_local "Cuda"), then install PyTorch 1.13+ [https://pytorch.org/get-started/locally/#anaconda](https://pytorch.org/get- started/locally/#anaconda "PyTorch with conda")
5. enter the conda environment using base or create a new environment using python 3.9 or higher
6. Install the python libraries using pip (if using Anaconda, many of the libraries may already be installed).
 		
`pip install matplotlib numpy numba cupy-cuda11x imageio PyOpenGL glfw`
 
7. If using PyTorch, install pyjion
 
`pip install matplotlib numpy pyjion imageio PyOpenGL glfw`
`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

8. The environment is ready.


## Workflow (shadertoy as an example) ##


- When the input file is glsl, assuming it is test.glsl, dxc is first used for preprocessing to obtain the tmp/test.i file.
- Hlsl2python checks whether there is a configuration file named test_args.dsl in the directory where test.glsl is located. If it exists, the configuration is loaded. Some configurations in the configuration file can also be specified as command-line parameters, which take precedence over the configurations in the configuration file.
- Hlsl2python first processes the conditional expressions in the test.i file (mainly those with assignment statements embedded in the conditional expressions, which need to be enclosed in parentheses, otherwise the syntax cannot be accepted by metadsl).
- Metadsl is used to load the content of test.i with processed conditional expressions, and then perform relatively simple program structure and lexical scope analysis, rewrite the syntax inconsistent between glsl and hlsl, and output the test.hlsl file. At this time, glsl.h is included, which contains some macro definitions and hlsl functions to provide syntax or functionality in glsl. In addition, for structures and arrays, initialization functions for structures and arrays will be generated.
- Note: If the input file is hlsl, only the step of reading the configuration file will be performed. The main process starts from the following step. However, the input file name in our example batch file is hlsl_test.hlsl, mainly to avoid conflicts with the output of test.glsl.
- Next, the modified version of dxr is used to read test.hlsl and output test.dsl. During this process, dxr will perform syntax checking on test.hlsl and report an error if there is any. We need to check the error location in test.hlsl based on the line number in the error message, and then make corresponding modifications in test.glsl.
- After that, hlsl2python enters the main process. For any subsequent errors, we need to check the error location in test.dsl, which is indicated by the line number in the error message.
- Hlsl2python will analyze test.dsl for 2 or more times, and then output the translation result according to the command-line parameters. If the command-line parameters specify not to perform vectorization, only normal scalar functions will be output. Otherwise, the vectorized shader translation result will be generated. The output of vectorization starts from the entry function and only outputs the functions defined in the shader that will actually be called.


## Frequently asked questions ##

1. Shadertoy's shader may use modifiable global variables. When translating, an error will be reported for variables involving vectorization. In this case, these global variables need to be moved to the function that uses them earliest and become local variables. Then, other functions that use them become in or inout parameters. (For simplicity, inout is used for all variables that are modified. If using out, be careful to check whether each branch of the function assigns a value to the variable.)
2. If an error is reported during translation that a certain loop cannot be unrolled, the line number in test.dsl is provided. It is necessary to open test.dsl to see which loop is causing the error. Generally, it is because the loop condition is too complex or the number of iterations cannot be calculated. For complex loop conditions, only the loop conditions such as i < n need to be retained, and other conditions can be moved into the loop body and judged with if(!cond), and then break.
	- If the loop is not a simple for(int i=0;i<n;++i), it needs to be manually converted to this style. Alternatively, an hlsl attribute can be added. The method of adding this attribute in glsl is to add hlsl_attr([unroll(maximum number of iterations)]) before the loop. Hlsl2python will parse this attribute and unroll the loop a specified number of times during translation.
	- The command-line parameter -unroll can also specify a maximum number of iterations. If the number of iterations cannot be calculated and there is no loop marked with the unroll attribute, the loop will be unrolled according to this parameter value. (Hlsl2python only unrolls loops less than 512 times, otherwise an error will be reported, because loop unrolling will cause a sharp increase in the number of lines of code in the translation result, which is not convenient for debugging and reading.)
3. If the translation is normal, but an error is reported at runtime that a certain function cannot be found, it may be because the corresponding API implementation is missing in this parameter type combination in our Python library function. You can modify the Python library file in the bin/Debug/.net60/shaderlib directory to add the relevant API. Please note that each API needs to have implementations in both NumPy and PyTorch, and APIs are usually implemented in hlsl_lib_numpy.py and hlsl_lib_torch.py, respectively.
4. If there is an exception in texture loading at runtime, it may be due to problems with the aforementioned API implementation or bugs in the startup framework loading part. There is also a startup framework file for NumPy and PyTorch, respectively, named hlsl_inc_numpy.py and hlsl_inc_torch.py.
5. Hlsl2python uses template code generation to generate swizzle APIs. The gencode directory contains the code generation tool and template code for generating code (gen_hlsl_lib_numpy_swizzle.dsl and gen_hlsl_lib_torch_swizzle.dsl for NumPy and PyTorch, respectively, and gen_glsl_h.dsl is used to generate some code that will be used when converting glsl to hlsl in glsl.h). The generated swizzle code is also in the shaderlib directory, with one copy for NumPy and one for PyTorch, named hlsl_lib_numpy_swizzle.py and hlsl_lib_torch_swizzle.py, respectively.
6. In very rare cases, an error may be reported during the test.hlsl stage of translation, indicating a compilation error. In this case, check the error in test.hlsl according to the error message, and then modify the corresponding content in test.glsl. This situation should not occur usually. If it does, there may be a problem with the glsl preprocessing stage, and it may be necessary to modify the hlsl2python source code to solve it.



## Chinese Version 



## 命令行 ##

```
    Hlsl2Python [-out outfile] [-args arg_dsl_file] [-entry main_func_name] [-tex2d0~3 tex_file] [-tex3d0~3 tex_file] [-texcube0~3 tex_file] [-shadertoy] [-gl] [-profiling] [-notorch] [-autodiff] [-gendsl] [-rewritedsl] [-printblocks] [-printfinalblocks] [-notvectorizebranch] [-novecterization] [-noconst] [-multiple] [-unroll max_loop] [-debug] [-src ] hlsl_file
    
     [-out outfile] output file path and name
    
     [-args arg_dsl_file] config file path and name, default is [hlsl_file_name]_args.dsl
    
     [-entry main_func_name] shader entry function, default is mainImage for shadertoy, main for glsl/compute, frag for hlsl
    
     [-tex2d0~3 tex_file] 2d channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer
    
     [-tex3d0~3 tex_file] 3d channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer
    
     [-texcube0~3 tex_file] cube channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer
    
     [-shadertoy] shadertoy mode
    
     [-gl] render with opengl [-ngl] render with matplotlib (default)
    
     [-profiling] profiling mode [-notprofiling] normal mode (default)
    
     [-notorch] dont use PyTorch lib [-torch] use PyTorch lib (default)
    
     [-autodiff] autodiff mode, only valid with torch mode [-noautodiff] normal mode (default)
    
     [-gendsl] generate prune dsl and final dsl [-notgendsl] dont generate prune dsl and final dsl (default)
    
     [-rewritedsl] rewrite [hlsl_file_name].dsl to [hlsl_file_name].txt [-notrewritedsl] dont rewrite dsl (default)
    
     [-printblocks] output dataflow structure built in scalar phase [-notprintblocks] dont output (default)
    
     [-printfinalblocks] output dataflow structure built in vectorizing phase [-notprintfinalblocks] dont output (default)
    
     [-notvectorizebranch] dont remove loop branches [-vectorizebranch] remove loop branches (default)
    
     [-novecterization] dont do vectorization [-vecterization] do vectorization (default)
    
     [-noconst] dont do const propagation [-const] do const propagation (default)
    
     [-multiple] output standalone python lib files [-single] output one python file including lib contents (default)
    
     [-unroll max_loop] max loop count while unroll loops, -1 default, means dont unroll an uncounted loops
    
     [-debug] debug mode
    
     [-src ] hlsl_file source hlsl/glsl file, -src can be omitted when file is the last argument
```
 
## 用法 ##

 **主要有2类用法**

- 一类是基于shadertoy的，此类别目前有相对比较完善的启动框架，只要翻译没有报错，运行时库函数没有需要补全的，就可以运行看到结果
  1. 在shadertoy上调通shader，尽量避免使用可写全局变量，结构体与数组
  2. 将shader内容拷贝到bin/Debug/net6.0/workdir下的test.glsl文件中
  3. 执行testglsl.bat
  4. 或者使用新文件名的话，就使用命令行参数来指定文件名并翻译（可以参考testglsl.bat里的命令）
  5. 在conda环境下，进入bin/Debug/net6.0/workdir/tmp目录，执行
	  
		`python test.py`


- 另一类是常规的hlsl的shader或compute shader，此类别目前不提供启动框架，只会生成一个空的启动函数，需要自己修改运行（主要是缺少相关参数、入口签名以及运行流程的规范），一般来说可以参照shadertoy的启动函数逻辑来修改启动函数
  
  1. 将shader内容拷入bin/Debug/net6.0/workdir下的hlsl_test.hlsl文件
  2. 执行testhlsl.bat
  3. 修改bin/Debug/net6.0/workdir/tmp/hlsl_test.py的启动函数，提供shader全局参数与入口函数参数，循环调用shader入口函数，并根据shader结果张量绘图
  4. 在conda环境下，进入bin/Debug/net6.0/workdir/tmp目录，执行
  
	  	`python hlsl_test.py`

**conda环境安装：**
  
   1. 安装Anaconda或Miniconda均可，Anaconda更全一些，还包括一个Jupyter Notebook，可以比较方便的做图形分析
   2. [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution "Anaconda")
   3. [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html "Miniconda")
   4. 如果需要做自动微分或使用PyTorch lib，安装Cuda 11.7以上SDK[https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local "Cuda")，然后安装PyTorch  1.13以上版本[https://pytorch.org/get-started/locally/#anaconda](https://pytorch.org/get-started/locally/#anaconda "PyTorch with conda")
   5. 进入conda环境，使用base或创建一个新的环境，使用python 3.9以上版本
   6. 使用pip安装python库（如果使用Anaconda，可能许多库已经安装），
   		
      `pip install matplotlib numpy numba cupy-cuda11x imageio PyOpenGL glfw`
 
   7. 如果使用PyTorch，安装pyjion
   
      `pip install matplotlib numpy pyjion imageio PyOpenGL glfw`
      `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

   8. 环境准备完毕。

## 工作流程（shadertoy为例） ##


- 输入文件为glsl时，假设是test.glsl，首先使用dxc进行预处理，得到tmp/test.i文件。
- hlsl2python会检查看test.glsl所在目录下是否有test_args.dsl的配置文件，如果存在则加载配置，配置文件里有一些配置也可以在命令行参数指定，此时命令行参数优先于配置文件里的配置
- hlsl2python首先对test.i文件的条件表达式进行处理（主要是条件表达式里嵌入赋值语句的，需要使用括号括起来，否则语法无法被metadsl接受）。
- 使用metadsl加载处理过条件表达式的test.i的内容，然后进行比较简单的程序结构、词法范围分析，改写glsl与hlsl不一致的语法，然后输出test.hlsl文件，此时会包含glsl.h，这个头文件有一些宏定义与hlsl的函数来提供glsl里的语法或功能，另外，对于结构与数组，会生成结构与数组的初始化函数。
- 注：如果输入文件是hlsl，前面只会有读取配置文件的步骤，主要流程从下面这步开始，不过我们的例子批处理里的输入文件名是hlsl_test.hlsl，主要是为了避免与test.glsl的输出冲突
- 接下来使用修改版本的dxr读取test.hlsl并输出test.dsl，这个过程，dxr会对test.hlsl做语法检查，如果有错会报错。我们需要根据错误中的行号在test.hlsl文件里查看出错位置，然后回到test.glsl里进行相应修改。
- 之后进入hlsl2python的主体流程，之后的报错，需要在test.dsl文件里查看出错位置，行号是test.dsl的行号。
- hlsl2python对test.dsl会进行2遍以上分析，然后根据命令行参数来输出翻译结果，如果命令行参数指明不进行向量化，则只输出普通标量函数，否则生成向量化的shader翻译结果，向量化的输出从入口函数开始推导，只输出实际会调用到的shader里定义的函数。


## 常见问题 ##

1. shadertoy的shader里可能使用了可修改的全局变量，此时翻译时会对涉及向量化的变量报错，需要将这些全局变量放到最早使用它的函数里变成局部变量，然后其它使用它的函数变为in或inout参数（为了简便，只要涉及修改的都用inout，使用out需要小心的检查函数的每个分支是否都对该变量进行了赋值）
2. 翻译时报某个循环无法展开，此时的行号是test.dsl的行号，需要打开test.dsl查看是哪个循环，一般都是循环的条件比较复杂或无法计算循环次数，对于复杂循环条件，可以只保留i<n这样的循环条件，然后将其它条件挪到循环体内用if(!cond)判断，然后break。
   - 如果循环不是简单的for(int i=0;i<n;++i)这类，那么需要人工变成这样的样式，或者，加一个hlsl的属性，在glsl里加这种属性的方法是在循环前加hlsl_attr([unroll(最大循环次数)])，hlsl2python在翻译时会解析这个属性并其指定次数展开循环。
   - 命令行参数的-unroll也可以指定一个最大次数，此时无法计算循环次数又没有使用unroll属性标记的循环就会按这个参数值展开。（hlsl2python只展开少于512次的循环，否则会报错，因为循环展开会导致翻译结果代码行数急剧增加，不便于调试与阅读）
3. 翻译时正常，然后在运行时报某个函数无法找到，这种情形是我们的python库函数里少了相应api在这个参数类型组合下的实现，可以修改bin/Debug/.net60/shaderlib目录下的python库文件来添加相关api，需要注意的是每个api都需要有NumPy与PyTorch两种实现，api通常是在hlsl_lib_numpy.py与hlsl_lib_torch.py里实现。
4. 如果运行时出现纹理加载异常，有可能是前述api实现有问题，或者是启动框架的加载部分有bug，启动框架文件也是NumPy与PyTorch各有一个，文件名为hlsl_inc_numpy.py与hlsl_inc_torch.py。
5. hlsl2python使用模板代码生成来生成swizzle的api，gencode目录下是代码生成工具与生成代码的模板代码（NumPy与PyTorch各有一个gen_hlsl_lib_numpy_swizzle.dsl与gen_hlsl_lib_torch_swizzle.dsl，另外有一个gen_glsl_h.dsl是用来生成glsl.h里面glsl转到hlsl时会用到的一些代码），生成的swizzle代码也在shaderlib目录下，numpy与torch各一份，分别为hlsl_lib_numpy_swizzle.py与hlsl_lib_torch_swizzle.py。
6. 极少数情形可能遇到翻译时在test.hlsl阶段报编译错误 ，此时根据错误提示检查在test.hlsl的错误原因，然后修改test.glsl里的对应内容，通常这种情况不应出现，出现就有可能是glsl预处理阶段有问题了，可能需要修改hlsl2python源码来解决。

## 实现原理 ##

需要了解实现细节的可以参考[https://zhuanlan.zhihu.com/p/618721749](https://zhuanlan.zhihu.com/p/618721749 "hlsl/glsl翻译到pytorch/numpy（笔记）")，当然更完整的都在代码里。
