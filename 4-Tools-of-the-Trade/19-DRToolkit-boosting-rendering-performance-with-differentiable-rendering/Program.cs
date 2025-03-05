using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;

namespace Hlsl2Python
{
    internal static class StringBuilderExtension
    {
        public static void Append(this StringBuilder sb, string fmt, params object[] args)
        {
            if (args.Length == 0) {
                sb.Append(fmt);
            }
            else {
                sb.AppendFormat(fmt, args);
            }
        }
        public static void AppendLine(this StringBuilder sb, string fmt, params object[] args)
        {
            if (args.Length == 0) {
                sb.AppendLine(fmt);
            }
            else {
                sb.AppendFormat(fmt, args);
                sb.AppendLine();
            }
        }
    }
    internal partial class Program
    {
        static void Main(string[] args) 
        {
            if (args.Length == 0) {
                PrintHelp();
                return;
            }
            else {
                bool isGlsl = false;
                bool isCompute = false;
                bool isDsl = false;
                bool fromShaderToy = false;
                bool opengl = false;
                bool profiling = false;
                bool autodiff = false;
                bool genDsl = false;
                bool rewriteDsl = false;
                bool printBlocks = false;
                bool printFinalBlocks = false;
                bool multiple = false;
                string srcFilePath = string.Empty;
                string outFilePath = string.Empty;
                string argFilePath = string.Empty;
                string mainEntryFunc = string.Empty;
                int maxLoop = -1;
                for (int i = 0; i < args.Length; ++i) {
                    if (0 == string.Compare(args[i], "-out", true)) {
                        if (i < args.Length - 1) {
                            string arg = args[i + 1];
                            if (!arg.StartsWith("-")) {
                                outFilePath = arg;
                                ++i;
                            }
                        }
                    }
                    else if (0 == string.Compare(args[i], "-src", true)) {
                        if (i < args.Length - 1) {
                            string arg = args[i + 1];
                            if (!arg.StartsWith("-")) {
                                srcFilePath = arg;
                                if (!File.Exists(srcFilePath)) {
                                    Console.WriteLine("file path not found ! {0}", srcFilePath);
                                }
                                ++i;
                            }
                        }
                    }
                    else if (0 == string.Compare(args[i], "-args", true)) {
                        if (i < args.Length - 1) {
                            string arg = args[i + 1];
                            if (!arg.StartsWith("-")) {
                                argFilePath = arg;
                                if (!File.Exists(argFilePath)) {
                                    Console.WriteLine("file path not found ! {0}", argFilePath);
                                }
                                ++i;
                            }
                        }
                    }
                    else if (0 == string.Compare(args[i], "-entry", true)) {
                        if (i < args.Length - 1) {
                            string key = args[i].Substring(1);
                            string arg = args[i + 1];
                            if (!arg.StartsWith("-")) {
                                mainEntryFunc = arg;
                                s_CmdArgKeys.Add(key);
                                ++i;
                            }
                        }
                    }
                    else if (0 == string.Compare(args[i], "-tex2d0", true) ||
                        0 == string.Compare(args[i], "-tex2d1", true) ||
                        0 == string.Compare(args[i], "-tex2d2", true) ||
                        0 == string.Compare(args[i], "-tex2d3", true) ||
                        0 == string.Compare(args[i], "-tex3d0", true) ||
                        0 == string.Compare(args[i], "-tex3d1", true) ||
                        0 == string.Compare(args[i], "-tex3d2", true) ||
                        0 == string.Compare(args[i], "-tex3d3", true) ||
                        0 == string.Compare(args[i], "-texcube0", true) ||
                        0 == string.Compare(args[i], "-texcube1", true) ||
                        0 == string.Compare(args[i], "-texcube2", true) ||
                        0 == string.Compare(args[i], "-texcube3", true)) {
                        if (i < args.Length - 1) {
                            string key = args[i].Substring(1);
                            string arg = args[i + 1];
                            if (!arg.StartsWith("-")) {
                                s_CmdArgKeys.Add(key);
                                SetShaderToyTex(s_MainShaderInfo, key, arg);
                                ++i;
                            }
                        }
                    }
                    else if (0 == string.Compare(args[i], "-shadertoy", true)) {
                        isGlsl = true;
                        fromShaderToy = true;
                    }
                    else if (0 == string.Compare(args[i], "-gl", true)) {
                        opengl = true;
                    }
                    else if (0 == string.Compare(args[i], "-ngl", true)) {
                        opengl = false;
                    }
                    else if (0 == string.Compare(args[i], "-profiling", true)) {
                        profiling = true;
                    }
                    else if (0 == string.Compare(args[i], "-notprofiling", true)) {
                        profiling = false;
                    }
                    else if (0 == string.Compare(args[i], "-torch", true)) {
                        s_IsTorch = true;
                    }
                    else if (0 == string.Compare(args[i], "-notorch", true)) {
                        s_IsTorch = false;
                    }
                    else if (0 == string.Compare(args[i], "-autodiff", true)) {
                        autodiff = true;
                    }
                    else if (0 == string.Compare(args[i], "-noautodiff", true)) {
                        autodiff = false;
                    }
                    else if (0 == string.Compare(args[i], "-gendsl", true)) {
                        genDsl = true;
                    }
                    else if (0 == string.Compare(args[i], "-notgendsl", true)) {
                        genDsl = false;
                    }
                    else if (0 == string.Compare(args[i], "-rewritedsl", true)) {
                        rewriteDsl = true;
                    }
                    else if (0 == string.Compare(args[i], "-notrewritedsl", true)) {
                        rewriteDsl = false;
                    }
                    else if (0 == string.Compare(args[i], "-printblocks", true)) {
                        printBlocks = true;
                    }
                    else if (0 == string.Compare(args[i], "-notprintblocks", true)) {
                        printBlocks = false;
                    }
                    else if (0 == string.Compare(args[i], "-printfinalblocks", true)) {
                        printFinalBlocks = true;
                    }
                    else if (0 == string.Compare(args[i], "-notprintfinalblocks", true)) {
                        printFinalBlocks = false;
                    }
                    else if (0 == string.Compare(args[i], "-vectorizebranch", true)) {
                        s_AutoVectorizeBranch = true;
                    }
                    else if (0 == string.Compare(args[i], "-notvectorizebranch", true)) {
                        s_AutoVectorizeBranch = false;
                    }
                    else if (0 == string.Compare(args[i], "-vectorization", true)) {
                        s_EnableVectorization = true;
                    }
                    else if (0 == string.Compare(args[i], "-novectorization", true)) {
                        s_EnableVectorization = false;
                    }
                    else if (0 == string.Compare(args[i], "-const", true)) {
                        s_EnableConstPropagation = true;
                    }
                    else if (0 == string.Compare(args[i], "-noconst", true)) {
                        s_EnableConstPropagation = false;
                    }
                    else if (0 == string.Compare(args[i], "-single", true)) {
                        multiple = false;
                    }
                    else if (0 == string.Compare(args[i], "-multiple", true)) {
                        multiple = true;
                    }
                    else if (0 == string.Compare(args[i], "-maxloop", true)) {
                        if (i < args.Length - 1) {
                            string key = args[i].Substring(1);
                            string arg = args[i + 1];
                            if (!arg.StartsWith("-")) {
                                int.TryParse(arg, out maxLoop);
                                ++i;
                            }
                        }
                    }
                    else if (0 == string.Compare(args[i], "-hlsl2018", true)) {
                        s_UseHlsl2018 = true;
                    }
                    else if (0 == string.Compare(args[i], "-hlsl2021", true)) {
                        s_UseHlsl2018 = false;
                    }
                    else if (0 == string.Compare(args[i], "-debug", true)) {
                        s_IsDebugMode = true;
                        s_StringBuilderPool.IsDebugMode = true;
                    }
                    else if (0 == string.Compare(args[i], "-h", true)) {
                        PrintHelp();
                    }
                    else if (args[i][0] == '-') {
                        Console.WriteLine("unknown command option ! {0}", args[i]);
                    }
                    else {
                        srcFilePath = args[i];
                        if (!File.Exists(srcFilePath)) {
                            Console.WriteLine("file path not found ! {0}", srcFilePath);
                        }
                        break;
                    }
                }

                string oldCurDir = Environment.CurrentDirectory;
                string exeFullName = System.Reflection.Assembly.GetExecutingAssembly().Location;
                string? exeDir = Path.GetDirectoryName(exeFullName);
                Debug.Assert(null != exeDir);
                Environment.CurrentDirectory = exeDir;
                Console.WriteLine("curdir {0} change to exedir {1}", oldCurDir, exeDir);
                try {
                    string? workDir = Path.GetDirectoryName(srcFilePath);
                    Debug.Assert(null != workDir);
                    string tmpDir = Path.Combine(workDir, "tmp");
                    if (!Directory.Exists(tmpDir)) {
                        Directory.CreateDirectory(tmpDir);
                    }
                    string srcFileName = Path.GetFileName(srcFilePath);
                    string srcFileNameWithoutExt = Path.GetFileNameWithoutExtension(srcFileName);
                    string hlslFilePath = srcFilePath;
                    bool needPreprocess = false;
                    string srcExt = Path.GetExtension(srcFilePath);
                    if (srcExt == ".compute") {
                        isCompute = true;
                    }
                    if (srcExt == ".dsl")
                    {
                        isDsl = true;
                    }
                    else if (srcExt != ".hlsl") {
                        hlslFilePath = Path.Combine(tmpDir, Path.ChangeExtension(srcFileName, "hlsl"));
                        needPreprocess = true;
                        if (srcExt == ".glsl") {
                            isGlsl = true;
                            s_UseHlsl2018 = true;
                        }
                    }

                    string dxcHlslVerOption = string.Empty;
                    if (s_UseHlsl2018)
                        dxcHlslVerOption = "-HV 2018 ";

                    string dslFilePath = Path.Combine(tmpDir, Path.ChangeExtension(srcFileName, "dsl"));
                    string dsl2FilePath = Path.Combine(tmpDir, srcFileNameWithoutExt + "_pruned.dsl");
                    string dsl3FilePath = Path.Combine(tmpDir, srcFileNameWithoutExt + "_finally.dsl");
                    string rewriteFilePath = Path.Combine(tmpDir, Path.ChangeExtension(srcFileName, "txt"));
                    string pyFilePath = outFilePath;
                    if (string.IsNullOrEmpty(pyFilePath)) {
                        pyFilePath = Path.Combine(tmpDir, Path.ChangeExtension(srcFileName, "py"));
                    }
                    string? pyDir = Path.GetDirectoryName(pyFilePath);
                    Debug.Assert(null != pyDir);

                    var additionalEntryFuncs = new HashSet<string>();
                    if (string.IsNullOrEmpty(argFilePath)) {
                        argFilePath = Path.Combine(workDir, srcFileNameWithoutExt + "_args.dsl");
                    }
                    if (File.Exists(argFilePath)) {
                        LoadShaderArgs(argFilePath);
                        if (!string.IsNullOrEmpty(s_MainShaderInfo.Entry)) {
                            mainEntryFunc = s_MainShaderInfo.Entry;
                        }
                        foreach(var bufInfo in s_ShaderBufferInfos) {
                            additionalEntryFuncs.Add(bufInfo.Entry);
                        }
                    }

                    //0、preprocess src file
                    string libDir = Path.Combine(pyDir, "shaderlib");
                    if (!Directory.Exists(libDir)) {
                        Directory.CreateDirectory(libDir);
                    }
                    var libFiles = Directory.GetFiles("shaderlib");
                    foreach (var libFile in libFiles) {
                        string libExt = Path.GetExtension(libFile);
                        string libFileName = Path.GetFileName(libFile);
                        string targetFile = Path.Combine(libDir, libFileName);
                        if (multiple && libExt == ".py") {
                            if (libFileName.StartsWith("hlsl_lib_numpy_")) {
                                var impLines = new List<string>();
                                impLines.Add("import numpy as np");
                                impLines.Add("from numba import njit");
                                var libLines = File.ReadAllLines(libFile);
                                impLines.AddRange(libLines);
                                File.WriteAllLines(targetFile, impLines);
                            }
                            else if (libFileName.StartsWith("hlsl_lib_torch_")) {
                                var impLines = new List<string>();
                                impLines.Add("import numpy as np");
                                impLines.Add("import torch");
                                impLines.Add("import pyjion #conflict with matplotlib");
                                impLines.Add("pyjion.enable()");
                                var libLines = File.ReadAllLines(libFile);
                                impLines.AddRange(libLines);
                                File.WriteAllLines(targetFile, impLines);
                            }
                            else {
                                File.Copy(libFile, targetFile, true);
                            }
                        }
                        else {
                            File.Copy(libFile, targetFile, true);
                        }
                    }
                    if (needPreprocess) {
                        if (isGlsl) {
                            //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                            //call dxc.exe -P -Fi input.i input.glsl
                            //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                            string iFilePath = Path.Combine(tmpDir, Path.ChangeExtension(srcFileName, "i"));
                            var coption = new ProcessStartOption();
                            int cr = -1;
                            if(OperatingSystem.IsWindows())
                                cr = RunProcess("dxc.exe", "-P -Fi \"" + iFilePath + "\" \"" + srcFilePath + "\"", coption, null, null, null, null, null, false, true, Encoding.UTF8);
                            else
                                cr = RunProcess("dxc", "-P -Fi \"" + iFilePath + "\" \"" + srcFilePath + "\"", coption, null, null, null, null, null, false, true, Encoding.UTF8);
                            if (cr != 0) {
                                Console.WriteLine("run dxc failed, exit code:{0}", cr);
                            }
                            PreprocessGlsl(iFilePath, hlslFilePath, fromShaderToy);
                        }
                        else {
                            PreprocessHlsl(srcFilePath, hlslFilePath, ref isCompute);
                        }
                    }
                    string dslTxt = string.Empty;
                    if (isDsl) {
                        dslTxt = File.ReadAllText(dslFilePath);
                    }
                    else {
                        //call dxr.exe convert hlsl to dsl
                        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        //dxr.exe compiled from https://github.com/dreamanlan/DirectXShaderCompiler/tree/Dxc2Dsl, a modified DXC fork
                        //dsl.dll compiled from https://github.com/dreamanlan/MetaDSL (master branch), a common meta-DSL language for data and logic (JSON, in contrast, is primarily used for data)
                        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        var option = new ProcessStartOption();
                        var output = new StringBuilder();
                        int r = -1;
                        if (OperatingSystem.IsWindows()) {
                            r = RunProcess("dxr.exe", dxcHlslVerOption + "\"" + hlslFilePath + "\"", option, null, null, null, output, null, false, true, Encoding.UTF8);
                        }
                        else {
                            r = RunProcess("dxr", dxcHlslVerOption + "\"" + hlslFilePath + "\"", option, null, null, null, output, null, false, true, Encoding.UTF8);
                        }
                        if (r != 0)
                        {
                            Console.WriteLine("run dxr failed, exit code:{0}", r);
                        }

                        if (string.IsNullOrEmpty(mainEntryFunc))
                        {
                            if (fromShaderToy)
                            {
                                mainEntryFunc = "mainImage";
                            }
                            else if (isGlsl)
                            {
                                mainEntryFunc = "main";
                            }
                            else if (isCompute)
                            {
                                mainEntryFunc = "main";
                            }
                            else
                            {
                                mainEntryFunc = "frag";
                            }
                        }

                        dslTxt = output.ToString();
                        File.WriteAllText(dslFilePath, dslTxt);
                    }

                    //dsl transform to python
                    var dslFile = new Dsl.DslFile();
                    if (dslFile.LoadFromString(dslTxt, msg => { Console.WriteLine(msg); })) {
                        if (rewriteDsl)
                            dslFile.Save(rewriteFilePath);

                        var sb = new StringBuilder();
                        if (multiple) {
                            sb.AppendLine("import sys");
                            sb.AppendLine("import os");
                            sb.AppendLine("import inspect");
                            sb.AppendLine("hlslLibPath = os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), './shaderlib'))");
                            sb.AppendLine("if hlslLibPath not in sys.path:");
                            sb.AppendLine("\tsys.path.append(hlslLibPath)");
                            sb.AppendLine("#print(sys.path)");
                        }
                        var imports = s_IsTorch ? s_TorchImports : s_NumpyImports;
                        var include = s_IsTorch ? s_TorchInc : s_NumpyInc;
                        if (!s_UseHlsl2018) {
                            imports = s_IsTorch ? s_TorchImportsFor2021 : s_NumpyImportsFor2021;
                            include = s_IsTorch ? s_TorchIncFor2021 : s_NumpyIncFor2021;
                        }
                        foreach (var importFile in imports) {
                            if (multiple) {
                                string ifn = Path.GetFileNameWithoutExtension(importFile);
                                sb.Append("from " + ifn + " import *");
                                sb.AppendLine();
                            }
                            else {
                                string txt = File.ReadAllText(Path.Combine("shaderlib", importFile));
                                sb.AppendLine(txt);
                            }
                        }
                        sb.AppendLine();
                        string incTxt = File.ReadAllText(Path.Combine("shaderlib", include));
                        sb.AppendLine(incTxt);

                        var globalSb = new StringBuilder();
                        var initSb = new StringBuilder();
                        //first pass
                        foreach (var info in dslFile.DslInfos) {
                            string id = info.GetId();
                            if (id == "struct") {
                                ParseStruct(info);
                            }
                            else if (id == "cbuffer") {
                                ParseCBuffer(info);
                            }
                            else if (id == "typedef") {
                                ParseTypeDef(info);
                            }
                        }
                        //second pass
                        foreach (var info in dslFile.DslInfos) {
                            string id = info.GetId();
                            if (id == "var") {
                                var vinfo = ParseVarDecl(info);
                                if (null != vinfo) {
                                    s_DeclGlobals.Add(vinfo);
                                }
                            }
                            else if (id == "=") {
                                string ty = string.Empty;
                                var tempSb = NewStringBuilder();
                                TransformAssignmentStatement(id, info, tempSb, 0, ref ty, out var vname);
                                var vinfo = GetVarInfo(vname, VarUsage.Find);
                                if (null != vinfo) {
                                    if(vinfo.IsConst || fromShaderToy && s_ShaderToyParamNames.Contains(vinfo.Name)) {
                                        s_InitGlobals.Remove(vinfo.Name);
                                        globalSb.Append(tempSb);
                                    }
                                    else {
                                        s_DeclGlobals.Add(vinfo);
                                        initSb.Append("\t");
                                        initSb.Append(tempSb);
                                    }
                                }
                                RecycleStringBuilder(tempSb);
                            }
                            else if (id == "func") {
                                ParseAndPruneFuncDef(info, mainEntryFunc, additionalEntryFuncs);
                            }
                        }

                        const int c_max_globals_per_line_for_init = 10;

                        foreach(var varInfo in s_DeclGlobals) {
                            GenDeclVar(globalSb, 0, varInfo);
                        }
                        globalSb.AppendLine();
                        globalSb.AppendLine("def init_globals():");
                        if (s_InitGlobals.Count > 0) {
                            for (int ix = 0; ix < s_InitGlobals.Count; ++ix) {
                                if (ix % c_max_globals_per_line_for_init == 0) {
                                    if (ix > 0)
                                        globalSb.AppendLine();
                                    globalSb.Append("\tglobal ");
                                }
                                else if (ix > 0) {
                                    globalSb.Append(", ");
                                }
                                var name = s_InitGlobals[ix];
                                globalSb.Append(name);
                            }
                            globalSb.AppendLine();
                            globalSb.Append(initSb);
                        }
                        else {
                            globalSb.AppendLine("\tpass");
                        }

                        if (genDsl) {
                            dslFile.Save(dsl2FilePath);
                        }
                        CacheGlobalCallInfo();

                        //final pass
                        foreach (var info in dslFile.DslInfos) {
                            string id = info.GetId();
                            if (id == "func") {
                                TransformFunc(info);
                            }
                        }

                        if (printBlocks) {
                            Console.WriteLine("===[Blocks:]===");
                            foreach (var pair in s_FuncInfos) {
                                string sig = pair.Key;
                                var func = pair.Value;
                                Console.WriteLine("func:{0}", sig);
                                Console.WriteLine("\tparams:");
                                foreach (var p in func.Params) {
                                    Console.WriteLine("\t\tname:{0} type:{1}", p.Name, p.Type);
                                }
                                Console.WriteLine("\treturn:{0}", null == func.RetInfo ? "void" : func.RetInfo.Type);
                                if (null != func.ToplevelBlock)
                                    func.ToplevelBlock.Print(1);
                            }
                        }

                        GenerateScalarFuncCode();
                        TryRemoveFuncBranches(maxLoop);

                        //3、vectorization
                        var transformSb = new StringBuilder();
                        if (s_EnableVectorization) {
                            s_IsVectorizing = true;
                            s_IsFullVectorized = true;
                            Vectorizing(transformSb);
                        }
                        else {
                            s_IsFullVectorized = false;
                            foreach (var sig in s_AllFuncSigs) {
                                if (s_AllFuncCodes.TryGetValue(sig, out var fsb)) {
                                    transformSb.Append(fsb);
                                    if (s_FuncInfos.TryGetValue(sig, out var fi)) {
                                        transformSb.AppendLine();
                                    }
                                }
                            }
                        }

                        if (genDsl) {
                            dslFile.Save(dsl3FilePath);
                        }

                        if (printFinalBlocks) {
                            Console.WriteLine("===[Final Blocks:]===");
                            foreach (var sig in s_BranchRemovedFuncs) {
                                if (s_FuncInfos.TryGetValue(sig, out var func)) {
                                    Console.WriteLine("func:{0}", sig);
                                    Console.WriteLine("\tparams:");
                                    foreach (var p in func.Params) {
                                        Console.WriteLine("\t\tname:{0} type:{1}", p.Name, p.Type);
                                    }
                                    Console.WriteLine("\treturn:{0}", null == func.RetInfo ? "void" : func.RetInfo.Type);
                                    if (null != func.ToplevelBlock)
                                        func.ToplevelBlock.Print(1);
                                }
                            }
                        }

                        foreach (var pair in s_AutoGenCodes) {
                            sb.Append(pair.Value);
                        }
                        sb.Append(transformSb);
                        sb.AppendLine();
                        sb.Append(globalSb);
                        sb.AppendLine();
                        if (fromShaderToy) {
                            sb.AppendLine("def shader_main(fc, fcd):");
                            sb.Append("\tglobal ");
                            var initBuffers = new SortedSet<string>();
                            var usingChannels = new SortedSet<string>();
                            var definedBuffers = new SortedSet<string>();
                            string prestr = string.Empty;
                            foreach (var bufInfo in s_ShaderBufferInfos) {
                                var bufId = bufInfo.BufferId;
                                sb.Append("{0}{1}", prestr, bufId);
                                prestr = ", ";
                                foreach (var pair in bufInfo.TexTypes) {
                                    var texname = pair.Key;
                                    var textype = pair.Value;
                                    string channel = bufId + "_" + texname;
                                    if (bufInfo.TexBuffers.TryGetValue(texname, out var texbuf)) {
                                        if (s_ShaderToyBufferNames.Contains(texbuf)) {
                                            if (!definedBuffers.Contains(texbuf)) {
                                                if (!initBuffers.Contains(texbuf))
                                                    initBuffers.Add(texbuf);
                                            }
                                        }
                                        if (!usingChannels.Contains(channel))
                                            usingChannels.Add(channel);
                                    }
                                    else if (bufInfo.TexFiles.TryGetValue(texname, out var texfile)) {
                                        if (!usingChannels.Contains(channel))
                                            usingChannels.Add(channel);
                                        sb.Append("{0}g_{1}_{2}", prestr, bufId, texname);
                                        prestr = ", ";
                                    }
                                }
                                if (!definedBuffers.Contains(bufId))
                                    definedBuffers.Add(bufId);
                            }
                            foreach (var pair in s_MainShaderInfo.TexTypes) {
                                var texname = pair.Key;
                                var textype = pair.Value;
                                string channel = texname;
                                if (s_MainShaderInfo.TexBuffers.TryGetValue(texname, out var texbuf)) {
                                    if (!usingChannels.Contains(channel))
                                        usingChannels.Add(channel);
                                }
                                else if (s_MainShaderInfo.TexFiles.TryGetValue(texname, out var texfile)) {
                                    if (!usingChannels.Contains(channel))
                                        usingChannels.Add(channel);
                                    sb.Append("{0}g_main_{1}", prestr, texname);
                                    prestr = ", ";
                                }
                            }
                            foreach(var channel in usingChannels) {
                                sb.Append("{0}{1}", prestr, channel);
                                prestr = ", ";
                            }
                            sb.AppendLine();
                            sb.AppendLine("\tinit_globals()");
                            sb.AppendLine("\tiChannelTime[0] = iTime");
                            sb.AppendLine("\tiChannelTime[1] = iTime");
                            sb.AppendLine("\tiChannelTime[2] = iTime");
                            sb.AppendLine("\tiChannelTime[3] = iTime");
                            sb.AppendLine();
                            foreach (var bufInfo in s_ShaderBufferInfos) {
                                var entry = bufInfo.Entry;
                                var bufId = bufInfo.BufferId;
                                if (s_FuncOverloads.TryGetValue(entry, out var overloads)) {
                                    foreach (var pair in bufInfo.TexTypes) {
                                        var texname = pair.Key;
                                        var textype = pair.Value;
                                        if (bufInfo.TexBuffers.TryGetValue(texname, out var texbuf)) {
                                            sb.AppendLine("\t{0}_{1} = {2}", bufId, texname, texbuf);
                                        }
                                        else if(bufInfo.TexFiles.TryGetValue(texname, out var texfile)) {
                                            sb.AppendLine("\t{0}_{1} = g_{0}_{2}", bufId, texname, texname);
                                        }
                                        int chanIx = s_ShaderToyChannels.IndexOf(texname);
                                        if (chanIx >= 0) {
                                            sb.AppendLine("\tset_channel_resolution({0}, {1}_{2})", chanIx, bufId, texname);
                                        }
                                    }
                                    foreach (var argInfo in bufInfo.ArgInfos) {
                                        sb.AppendLine("\t{0}_{1} = {2}", bufId, argInfo.ArgName, argInfo.ArgValue);
                                    }
                                    foreach (var sig in overloads) {
                                        var vsig = sig + c_VectorialNameSuffix;
                                        if (s_IsFullVectorized || s_AllUsingFuncOrApis.Contains(vsig)) {
                                            sb.AppendLine("\t{0} = buffer_to_tex({1}{2}(fc, fcd))", bufInfo.BufferId, sig, c_VectorialNameSuffix);
                                        }
                                        else {
                                            sb.AppendLine("\t{0} = buffer_to_tex({1}{2}(fc, fcd))", bufInfo.BufferId, sig, c_VectorialAdapterNameSuffix);
                                        }
                                        break;
                                    }
                                }
                            }
                            foreach (var pair in s_MainShaderInfo.TexTypes) {
                                var texname = pair.Key;
                                var textype = pair.Value;
                                if (s_MainShaderInfo.TexBuffers.TryGetValue(texname, out var texbuf)) {
                                    sb.AppendLine("\t{0} = {1}", texname, texbuf);
                                }
                                else {
                                    sb.AppendLine("\t{0} = g_main_{0}", texname);
                                }
                                int chanIx = s_ShaderToyChannels.IndexOf(texname);
                                if (chanIx >= 0) {
                                    sb.AppendLine("\tset_channel_resolution({0}, {1})", chanIx, texname);
                                }
                            }
                            foreach (var argInfo in s_MainShaderInfo.ArgInfos) {
                                sb.AppendLine("\t{0} = {1}", argInfo.ArgName, argInfo.ArgValue);
                            }
                            if (s_IsFullVectorized || s_AllUsingFuncOrApis.Contains(s_MainEntryFuncSignature + c_VectorialNameSuffix)) {
                                sb.AppendLine("\treturn {0}{1}(fc, fcd)", s_MainEntryFuncSignature, c_VectorialNameSuffix);
                            }
                            else {
                                sb.AppendLine("\treturn {0}{1}(fc, fcd)", s_MainEntryFuncSignature, c_VectorialAdapterNameSuffix);
                            }
                            sb.AppendLine();
                            sb.Append("if __name__ == \"__main__\":");
                            sb.AppendLine();
                            if (opengl)
                                sb.AppendLine("\tg_show_with_opengl = True");
                            else
                                sb.AppendLine("\tg_show_with_opengl = False");
                            if (autodiff)
                                sb.AppendLine("\tg_is_autodiff = True");
                            else
                                sb.AppendLine("\tg_is_autodiff = False");
                            if (profiling)
                                sb.AppendLine("\tg_is_profiling = True");
                            else
                                sb.AppendLine("\tg_is_profiling = False");
                            if (s_IsFullVectorized)
                                sb.AppendLine("\tg_is_full_vectorized = True");
                            else
                                sb.AppendLine("\tg_is_full_vectorized = False");

                            sb.AppendLine("\tg_face_color = \"{0}\"", s_MainShaderInfo.FaceColor);
                            sb.AppendLine("\tg_win_zoom = {0}", s_MainShaderInfo.WinZoom > 0.01f ? s_MainShaderInfo.WinZoom.ToString() : "None");
                            sb.AppendLine("\tg_win_size = {0}", s_MainShaderInfo.WinSize > 0.01f ? s_MainShaderInfo.WinSize.ToString() : "None");

                            var resoFast = s_MainShaderInfo.ResolutionOnFullVec;
                            if (s_IsTorch)
                                resoFast = s_MainShaderInfo.ResolutionOnGpuFullVec;
                            var resoSlow = s_MainShaderInfo.Resolution;
                            var reso = s_IsFullVectorized ? resoFast : resoSlow;
                            var initMouse = s_MainShaderInfo.InitMousePos;
                            sb.AppendLine("\tiResolution = {0}.asarray([{1}, {2}, {3}])", s_IsTorch ? "torch" : "np", reso.x, reso.y, reso.z);
                            sb.AppendLine();
                            sb.AppendLine("\tiMouse[0] = iResolution[0] * {0}", initMouse.x);
                            sb.AppendLine("\tiMouse[1] = iResolution[1] * {0}", initMouse.y);
                            sb.AppendLine("\tiMouse[2] = iResolution[0] * {0}", initMouse.z);
                            sb.AppendLine("\tiMouse[3] = iResolution[1] * {0}", initMouse.w);
                            sb.AppendLine();

                            foreach(var bufId in initBuffers) {
                                sb.AppendLine("\t{0} = init_buffer()", bufId);
                            }
                            foreach (var bufInfo in s_ShaderBufferInfos) {
                                var bufId = bufInfo.BufferId;
                                foreach (var pair in bufInfo.TexTypes) {
                                    var texname = pair.Key;
                                    var textype = pair.Value;
                                    if (bufInfo.TexBuffers.TryGetValue(texname, out var texbuf)) {
                                    }
                                    else if (bufInfo.TexFiles.TryGetValue(texname, out var texfile)) {
                                        if (textype == "sampler2D")
                                            sb.AppendLine("\tg_{0}_{1} = load_tex_2d(\"{2}\")", bufId, texname, texfile);
                                        else if(textype == "sampler3D")
                                            sb.AppendLine("\tg_{0}_{1} = load_tex_3d(\"{2}\")", bufId, texname, texfile);
                                        else
                                            sb.AppendLine("\tg_{0}_{1} = load_tex_cube(\"{2}\")", bufId, texname, texfile);
                                    }
                                }
                            }
                            foreach (var pair in s_MainShaderInfo.TexTypes) {
                                var texname = pair.Key;
                                var textype = pair.Value; 
                                if (s_MainShaderInfo.TexBuffers.TryGetValue(texname, out var texbuf)) {
                                }
                                else if (s_MainShaderInfo.TexFiles.TryGetValue(texname, out var texfile)) {
                                    if (textype == "sampler2D")
                                        sb.AppendLine("\tg_main_{0} = load_tex_2d(\"{1}\")", texname, texfile);
                                    else if (textype == "sampler3D")
                                        sb.AppendLine("\tg_main_{0} = load_tex_3d(\"{1}\")", texname, texfile);
                                    else
                                        sb.AppendLine("\tg_main_{0} = load_tex_cube(\"{1}\")", texname, texfile);
                                }
                            }

                            sb.AppendLine("\tif g_is_autodiff and g_is_profiling:");
                            sb.AppendLine("\t\tprofile_entry(main_entry_autodiff)");
                            sb.AppendLine("\telif g_is_autodiff:");
                            sb.AppendLine("\t\tmain_entry_autodiff()");
                            sb.AppendLine("\telif g_is_profiling:");
                            sb.AppendLine("\t\tprofile_entry(main_entry)");
                            sb.AppendLine("\telse:");
                            sb.AppendLine("\t\tmain_entry()");
                            sb.AppendLine();
                        }
                        else {
                            string vecsig = s_MainEntryFuncSignature + c_VectorialNameSuffix;
                            if (isCompute) {
                                sb.AppendLine("def shader_main(fc, fcd):");
                                if (s_IsFullVectorized || s_AllUsingFuncOrApis.Contains(vecsig)) {
                                    sb.AppendLine("\treturn compute_dispatch(fc, fcd, {0}{1})", s_MainEntryFuncSignature, c_VectorialNameSuffix);
                                }
                                else {
                                    sb.AppendLine("\treturn compute_dispatch(fc, fcd, {0}{1})", s_MainEntryFuncSignature, c_VectorialAdapterNameSuffix);
                                }
                            }
                            else {
                                sb.AppendLine("def shader_main(fc, fcd):");
                                sb.AppendLine("\tiChannelTime[0] = iTime");
                                sb.AppendLine("\tiChannelTime[1] = iTime");
                                sb.AppendLine("\tiChannelTime[2] = iTime");
                                sb.AppendLine("\tiChannelTime[3] = iTime");
                                sb.AppendLine();
                                if (s_IsFullVectorized || s_AllUsingFuncOrApis.Contains(vecsig)) {
                                    sb.AppendLine("\treturn shader_dispatch(fc, fcd, {0}{1})", s_MainEntryFuncSignature, c_VectorialNameSuffix);
                                }
                                else {
                                    sb.AppendLine("\treturn shader_dispatch(fc, fcd, {0}{1})", s_MainEntryFuncSignature, c_VectorialAdapterNameSuffix);
                                }
                            }
                            sb.AppendLine();
                            sb.Append(globalSb);
                            sb.AppendLine();
                            sb.Append("if __name__ == \"__main__\":");
                            sb.AppendLine();
                            if (opengl)
                                sb.AppendLine("\tg_show_with_opengl = True");
                            else
                                sb.AppendLine("\tg_show_with_opengl = False");
                            if (autodiff)
                                sb.AppendLine("\tg_is_autodiff = True");
                            else
                                sb.AppendLine("\tg_is_autodiff = False");
                            if (profiling)
                                sb.AppendLine("\tg_is_profiling = True");
                            else
                                sb.AppendLine("\tg_is_profiling = False");
                            if (s_IsFullVectorized)
                                sb.AppendLine("\tg_is_full_vectorized = True");
                            else
                                sb.AppendLine("\tg_is_full_vectorized = False");

                            sb.AppendLine("\tg_face_color = \"{0}\"", s_MainShaderInfo.FaceColor);
                            sb.AppendLine("\tg_win_zoom = {0}", s_MainShaderInfo.WinZoom > 0.01f ? s_MainShaderInfo.WinZoom.ToString() : "None");
                            sb.AppendLine("\tg_win_size = {0}", s_MainShaderInfo.WinSize > 0.01f ? s_MainShaderInfo.WinSize.ToString() : "None");

                            var resoFast = s_MainShaderInfo.ResolutionOnFullVec;
                            if (s_IsTorch)
                                resoFast = s_MainShaderInfo.ResolutionOnGpuFullVec;
                            var resoSlow = s_MainShaderInfo.Resolution;
                            var reso = s_IsFullVectorized ? resoFast : resoSlow;
                            var initMouse = s_MainShaderInfo.InitMousePos;
                            sb.AppendLine("\tiResolution = {0}.asarray([{1}, {2}, {3}])", s_IsTorch ? "torch" : "np", reso.x, reso.y, reso.z);
                            sb.AppendLine();
                            sb.AppendLine("\tiMouse[0] = iResolution[0] * {0}", initMouse.x);
                            sb.AppendLine("\tiMouse[1] = iResolution[1] * {0}", initMouse.y);
                            sb.AppendLine("\tiMouse[2] = iResolution[0] * {0}", initMouse.z);
                            sb.AppendLine("\tiMouse[3] = iResolution[1] * {0}", initMouse.w);
                            sb.AppendLine();

                            foreach (var argInfo in s_MainShaderInfo.ArgInfos) {
                                if (argInfo.ArgType == "tex2d") {
                                    sb.AppendLine("\t{0} = load_tex_2d(\"{1}\")", argInfo.ArgName, argInfo.ArgValue);
                                }
                                else if (argInfo.ArgType == "tex3d") {
                                    sb.AppendLine("\t{0} = load_tex_3d(\"{1}\")", argInfo.ArgName, argInfo.ArgValue);
                                }
                                else if (argInfo.ArgType == "texcube") {
                                    sb.AppendLine("\t{0} = load_tex_cube(\"{1}\")", argInfo.ArgName, argInfo.ArgValue);
                                }
                                else {
                                    sb.AppendLine("\t{0} = {1}", argInfo.ArgName, argInfo.ArgValue);
                                }
                            }

                            sb.AppendLine("\tif g_is_profiling:");
                            sb.AppendLine("\t\tprofile_entry(main_entry)");
                            sb.AppendLine("\telse:");
                            sb.AppendLine("\t\tmain_entry()");
                            sb.AppendLine();
                        }
                        File.WriteAllText(pyFilePath, sb.ToString());
                    }
                    Console.WriteLine("Transform done.");
                    Console.Out.Flush();
                }
                catch(Exception ex) {
                    Console.WriteLine("{0}", ex.Message);
                    Console.WriteLine("[Stack]:");
                    Console.WriteLine("{0}", ex.StackTrace);
                }
                finally {
                    Environment.CurrentDirectory = oldCurDir;
                }
            }
        }
        static void PrintHelp()
        {
            Console.WriteLine("Usage:Hlsl2Python [-out outfile] [-args arg_dsl_file] [-entry main_func_name] [-tex2d0~3 tex_file] [-tex3d0~3 tex_file] [-texcube0~3 tex_file] [-shadertoy] [-gl] [-profiling] [-notorch] [-autodiff] [-gendsl] [-rewritedsl] [-printblocks] [-printfinalblocks] [-notvectorizebranch] [-novecterization] [-noconst] [-multiple] [-maxloop max_loop] [-hlsl2018] [-hlsl2021] [-debug] [-src ] hlsl_file");
            Console.WriteLine(" [-out outfile] output file path and name");
            Console.WriteLine(" [-args arg_dsl_file] config file path and name, default is [hlsl_file_name]_args.dsl");
            Console.WriteLine(" [-entry main_func_name] shader entry function, default is mainImage for shadertoy, main for glsl/compute, frag for hlsl");
            Console.WriteLine(" [-tex2d0~3 tex_file] 2d channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer");
            Console.WriteLine(" [-tex3d0~3 tex_file] 3d channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer");
            Console.WriteLine(" [-texcube0~3 tex_file] cube channel texture file for shadertoy, a channel can be 2d or 3d or cube or buffer");
            Console.WriteLine(" [-shadertoy] shadertoy mode");
            Console.WriteLine(" [-gl] render with opengl [-ngl] render with matplotlib (default)");
            Console.WriteLine(" [-profiling] profiling mode [-notprofiling] normal mode (default)");
            Console.WriteLine(" [-notorch] dont use pytorch lib [-torch] use pytorch lib (default)");
            Console.WriteLine(" [-autodiff] autodiff mode, only valid with torch mode [-noautodiff] normal mode (default)");
            Console.WriteLine(" [-gendsl] generate prune dsl and final dsl [-notgendsl] dont generate prune dsl and final dsl (default)");
            Console.WriteLine(" [-rewritedsl] rewrite [hlsl_file_name].dsl to [hlsl_file_name].txt [-notrewritedsl] dont rewrite dsl (default)");
            Console.WriteLine(" [-printblocks] output dataflow structure built in scalar phase [-notprintblocks] dont output (default)");
            Console.WriteLine(" [-printfinalblocks] output dataflow structure built in vectorizing phase [-notprintfinalblocks] dont output (default)");
            Console.WriteLine(" [-notvectorizebranch] dont remove loop branches [-vectorizebranch] remove loop branches (default)");
            Console.WriteLine(" [-novecterization] dont do vectorization [-vecterization] do vectorization (default)");
            Console.WriteLine(" [-noconst] dont do const propagation [-const] do const propagation (default)");
            Console.WriteLine(" [-multiple] output standalone python lib files [-single] output one python file including lib contents (default)");
            Console.WriteLine(" [-maxloop max_loop] max loop count while unroll loops, -1 default, means dont unroll an uncounted loops");
            Console.WriteLine(" [-hlsl2018] use hlsl 2018, default");
            Console.WriteLine(" [-hlsl2021] use hlsl 2021 or later");
            Console.WriteLine(" [-debug] debug mode");
            Console.WriteLine(" [-src ] hlsl_file source hlsl/glsl file, -src can be omitted when file is the last argument");
        }
        private static void CacheGlobalCallInfo()
        {
            foreach (var fn in s_AllUsingFuncOrApis) {
                s_GlobalUsingFuncOrApis.Add(fn);
            }
            foreach (var pair in s_AutoGenCodes) {
                s_GlobalAutoGenCodes.Add(pair.Key, pair.Value);
            }
        }
        private static void SwapScalarCallInfo()
        {
            var apiNames = s_ScalarUsingFuncOrApis;
            s_ScalarUsingFuncOrApis = s_AllUsingFuncOrApis;
            s_AllUsingFuncOrApis = apiNames;

            var codes = s_ScalarAutoGenCodes;
            s_ScalarAutoGenCodes = s_AutoGenCodes;
            s_AutoGenCodes = codes;
        }
        private static void MergeGlobalCallInfo()
        {
            foreach (var fn in s_GlobalCalledScalarFuncs) {
                if (s_FuncInfos.TryGetValue(fn, out var fi)) {
                    MarkCalledScalarFunc(fi);
                }
            }
            foreach (var fn in s_GlobalUsingFuncOrApis) {
                if(!s_AllUsingFuncOrApis.Contains(fn))
                    s_AllUsingFuncOrApis.Add(fn);
            }
            foreach (var pair in s_GlobalAutoGenCodes) {
                if (!s_AutoGenCodes.ContainsKey(pair.Key))
                    s_AutoGenCodes.Add(pair.Key, pair.Value);
            }
        }
        private static void AddUsingFuncOrApi(string funcName)
        {
            if (!s_AllUsingFuncOrApis.Contains(funcName))
                s_AllUsingFuncOrApis.Add(funcName);
            var curFunc = CurFuncInfo();
            if (null != curFunc && !curFunc.UsingFuncOrApis.Contains(funcName)) {
                curFunc.UsingFuncOrApis.Add(funcName);
            }
        }
        private static void MarkCalledScalarFunc(FuncInfo funcInfo)
        {
            MarkCalledScalarFunc(funcInfo, false);
        }
        private static void MarkCalledScalarFunc(FuncInfo funcInfo, bool isVecAdapter)
        {
            string sig = funcInfo.Signature;
            if (s_IsVectorizing) {
                if (!isVecAdapter)
                    funcInfo.ResetScalarFuncInfo();
                MarkCalledScalarFuncRecursively(sig);
            }
            else {
                var curFunc = CurFuncInfo();
                if (null != curFunc) {
                    string csig = curFunc.Signature;
                    if (!s_FuncCallFuncs.TryGetValue(csig, out var funcs)) {
                        funcs = new HashSet<string>();
                        s_FuncCallFuncs.Add(csig, funcs);
                    }
                    if (!funcs.Contains(sig)) {
                        funcs.Add(sig);
                    }
                }
                else if (!s_GlobalCalledScalarFuncs.Contains(sig)) {
                    s_GlobalCalledScalarFuncs.Add(sig);
                }
            }
        }
        private static void MarkCalledScalarFuncRecursively(string funcSig)
        {
            if(s_FuncCallFuncs.TryGetValue(funcSig, out var funcs)) {
                foreach(var sig in funcs) {
                    MarkCalledScalarFuncRecursively(sig);
                }
            }
            if (!s_CalledScalarFuncs.Contains(funcSig))
                s_CalledScalarFuncs.Add(funcSig);
        }
        private static bool IsSameType(string lhsType, string rhsType)
        {
            bool ret = true;
            if (lhsType != rhsType) {
                ret = false;
                string struName = GetTypeNoVecPrefix(lhsType, out var isVecBefore);
                if (IsTupleMatchStruct(struName, rhsType, out var fct, out var vct)) {
                    if (vct == 0) {
                        if (!isVecBefore)
                            ret = true;
                    }
                    else if (vct == fct) {
                        ret = true;
                    }
                }
            }
            return ret;
        }
        private static bool IsTupleMatchStruct(string struName, string type, out int fieldCount, out int vecCt)
        {
            fieldCount = 0;
            vecCt = 0;
            bool ret = false;
            if(s_StructInfos.TryGetValue(struName, out var struInfo)) {
                fieldCount = struInfo.Fields.Count;
                int ix = 0;
                if(IsTupleMatchStruct(struInfo, type, ref ix, out var vecCt0)) {
                    vecCt = vecCt0;
                    ret = true;
                }
            }
            return ret;
        }
        private static bool IsTupleMatchStruct(StructInfo struInfo, string type, ref int ix, out int vecCt)
        {
            vecCt = 0;
            bool ret = true;
            int nix0 = type.IndexOf(struInfo.Name, ix);
            if (nix0 == ix) {
                ix += struInfo.Name.Length;
            }
            else {
                string vty0 = GetTypeVec(struInfo.Name);
                nix0 = type.IndexOf(vty0, ix);
                if (nix0 == ix) {
                    ix += vty0.Length;
                    vecCt += struInfo.Fields.Count;
                }
                else {
                    ret = false;
                    string tag = c_TupleTypePrefix + struInfo.Fields.Count.ToString();
                    string struType = struInfo.Name;
                    string vecStruType = c_VectorialTypePrefix + struInfo.Name;
                    if (type.IndexOf(tag, ix) == ix) {
                        ix += tag.Length;
                        ret = true;
                        foreach (var fi in struInfo.Fields) {
                            string ty = fi.Type;
                            if (ix + 1 < type.Length && type[ix] == '_' && type[ix + 1] == '_') {
                                ix += 2;
                                if (s_StructInfos.TryGetValue(ty, out var cstruInfo)) {
                                    if (IsTupleMatchStruct(cstruInfo, type, ref ix, out var vct)) {
                                        if (cstruInfo.Fields.Count == vct)
                                            ++vecCt;
                                    }
                                    else {
                                        ret = false;
                                        break;
                                    }
                                }
                                else {
                                    int nix = type.IndexOf(ty, ix);
                                    if (nix == ix) {
                                        ix += ty.Length;
                                    }
                                    else {
                                        string vty = GetTypeVec(ty);
                                        nix = type.IndexOf(vty, ix);
                                        if (nix == ix) {
                                            ix += vty.Length;
                                            ++vecCt;
                                        }
                                        else {
                                            ret = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            else {
                                ret = false;
                                break;
                            }
                        }
                    }
                    else if(type.IndexOf(struType, ix) == ix) {
                        ix += struType.Length;
                        ret = true;
                    }
                    else if (type.IndexOf(vecStruType, ix) == ix) {
                        ix += vecStruType.Length;
                        ret = true;
                    }
                }
            }
            return ret;
        }
        private static string GetWhereResultType(string condExpType, string exp1Type, string exp2Type)
        {
            bool isVec = false;
            string opd1 = condExpType;
            string opd2 = exp1Type;
            string opd3 = exp2Type;
            string oriOpd2 = opd2;
            string oriOpd3 = opd3;
            if (s_IsVectorizing) {
                if (IsTypeVec(opd1)) {
                    isVec = true;
                }
                oriOpd2 = GetTypeNoVec(opd2, out var isTuple2, out var isStruct2, out var isVecBefore2);
                if (isVecBefore2) {
                    isVec = true;
                }
                oriOpd3 = GetTypeNoVec(opd3, out var isTuple3, out var isStruct3, out var isVecBefore3);
                if (isVecBefore3) {
                    isVec = true;
                }
            }
            Debug.Assert(oriOpd2 == oriOpd3);
            string resultType = GetMaxType(oriOpd2, oriOpd3);
            if (isVec)
                resultType = GetTypeVec(resultType);
            return resultType;
        }
        private static string GetFuncResultType(string resultTypeTag, string funcOrObjType, IList<string> args, IList<string> oriArgs, bool isVec, bool forGlsl)
        {
            string ret;
            if (resultTypeTag == "@@")
                ret = funcOrObjType;
            else if (resultTypeTag == "@0-$0")
                ret = GetTypeRemoveSuffix(args[0]);
            else if (resultTypeTag == "@0")
                ret = args[0];
            else if (resultTypeTag == "@m") {
                if (forGlsl)
                    ret = GetGlslMaxType(oriArgs);
                else
                    ret = GetMaxType(oriArgs);
            }
            else if (!resultTypeTag.Contains('@') && !resultTypeTag.Contains('$'))
                ret = resultTypeTag;
            else {
                string rt = resultTypeTag.Replace("@@", funcOrObjType).Replace("@0-$0", GetTypeRemoveSuffix(args[0])).Replace("@0", args[0]).Replace("$0", GetTypeSuffix(args[0])).Replace("$R0", GetTypeSuffixReverse(args[0]));
                if (args.Count > 1) {
                    rt = rt.Replace("@1-$1", GetTypeRemoveSuffix(args[1])).Replace("@1", args[1]).Replace("$1", GetTypeSuffix(args[1])).Replace("$R1", GetTypeSuffixReverse(args[1]));
                }
                if (rt.Contains("m")) {
                    string mt;
                    if (forGlsl)
                        mt = GetGlslMaxType(oriArgs);
                    else
                        mt = GetMaxType(oriArgs);
                    rt = rt.Replace("@m-$m", GetTypeRemoveSuffix(mt)).Replace("@m", mt).Replace("$m", GetTypeSuffix(mt)).Replace("$Rm", GetTypeSuffixReverse(mt));
                }
                ret = rt;
            }
            if (isVec)
                ret = GetTypeVec(ret);
            return ret;
        }
        private static string GetMemberResultType(string resultTypeTag, string oriObjType, string memberOrType)
        {
            string ret;
            if (resultTypeTag == "@@")
                ret = oriObjType;
            else if (!resultTypeTag.Contains('@') && !resultTypeTag.Contains('$'))
                ret = resultTypeTag;
            else
                ret = resultTypeTag.Replace("@@", oriObjType).Replace("$0", memberOrType.Length.ToString());
            return ret;
        }
        private static string GetMaxType(params string[] oriArgs)
        {
            IList<string> ats = oriArgs;
            return GetMaxType(ats);
        }
        private static string GetMaxType(IList<string> oriArgs)
        {
            string ret = oriArgs[0];
            for (int i = 1; i < oriArgs.Count; ++i) {
                ret = ret.Length >= oriArgs[i].Length ? ret : oriArgs[i];
            }
            return ret;
        }
        private static string GetMatmulType(string oriTypeA, string oriTypeB, bool isVec)
        {
            string ret;
            string bt1 = GetTypeRemoveSuffix(oriTypeA);
            string s1 = GetTypeSuffix(oriTypeA);
            string bt2 = GetTypeRemoveSuffix(oriTypeB);
            string s2 = GetTypeSuffix(oriTypeB);
            if (s1.Length == 0 && s2.Length == 0)
                ret = GetMaxType(oriTypeA, oriTypeB);
            else if (s1.Length == 0)
                ret = oriTypeB;
            else if (s2.Length == 0)
                ret = oriTypeA;
            else if (s1.Length == 1 && s2.Length == 1)
                ret = GetMaxType(oriTypeA, oriTypeB);
            else if (s1.Length == 1)
                ret = oriTypeA;
            else if (s2.Length == 1)
                ret = oriTypeB;
            else {
                string mt = GetMaxType(bt1, bt2);
                ret = mt + s1[0] + "x" + s2[2];
            }
            if (isVec)
                ret = GetTypeVec(ret);
            return ret;
        }
        private static string GetFullTypeFuncSig(string funcName, IList<string> argTypes)
        {
            var sb = new StringBuilder();
            sb.Append(funcName);
            foreach (var t in argTypes) {
                sb.Append("_");
                sb.Append(GetTypeAbbr(t));
            }
            return sb.ToString();
        }
        private static string GetBaseTypeCtorSig(string funcName, IList<string> argTypes)
        {
            bool hasVectorization = false;
            foreach (var argType in argTypes) {
                if (s_IsVectorizing && IsTypeVec(argType))
                    hasVectorization = true;
            }
            if (hasVectorization) {
                funcName = GetTypeVec(funcName);
            }
            var sb = new StringBuilder();
            sb.Append("h_");
            sb.Append(GetTypeAbbr(funcName));
            foreach(var argType in argTypes) {
                sb.Append(GetSuffixInfoFuncArgTag(argType));
            }
            return sb.ToString();
        }
        private static string GetAssignCastFuncSig(string argTypeDest, string argTypeSrc, out bool needTarget)
        {
            needTarget = false;
            if (s_IsVectorizing) {
                if (!IsTypeVec(argTypeSrc) && IsTypeVec(argTypeDest)) {
                    needTarget = true;
                }
                if (IsTypeVec(argTypeSrc)) {
                    argTypeDest = GetTypeVec(argTypeDest);
                }
            }
            var sb = new StringBuilder();
            if(needTarget)
                sb.Append("h_broadcast_");
            else
                sb.Append("h_cast_");
            sb.Append(GetTypeAbbr(argTypeDest));
            sb.Append("_");
            sb.Append(GetTypeAbbr(argTypeSrc));
            return sb.ToString();
        }
        private static string GetCastFuncSig(string argTypeDest, string argTypeSrc)
        {
            if (s_IsVectorizing) {
                if (IsTypeVec(argTypeSrc)) {
                    argTypeDest = GetTypeVec(argTypeDest);
                }
                if(IsTypeVec(argTypeDest) && !IsTypeVec(argTypeSrc)) {
                    Console.WriteLine("internal error, cast '{0}' to '{1}' !!!", argTypeSrc, argTypeDest);
                    if (!Debugger.IsAttached) {
                        Debug.Assert(false);
                    }
                }
            }
            var sb = new StringBuilder();
            sb.Append("h_cast_");
            sb.Append(GetTypeAbbr(argTypeDest));
            sb.Append("_");
            sb.Append(GetTypeAbbr(argTypeSrc));
            return sb.ToString();
        }
        private static string GetFuncArgsTag(string funcName, params string[] argTypes)
        {
            IList<string> ats = argTypes;
            return GetFuncArgsTag(funcName, ats);
        }
        private static string GetFuncArgsTag(string funcName, IList<string> argTypes)
        {
            var sb = new StringBuilder();
            if (s_KeepFullTypeFuncs.Contains(funcName)) {
                foreach(var t in argTypes) {
                    sb.Append("_");
                    sb.Append(GetTypeAbbr(t));
                }
            }
            else if (s_KeepBaseTypeFuncs.Contains(funcName)) {
                foreach (var t in argTypes) {
                    sb.Append(GetBaseTypeFuncArgTag(t));
                }
            }
            else {
                foreach (var t in argTypes) {
                    sb.Append(GetSimpleFuncArgTag(t));
                }
            }
            return sb.ToString();
        }

        private static string GetSimpleFuncArgTag(string type)
        {
            string suffix = GetTypeSuffix(type, out var isTuple, out var isStruct, out var isArr, out var arrNums, out var typeWithoutArrTag);
            if (isTuple || isStruct)
                return "_" + GetSimpleArrayTypeAbbr(type);
            int dim = (suffix.Length == 3 ? 2 : (suffix.Length > 0 ? 1 : 0));
            string t = arrNums.Count > 0 ? "_" + new string('a', arrNums.Count) : "_";
            switch (dim) {
                case 0:
                    t += "n";
                    break;
                case 1:
                    t += "v";
                    break;
                case 2:
                default:
                    t += "m";
                    break;
            }
            string tag = (isArr ? "_t" : string.Empty) + t;
            return tag;
        }
        private static string GetBaseTypeFuncArgTag(string type)
        {
            string suffix = GetTypeSuffix(type, out var isTuple, out var isStruct, out var isArr, out var arrNums, out var typeWithoutArrTag);
            if (isTuple || isStruct)
                return "_" + GetSimpleArrayTypeAbbr(type);
            string baseType = GetTypeNoVec(GetTypeRemoveSuffix(type));
            int dim = (suffix.Length == 3 ? 2 : (suffix.Length > 0 ? 1 : 0));
            string t = arrNums.Count > 0 ? "_" + new string('a', arrNums.Count) : "_";
            bool addUl = true;
            switch (dim) {
                case 0:
                    addUl = arrNums.Count > 0;
                    break;
                case 1:
                    t += "v";
                    break;
                case 2:
                default:
                    t += "m";
                    break;
            }
            string tag = (isArr ? "_t" : string.Empty) + t + GetBaseTypeAbbr(baseType, addUl);
            return tag;
        }
        private static string GetSuffixInfoFuncArgTag(string type)
        {
            string suffix = GetTypeSuffix(type, out var isTuple, out var isStruct, out var isArr, out var arrNums, out var typeWithoutArrTag);
            if (isTuple || isStruct)
                return "_" + GetSimpleArrayTypeAbbr(type);
            string elemTag = "n";
            string struName = GetTypeNoVecPrefix(typeWithoutArrTag);
            if(s_StructInfos.TryGetValue(struName, out var struInfo)) {
                elemTag = GetTypeAbbr(struName);
            }
            string t = arrNums.Count > 0 ? "_" + new string('a', arrNums.Count) + elemTag : "_n";
            string tag = (isArr ? "_t" : string.Empty) + t + suffix;
            return tag;
        }
        private static string GetSimpleArrayTypeAbbr(string type)
        {
            string typeWithoutArrTag = GetTypeRemoveArrTag(type, out var isTuple, out var isStruct, out var isVec, out var arrNums);
            string tag = GetTypeNoVecPrefix(GetTypeAbbr(typeWithoutArrTag, arrNums.Count > 0));
            string t = (arrNums.Count > 0 ? new string('a', arrNums.Count) : string.Empty) + tag;
            return (isVec ? GetTypeVec(t) : t);
        }
        private static string GetTypeAbbr(string type)
        {
            return GetTypeAbbr(type, false);
        }
        private static string GetTypeAbbr(string type, bool addUlOnComplex)
        {
            if (type.StartsWith(c_TupleTypePrefix)) {
                var sb = new StringBuilder();
                if (addUlOnComplex)
                    sb.Append("_");
                var strs = type.Split("__");
                string prestr = string.Empty;
                foreach (var str in strs) {
                    sb.Append(prestr);
                    if (str.StartsWith(c_TupleTypePrefix)) {
                        sb.Append("tp");
                        sb.Append(str.Substring(c_TupleTypePrefix.Length));
                    }
                    else {
                        sb.Append(GetTypeAbbr(str));
                    }
                    prestr = "_";
                }
                return sb.ToString();
            }
            string typeWithoutVec = GetTypeNoVec(type, out var isTuple, out var isStruct, out var isArr);
            string t;
            if (isStruct) {
                t = (addUlOnComplex ? "_" : string.Empty) + typeWithoutVec;
            }
            else {
                string baseType = GetTypeRemoveSuffix(typeWithoutVec);
                string suffixWithArr = typeWithoutVec.Substring(baseType.Length);
                t = GetBaseTypeAbbr(baseType, addUlOnComplex) + suffixWithArr;
            }
            return (isArr ? GetTypeVec(t) : t);
        }
        private static string GetBaseTypeAbbr(string type, bool addUlOnComplex)
        {
            string ret;
            if (s_BaseTypeAbbrs.TryGetValue(type, out var r)) {
                ret = r;
            }
            else {
                ret = (addUlOnComplex ? "_" : string.Empty) + type;
            }
            return ret;
        }
        private static bool IsBaseType(string type)
        {
            type = GetTypeNoVec(type);
            return type == "bool" || type == "int" || type == "uint" || type == "dword" ||
                        type == "float" || type == "double" || type == "half" ||
                        type == "min16float" || type == "min10float" || type == "min16int" ||
                        type == "min12int" || type == "min16uint";
        }
        private static string GetTypeDims(string objType)
        {
            string ret = string.Empty;
            string oriObjType = GetTypeNoVec(objType);
            string baseType = GetTypeRemoveSuffix(oriObjType, out var isTuple, out var isStruct, out var isVec, out var arrNums, out var typeWithoutArrTag);
            if (isTuple || isStruct) {
                ret = "1";
            }
            else {
                string suffix = GetTypeSuffix(oriObjType);
                if (string.IsNullOrEmpty(suffix))
                    ret = "1";
                else if (suffix.Length == 1)
                    ret = suffix;
                else
                    ret = suffix.Replace("x", ", ");
            }
            return ret;
        }
        private static bool IsMemberAccess(Dsl.ISyntaxComponent func, out Dsl.FunctionData? funcData)
        {
            bool ret = false;
            funcData = func as Dsl.FunctionData;
            if (null != funcData && funcData.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PERIOD) {
                ret = true;
            }
            return ret;
        }
        private static bool IsElementAccess(Dsl.ISyntaxComponent func, out Dsl.FunctionData? funcData)
        {
            bool ret = false;
            funcData = func as Dsl.FunctionData;
            if (null != funcData && funcData.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                ret = true;
            }
            return ret;
        }
        private static bool IsArgsMatch(IList<string> args, FuncInfo funcInfo, out int score)
        {
            bool ret = false;
            score = 0;
            if (args.Count <= funcInfo.Params.Count) {
                ret = true;
                for (int ix = 0; ix < args.Count; ++ix) {
                    int argScore;
                    if (!IsTypeMatch(args[ix], funcInfo.Params[ix].Type, out argScore)) {
                        ret = false;
                        break;
                    }
                    score += argScore;
                }
                if (ret && args.Count < funcInfo.Params.Count) {
                    if (null == funcInfo.Params[args.Count].DefaultValueSyntax)
                        ret = false;
                }
            }
            return ret;
        }
        private static bool IsTypeMatch(string argType, string paramType, out int score)
        {
            bool ret = false;
            score = 0;
            if (argType == paramType) {
                score = 2;
                ret = true;
            }
            else if ((argType == "bool" || argType == "int" || argType == "uint" || argType == "dword" || argType == "float" || argType == "double" || argType == "half")
                && (paramType == "bool" || paramType == "int" || paramType == "uint" || paramType == "dword" || paramType == "float" || paramType == "double" || paramType == "half")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bool2" || argType == "int2" || argType == "uint2" || argType == "dword2" || argType == "float2" || argType == "double2" || argType == "half2")
                && (paramType == "bool2" || paramType == "int2" || paramType == "uint2" || paramType == "dword2" || paramType == "float2" || paramType == "double2" || paramType == "half2")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bool3" || argType == "int3" || argType == "uint3" || argType == "dword3" || argType == "float3" || argType == "double3" || argType == "half3")
                && (paramType == "bool3" || paramType == "int3" || paramType == "uint3" || paramType == "dword3" || paramType == "float3" || paramType == "double3" || paramType == "half3")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bool4" || argType == "int4" || argType == "uint4" || argType == "dword4" || argType == "float4" || argType == "double4" || argType == "half4")
                && (paramType == "bool4" || paramType == "int4" || paramType == "uint4" || paramType == "dword4" || paramType == "float4" || paramType == "double4" || paramType == "half4")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bool2x2" || argType == "int2x2" || argType == "uint2x2" || argType == "dword2x2" || argType == "float2x2" || argType == "double2x2" || argType == "half2x2")
                && (paramType == "bool2x2" || paramType == "int2x2" || paramType == "uint2x2" || paramType == "dword2x2" || paramType == "float2x2" || paramType == "double2x2" || paramType == "half2x2")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bool3x3" || argType == "int3x3" || argType == "uint3x3" || argType == "dword3x3" || argType == "float3x3" || argType == "double3x3" || argType == "half3x3")
                && (paramType == "bool3x3" || paramType == "int3x3" || paramType == "uint3x3" || paramType == "dword3x3" || paramType == "float3x3" || paramType == "double3x3" || paramType == "half3x3")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bool4x4" || argType == "int4x4" || argType == "uint4x4" || argType == "dword4x4" || argType == "float4x4" || argType == "double4x4" || argType == "half4x4")
                && (paramType == "bool4x4" || paramType == "int4x4" || paramType == "uint4x4" || paramType == "dword4x4" || paramType == "float4x4" || paramType == "double4x4" || paramType == "half4x4")) {
                score = 1;
                ret = true;
            }
            else if (argType == "bool" || argType == "bool2" || argType == "bool3" || argType == "bool4") {
                ret = true;
            }
            else if (argType == "int" || argType == "int2" || argType == "int3" || argType == "int4") {
                ret = true;
            }
            else if (argType == "uint" || argType == "uint2" || argType == "uint3" || argType == "uint4") {
                ret = true;
            }
            else if (argType == "dword" || argType == "dword2" || argType == "dword3" || argType == "dword4") {
                ret = true;
            }
            else if(argType == "float" || argType== "float2" || argType == "float3" || argType == "float4") {
                ret = true;
            }
            else if (argType == "double" || argType == "double2" || argType == "double3" || argType == "double4") {
                ret = true;
            }
            else if (argType == "half" || argType == "half2" || argType == "half3" || argType == "half4") {
                ret = true;
            }
            return ret;
        }
        private static string GetTypeNoVecPrefix(string type)
        {
            return GetTypeNoVecPrefix(type, out var isVecBefore);
        }
        private static string GetTypeNoVecPrefix(string type, out bool isVecBefore)
        {
            isVecBefore = false;
            if (type.StartsWith(c_VectorialTypePrefix)) {
                type = type.Substring(c_VectorialTypePrefix.Length);
                isVecBefore = true;
            }
            return type;
        }        
        private static bool IsTypeVec(string type)
        {
            return IsTypeVec(type, out var isTuple, out var isStruct);
        }
        private static bool IsTypeVec(string type, out bool isTuple, out bool isStruct)
        {
            isTuple = false;
            isStruct = false;
            bool ret = false;
            if (type.StartsWith(c_TupleTypePrefix)) {
                isTuple = true;
                if (type.Contains(c_VectorialTupleTypeTag)) {
                    ret = true;
                }
            }
            else {
                var struName = GetTypeNoVecPrefix(type, out ret);
                if (s_StructInfos.TryGetValue(struName, out var struInfo)) {
                    isStruct = true;
                }
            }
            return ret;
        }
        private static string GetTypeVec(string type)
        {
            return GetTypeVec(type, out var isTuple, out var isStruct, out var isvec);
        }
        private static string GetTypeVec(string type, out bool isTuple, out bool isStruct, out bool isVecBefore)
        {
            isTuple = false;
            isStruct = false;
            isVecBefore = false;
            if (type.StartsWith(c_TupleTypePrefix)) {
                isTuple = true;
                if (type.Contains(c_VectorialTupleTypeTag)) {
                    isVecBefore = true;
                }
            }
            else {
                var struName = GetTypeNoVecPrefix(type, out isVecBefore);
                if (s_StructInfos.TryGetValue(struName, out var struInfo)) {
                    isStruct = true;
                }
                if (!isVecBefore)
                    type = c_VectorialTypePrefix + struName;
            }
            return type;
        }
        private static string GetTypeNoVec(string type)
        {
            return GetTypeNoVec(type, out var isTuple, out var isStruct, out var isvec);
        }
        private static string GetTypeNoVec(string type, out bool isTuple, out bool isStruct, out bool isVecBefore)
        {
            isTuple = false;
            isStruct = false;
            isVecBefore = false;
            if (type.StartsWith(c_TupleTypePrefix)) {
                isTuple = true;
                if (type.Contains(c_VectorialTupleTypeTag)) {
                    isVecBefore = true;
                }
            }
            else {
                var struName = GetTypeNoVecPrefix(type, out isVecBefore);
                if (s_StructInfos.TryGetValue(struName, out var struInfo)) {
                    isStruct = true;
                }
                type = struName;
            }
            return type;
        }
        private static string GetTypeRemoveSuffix(string type)
        {
            return GetTypeRemoveSuffix(type, out var isTuple, out var isStruct, out var isVec, out var arrNums, out var typeWithoutArrTag);
        }
        private static string GetTypeRemoveSuffix(string type, out bool isTuple, out bool isStruct, out bool isVec, out IList<int> arrNums, out string typeWithoutArrTag)
        {
            typeWithoutArrTag = GetTypeRemoveArrTag(type, out isTuple, out isStruct, out isVec, out arrNums);
            if (isTuple || isStruct) {
                return typeWithoutArrTag;
            }
            if (typeWithoutArrTag.Length >= 3) {
                char last = typeWithoutArrTag[typeWithoutArrTag.Length - 1];
                if (last == '2' || last == '3' || last == '4') {
                    string last3 = typeWithoutArrTag.Substring(typeWithoutArrTag.Length - 3);
                    if (last3 == "2x2" || last3 == "3x3" || last3 == "4x4")
                        return typeWithoutArrTag.Substring(0, typeWithoutArrTag.Length - 3);
                    else if (last3 == "2x3" || last3 == "3x2" || last3 == "3x4" || last3 == "4x3" || last3 == "2x4" || last3 == "4x2")
                        return typeWithoutArrTag.Substring(0, typeWithoutArrTag.Length - 3);
                    else
                        return typeWithoutArrTag.Substring(0, typeWithoutArrTag.Length - 1);
                }
            }
            return typeWithoutArrTag;
        }
        private static string GetTypeSuffix(string type)
        {
            return GetTypeSuffix(type, out var isTuple, out var isStruct, out var IsArr, out var arrNums, out var typeWithoutArrTag);
        }
        private static string GetTypeSuffix(string type, out bool isTuple, out bool isStruct, out bool isVec, out IList<int> arrNums, out string typeWithoutArrTag)
        {
            typeWithoutArrTag = GetTypeRemoveArrTag(type, out isTuple, out isStruct, out isVec, out arrNums);
            if (isTuple || isStruct) {
                return string.Empty;
            }
            if (typeWithoutArrTag.Length >= 3) {
                char last = typeWithoutArrTag[typeWithoutArrTag.Length - 1];
                if (last == '2' || last == '3' || last == '4') {
                    string last3 = typeWithoutArrTag.Substring(typeWithoutArrTag.Length - 3);
                    if (last3 == "2x2" || last3 == "3x3" || last3 == "4x4") {
                        return last3;
                    }
                    else if (last3 == "2x3" || last3 == "3x2" || last3 == "3x4" || last3 == "4x3" || last3 == "2x4" || last3 == "4x2") {
                        return last3;
                    }
                    else {
                        return last.ToString();
                    }
                }
            }
            return string.Empty;
        }
        private static string GetTypeSuffixReverse(string type)
        {
            return GetTypeSuffixReverse(type, out var isTuple, out var isStruct, out var IsArr, out var arrNums, out var typeWithoutArrTag);
        }
        private static string GetTypeSuffixReverse(string type, out bool isTuple, out bool isStruct, out bool isVec, out IList<int> arrNums, out string typeWithoutArrTag)
        {
            typeWithoutArrTag = GetTypeRemoveArrTag(type, out isTuple, out isStruct, out isVec, out arrNums);
            if(isTuple || isStruct) {
                return string.Empty;
            }
            if (typeWithoutArrTag.Length >= 3) {
                char last = typeWithoutArrTag[typeWithoutArrTag.Length - 1];
                if (last == '2' || last == '3' || last == '4') {
                    string last3 = typeWithoutArrTag.Substring(typeWithoutArrTag.Length - 3);
                    if (last3 == "2x2" || last3 == "3x3" || last3 == "4x4") {
                        return last3;
                    }
                    else if (last3 == "2x3" || last3 == "3x2" || last3 == "3x4" || last3 == "4x3" || last3 == "2x4" || last3 == "4x2") {
                        return new string(new char[] { last3[2], last3[1], last3[0] });
                    }
                    else {
                        return last.ToString();
                    }
                }
            }
            return string.Empty;
        }
        private static string GetTypeRemoveArrTag(string type, out bool isTuple, out bool isStruct, out bool isVec, out IList<int> arrNums)
        {
            var list = new List<int>();
            isVec = IsTypeVec(type, out isTuple, out isStruct);
            if (isTuple) {
                arrNums = list;
                return type;
            }
            var r = GetTypeRemoveArrTagRecursively(type, list);
            arrNums = list;
            return r;
        }
        private static string GetTypeRemoveArrTagRecursively(string type, List<int> arrNums)
        {
            int st = type.LastIndexOf("_x");
            if (st > 0) {
                string arrNumStr = type.Substring(st + 2);
                if (int.TryParse(arrNumStr, out int arrNum)) {
                    arrNums.Add(arrNum);
                    type = GetTypeRemoveArrTagRecursively(type.Substring(0, st), arrNums);
                }
            }
            return type;
        }
        private static int GetMemberCount(string member)
        {
            int ct = 0;
            foreach (char c in member) {
                if (c == '_')
                    ++ct;
            }
            return ct;
        }
        private static string GetNamespaceName(Dsl.FunctionData func)
        {
            if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PERIOD) {
                string m = func.GetParamId(0);
                if (func.IsHighOrder) {
                    return GetNamespaceName(func.LowerOrderFunction) + "." + m;
                }
                else {
                    return func.GetId() + "." + m;
                }
            }
            else {
                if (func.IsHighOrder) {
                    return GetNamespaceName(func.LowerOrderFunction);
                }
                else {
                    return func.GetId();
                }
            }
        }
        private static void MarkBranch()
        {
            var funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                funcInfo.HasBranches = true;
            }
        }
        private static string SwizzleConvert(string m)
        {
            string v = m.Replace('r', 'x').Replace('g', 'y').Replace('b', 'z').Replace('a', 'w');
            v = v.Replace("_11", "_m00").Replace("_12", "_m01").Replace("_13", "_m02").Replace("_14", "_m03");
            v = v.Replace("_21", "_m10").Replace("_22", "_m11").Replace("_23", "_m12").Replace("_24", "_m13");
            v = v.Replace("_31", "_m20").Replace("_32", "_m21").Replace("_33", "_m22").Replace("_34", "_m23");
            v = v.Replace("_41", "_m30").Replace("_42", "_m31").Replace("_43", "_m32").Replace("_44", "_m33");
            return v;
        }


        // TODO: This only considers the case where variables are assigned as a whole. For assignments to matrices,
        // arrays, and structure members, there may also be cases where value types become reference types after
        // vectorization. Currently, these have not been handled.
        // The reason why it is difficult to handle is that we need to allow set operations on the members obtained
        // by get, so get actually needs to support returning references (which may require a copy-on-write mechanism).
        // In addition, for swizzle operations such as float3.x, there may also be cases where value types become
        // reference types after vectorization. Currently, this is handled in the swizzle implementation, where a new
        // array is copied and returned for each xyzw returned after vectorization (although semantically consistent
        // with the original shader, it will have certain performance impact after vectorization).
        private static bool ExistsSetObj(string vname)
        {
            bool ret = false;
            /*
            var vinfo = GetVarInfo(vname, VarUsage.Find);
            if (null != vinfo) {
                var blockInfo = vinfo.OwnerBlock;
                if (null != blockInfo) {
                    ret = blockInfo.ExistsSetObjInBlockScope(0, 0, vname, false, blockInfo);
                }
                else if(vinfo.IsConst) {
                    ret = false;
                }
                else {
                    ret = true;
                }
            }
            */
            var blockInfo = CurBlockInfo();
            if (null != blockInfo) {
                ret = blockInfo.ExistsSetObjInCurBlockScope(vname);
            }
            else {
                var vinfo = GetVarInfo(vname, VarUsage.Find);
                if (null != vinfo) {
                    if (vinfo.IsConst) {
                        ret = false;
                    }
                    else {
                        ret = true;
                    }
                }
            }
            return ret;
        }
        private static bool IsHlslBool(string val)
        {
            return val == "true" || val == "false";
        }
        private static string ConstToPython(string constVal)
        {
            if (constVal == "true")
                return "True";
            if (constVal == "false")
                return "False";
            return constVal;
        }
        private static string GetDefaultValueInPython(string type)
        {
            if (type == "bool")
                return "False";
            else if (type == "int" || type == "uint" || type == "dword")
                return "0";
            else if (type == "float" || type == "double" || type == "half")
                return "0.0";
            else {
                string typeWithoutArrTag = GetTypeRemoveArrTag(type, out var isTuple, out var isStruct, out var isVec, out var arrNums);
                if (isStruct) {
                    string fn = GetSimpleArrayTypeAbbr(type) + "_defval";
                    GenOrRecordDefValFunc(fn, typeWithoutArrTag, isTuple, isStruct, isVec, arrNums, Dsl.AbstractSyntaxComponent.NullSyntax);
                    return fn + "()";
                }
                else if (arrNums.Count > 0) {
                    string fn = "h_" + GetSimpleArrayTypeAbbr(type) + "_defval";
                    GenOrRecordDefValFunc(fn, typeWithoutArrTag, isTuple, isStruct, isVec, arrNums, Dsl.AbstractSyntaxComponent.NullSyntax);
                    return fn + "(" + string.Join(", ", arrNums) + ")";
                }
                else {
                    string suffix = GetTypeSuffix(type);
                    if (suffix.Length > 0) {
                        return "h_" + GetTypeAbbr(type) + "_defval()";
                    }
                    else {
                        return "None";
                    }
                }
            }
        }
        private static string HlslType2Python(string type)
        {
            if (s_IsTorch) {
                if (!s_HlslType2TorchTypes.TryGetValue(type, out var ty)) {
                    s_HlslType2TorchTypes.TryGetValue("float", out ty);
                }
                Debug.Assert(null != ty);
                return ty;
            }
            else {
                if (!s_HlslType2NumpyTypes.TryGetValue(type, out var ty)) {
                    s_HlslType2NumpyTypes.TryGetValue("float", out ty);
                }
                Debug.Assert(null != ty);
                return ty;
            }
        }

    }
}