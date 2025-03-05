using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Text;

namespace Hlsl2Python
{
    internal partial class Program
    {
        private static void LoadShaderArgs(string argFilePath)
        {
            var cfgFile = new Dsl.DslFile();
            if(cfgFile.Load(argFilePath, msg => { Console.WriteLine(msg); })) {
                foreach (var cfg in cfgFile.DslInfos) {
                    string id = cfg.GetId();
                    var fd = cfg as Dsl.FunctionData;
                    if (null != fd) {
                        if (id == "facecolor") {
                            s_MainShaderInfo.FaceColor = fd.GetParamId(0);
                        }
                        else if (id == "winzoom") {
                            float.TryParse(fd.GetParamId(0), out s_MainShaderInfo.WinZoom);
                        }
                        else if (id == "winsize") {
                            float.TryParse(fd.GetParamId(0), out s_MainShaderInfo.WinSize);
                        }
                        else if (id == "addbuffer") {
                            if (fd.IsHighOrder) {
                                string bufId = fd.LowerOrderFunction.GetParamId(0);
                                if(s_ShaderToyBufferNames.Contains(bufId)) {
                                    var bufInfo = new ShaderBufferInfo();
                                    bufInfo.BufferId = bufId;
                                    s_ShaderBufferInfos.Add(bufInfo);

                                    foreach (var bufArg in fd.Params) {
                                        var bufArgFd = bufArg as Dsl.FunctionData;
                                        if (null != bufArgFd) {
                                            SetShaderBufferInfo(bufInfo, true, bufArgFd);
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            SetShaderBufferInfo(s_MainShaderInfo, false, fd);
                        }
                    }
                }
            }
        }
        private static void SetShaderBufferInfo(ShaderBufferInfo shaderInfo, bool isBuffer, Dsl.FunctionData bufArg)
        {
            string argId = bufArg.GetId();
            if (argId == "shaderargs") {
                foreach (var p in bufArg.Params) {
                    var pfd = p as Dsl.FunctionData;
                    if (null != pfd && pfd.GetParamNum() == 2) {
                        var argInfo = new ShaderArgInfo();
                        argInfo.ArgType = pfd.GetId();
                        argInfo.ArgName = pfd.GetParamId(0);
                        argInfo.ArgValue = pfd.GetParamId(1);
                        shaderInfo.ArgInfos.Add(argInfo);
                    }
                }
            }
            else if (argId == "resolution") {
                float.TryParse(bufArg.GetParamId(0), out var x);
                float.TryParse(bufArg.GetParamId(1), out var y);
                float.TryParse(bufArg.GetParamId(2), out var z);
                shaderInfo.Resolution = new Float3 { x = x, y = y, z = z };
            }
            else if (argId == "resolution_on_full_vec") {
                float.TryParse(bufArg.GetParamId(0), out var x);
                float.TryParse(bufArg.GetParamId(1), out var y);
                float.TryParse(bufArg.GetParamId(2), out var z);
                shaderInfo.ResolutionOnFullVec = new Float3 { x = x, y = y, z = z };
            }
            else if (argId == "resolution_on_gpu_full_vec") {
                float.TryParse(bufArg.GetParamId(0), out var x);
                float.TryParse(bufArg.GetParamId(1), out var y);
                float.TryParse(bufArg.GetParamId(2), out var z);
                shaderInfo.ResolutionOnGpuFullVec = new Float3 { x = x, y = y, z = z };
            }
            else if (argId == "init_mouse_pos") {
                float.TryParse(bufArg.GetParamId(0), out var x);
                float.TryParse(bufArg.GetParamId(1), out var y);
                float.TryParse(bufArg.GetParamId(2), out var z);
                float.TryParse(bufArg.GetParamId(3), out var w);
                shaderInfo.InitMousePos = new Float4 { x = x, y = y, z = z, w = w };
            }
            else if (isBuffer || !s_CmdArgKeys.Contains(argId)) {
                if (argId == "entry") {
                    shaderInfo.Entry = bufArg.GetParamId(0);
                }
                else if (argId == "tex2d0" || argId == "tex2d1" || argId == "tex2d2" || argId == "tex2d3" ||
                    argId == "tex3d0" || argId == "tex3d1" || argId == "tex3d2" || argId == "tex3d3" ||
                    argId == "texcube0" || argId == "texcube1" || argId == "texcube2" || argId == "texcube3") {
                    SetShaderToyTex(shaderInfo, argId, bufArg.GetParamId(0));
                }
            }
        }
        private static void SetShaderToyTex(ShaderBufferInfo bufInfo, string key, string arg)
        {
            var texFilePath = arg;
            if (!s_ShaderToyBufferNames.Contains(arg)) {
                if (!File.Exists(texFilePath)) {
                    Console.WriteLine("file path not found ! {0}", texFilePath);
                }
            }
            string ty = key.Substring(key.Length - 3, 2);
            int texIx = key[key.Length - 1] - '0';
            string texName = s_ShaderToyTexNamePrefix + texIx.ToString();
            if (ty == "2d") {
                bufInfo.TexTypes[texName] = "sampler2D";
            }
            else if (ty == "3d") {
                bufInfo.TexTypes[texName] = "sampler3D";
            }
            else {
                bufInfo.TexTypes[texName] = "samplerCube";
            }
            if (s_ShaderToyBufferNames.Contains(arg)) {
                bufInfo.TexBuffers[texName] = arg;
            }
            else {
                bufInfo.TexFiles[texName] = texFilePath;
            }
        }

        private struct Float2
        {
            internal float x = 0.0f;
            internal float y = 0.0f;

            public Float2()
            { }
        }
        private struct Float3
        {
            internal float x = 0.0f;
            internal float y = 0.0f;
            internal float z = 0.0f;

            public Float3()
            { }
        }
        private struct Float4
        {
            internal float x = 0.0f;
            internal float y = 0.0f;
            internal float z = 0.0f;
            internal float w = 0.0f;

            public Float4()
            { }
        }
        private sealed class ShaderArgInfo
        {
            internal string ArgName = string.Empty;
            internal string ArgType = string.Empty;
            internal string ArgValue = string.Empty;
        }
        private class ShaderBufferInfo
        {
            internal string BufferId = string.Empty;
            internal string Entry = string.Empty;
            internal Float3 Resolution = new Float3 { x = 32, y = 24, z = 1 };
            internal Float3 ResolutionOnFullVec = new Float3 { x = 160, y = 120, z = 1 };
            internal Float3 ResolutionOnGpuFullVec = new Float3 { x = 640, y = 480, z = 1 };
            internal Float4 InitMousePos = new Float4 { x = 0.5f, y = 0.5f, z = 0.0f, w = 0.0f };
            internal SortedDictionary<string, string> TexTypes = new SortedDictionary<string, string>();
            internal SortedDictionary<string, string> TexFiles = new SortedDictionary<string, string>();
            internal SortedDictionary<string, string> TexBuffers = new SortedDictionary<string, string>();
            internal List<ShaderArgInfo> ArgInfos = new List<ShaderArgInfo>();
        }
        private sealed class ShaderCanvasInfo : ShaderBufferInfo
        {
            internal string FaceColor = "gray";
            internal float WinZoom = 1.0f;
            internal float WinSize = 0.0f;
        }

        private static string s_ShaderToyTexNamePrefix = "iChannel";
        private static HashSet<string> s_ShaderToyParamNames = new HashSet<string> {
            "iChannel0",
            "iChannel1",
            "iChannel2",
            "iChannel3",
            "iResolution",
            "iTime",
            "iTimeDelta",
            "iFrameRate",
            "iFrame",
            "iChannelTime",
            "iChannelResolution",
            "iMouse",
            "iDate",
            "iSampleRate",
        };
        private static HashSet<string> s_ShaderToyBufferNames = new HashSet<string> {
            "bufferA",
            "bufferB",
            "bufferC",
            "bufferD",
            "bufferCubemap",
            "bufferSound",
        };
        private static List<string> s_ShaderToyChannels = new List<string> {
            "iChannel0",
            "iChannel1",
            "iChannel2",
            "iChannel3",
        };
        private static SortedDictionary<string, string> s_ShaderToyTexTypes = new SortedDictionary<string, string> {
            { s_ShaderToyChannels[0], "sampler2D" },
            { s_ShaderToyChannels[1], "sampler2D" },
            { s_ShaderToyChannels[2], "sampler2D" },
            { s_ShaderToyChannels[3], "sampler2D" },
        };
        private static SortedDictionary<string, string> s_ShaderToyTexFiles = new SortedDictionary<string, string> {
            { s_ShaderToyChannels[0], "shaderlib/noise1.jpg" },
            { s_ShaderToyChannels[1], "shaderlib/noise2.jpg" },
            { s_ShaderToyChannels[2], "shaderlib/noise3.jpg" },
            { s_ShaderToyChannels[3], "shaderlib/noise4.jpg" },
        };

        private static HashSet<string> s_CmdArgKeys = new HashSet<string>();
        private static ShaderCanvasInfo s_MainShaderInfo = new ShaderCanvasInfo { TexTypes = s_ShaderToyTexTypes, TexFiles = s_ShaderToyTexFiles };
        private static List<ShaderBufferInfo> s_ShaderBufferInfos = new List<ShaderBufferInfo>();

        private static SortedDictionary<string, string> s_EmptyDictionarys = new SortedDictionary<string, string>();
    }
}
