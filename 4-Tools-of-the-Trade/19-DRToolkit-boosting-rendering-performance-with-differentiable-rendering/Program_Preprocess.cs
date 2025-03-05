using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace Hlsl2Python
{
    internal partial class Program
    {
        private static void PreprocessGlsl(string srcFile, string outFile, bool fromShaderToy)
        {
            File.Delete(outFile);
            var glslLines = new List<string>();
            if (fromShaderToy) {
                glslLines.Add(string.Empty);
                glslLines.Add("vec3 iResolution = vec3(1.0, 1.0, 1.0);");
                glslLines.Add("float iTime = 0.0;");
                glslLines.Add("float iTimeDelta = 0.0;");
                glslLines.Add("float iFrameRate = 10.0;");
                glslLines.Add("int iFrame = 0;");
                glslLines.Add("float iChannelTime[4] = float[4](0.0, 0.0, 0.0, 0.0);");
                glslLines.Add("vec3 iChannelResolution[4] = vec3[4](vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0));");
                glslLines.Add("vec4 iMouse = vec4(0.0, 0.0, 0.0, 0.0);");
                glslLines.Add("vec4 iDate = vec4(0.0, 0.0, 0.0, 0.0);");
                glslLines.Add("float iSampleRate = 44100.0;");
                foreach(var pair in s_MainShaderInfo.TexTypes) {
                    glslLines.Add(string.Format("{0} {1};", pair.Value, pair.Key));
                }
                foreach(var bufInfo in s_ShaderBufferInfos) {
                    foreach(var pair in bufInfo.TexTypes) {
                        glslLines.Add(string.Format("{0} {1}_{2};", pair.Value, bufInfo.BufferId, pair.Key));
                    }
                }
                glslLines.Add(string.Empty);
            }
            var glslFileLines = File.ReadAllLines(srcFile);
            glslLines.AddRange(glslFileLines);
            string glslTxt = PreprocessGlslCondExp(glslLines);

            var file = new Dsl.DslFile();
            file.onGetToken = (ref Dsl.Common.DslAction dslAction, ref Dsl.Common.DslToken dslToken, ref string tok, ref short val, ref int line) => {
                if (tok == "return") {
                    var oldCurTok = dslToken.getCurToken();
                    var oldLastTok = dslToken.getLastToken();
                    if (dslToken.PeekNextValidChar(0) == ';')
                        return false;
                    dslToken.setCurToken("<-");
                    dslToken.setLastToken(oldCurTok);
                    dslToken.enqueueToken(dslToken.getCurToken(), dslToken.getOperatorTokenValue(), line);
                    dslToken.setCurToken(oldCurTok);
                    dslToken.setLastToken(oldLastTok);
                    return true;
                }
                return false;
            };
            file.onBeforeAddFunction = (ref Dsl.Common.DslAction dslAction, Dsl.StatementData statement) => {
                string sid = statement.GetId();
                var func = statement.Last.AsFunction;
                if (null != func) {
                    if (sid == "hlsl_attr" && statement.GetFunctionNum() >= 2) {
                        sid = statement.Second.GetId();
                    }
                    if (func.HaveStatement()) {
                        if(string.IsNullOrEmpty(sid) || sid == "for" || sid == "while" || sid == "else" || sid == "switch") {
                            //End the current statement and start a new empty statement.
                            dslAction.endStatement();
                            dslAction.beginStatement();
                            return true;
                        }
                    }
                    else {
                        if (sid == "do") {
                            //End the current statement and start a new empty statement.
                            dslAction.endStatement();
                            dslAction.beginStatement();
                            return true;
                        }
                    }
                }
                return false;
            };
            file.onAddFunction = (ref Dsl.Common.DslAction dslAction, Dsl.StatementData statement, Dsl.FunctionData function) => {
                //Do not change the program structure at this point. The function is still an empty function, and the actual function
                //information has not been filled in yet. The difference between this and onBeforeAddFunction is that at this point, the
                //function is constructed and added to the function table of the current statement.
                return false;
            };
            file.onBeforeEndStatement = (ref Dsl.Common.DslAction dslAction, Dsl.StatementData statement) => {
                //Here, the statement can be split
                return false;
            };
            file.onEndStatement = (ref Dsl.Common.DslAction dslAction, ref Dsl.StatementData statement) => {
                //Here, the entire statement can be replaced, but do not modify the structure of other parts of the program. The difference
                //between this and onBeforeEndStatement is that at this point, the statement has been popped off the stack and will be
                //simplified and added to the upper syntax unit later.
                return false;
            };
            file.onBeforeBuildOperator = (ref Dsl.Common.DslAction dslAction, string op, Dsl.StatementData statement) => {
                //Split the statement here.
                string sid = statement.GetId();
                var func = statement.Last.AsFunction;
                if (null != func) {
                    if (sid == "hlsl_attr" && statement.GetFunctionNum() > 2) {
                        sid = statement.Second.GetId();
                    }
                    if (sid == "if") {
                        statement.Functions.Remove(func);
                        dslAction.endStatement();
                        dslAction.beginStatement();
                        var stm = dslAction.getCurStatement();
                        stm.AddFunction(func);
                        return true;
                    }
                }
                return false;
            };
            file.onBuildOperator = (ref Dsl.Common.DslAction dslAction, string op, ref Dsl.StatementData statement) => {
                //Replace the statement here without modifying other syntax structures.
                return false;
            };
            file.onSetFunctionId = (ref Dsl.Common.DslAction dslAction, string name, Dsl.StatementData statement, Dsl.FunctionData function) => {
                //Here, the statement can be split
                string sid = statement.GetId();
                var func = statement.Last.AsFunction;
                if (null != func) {
                    if (sid == "hlsl_attr" && statement.GetFunctionNum() == 2 && name != "for" && name != "while" && name != "do" && name != "if" && name != "else" && name != "switch") {
                        statement.Functions.Remove(func);
                        dslAction.endStatement();
                        dslAction.beginStatement();
                        var stm = dslAction.getCurStatement();
                        stm.AddFunction(func);
                        return true;
                    }
                    if (sid == "hlsl_attr" && statement.GetFunctionNum() > 2) {
                        sid = statement.Second.GetId();
                    }
                    if (sid == "if" && name != "else") {
                        statement.Functions.Remove(func);
                        dslAction.endStatement();
                        dslAction.beginStatement();
                        var stm = dslAction.getCurStatement();
                        stm.AddFunction(func);
                        return true;
                    }
                    else if (name == "struct") {
                        statement.Functions.Remove(func);
                        dslAction.endStatement();
                        dslAction.beginStatement();
                        var stm = dslAction.getCurStatement();
                        stm.AddFunction(func);
                        return true;
                    }
                }
                return false;
            };
            file.onSetMemberId = (ref Dsl.Common.DslAction dslAction, string name, Dsl.StatementData statement, Dsl.FunctionData function) => {
                //Here, the statement can be split
                return false;
            };
            file.onBeforeBuildHighOrder = (ref Dsl.Common.DslAction dslAction, Dsl.StatementData statement, Dsl.FunctionData function) => {
                //Here, the statement can be split
                return false;
            };
            file.onBuildHighOrder = (ref Dsl.Common.DslAction dslAction, Dsl.StatementData statement, Dsl.FunctionData function) => {
                //Here, the statement can be split
                return false;
            };
            if (file.LoadFromString(glslTxt, msg => { Console.WriteLine(msg); })) {
                //The syntax of GLSL is a valid DSL syntax, but the semantic structure is different. We are only dealing
                //with minor differences between GLSL and HLSL, so we can make do with the DSL representation for
                //processing. The transformations we need to perform mainly include replacing single-parameter
                //constructors like vec3(float) and matrix multiplication operations with mul(mat, vec). Although
                //these are relatively simple syntax changes, they also rely on type information. Therefore, this also
                //includes variable lexical scope analysis and simplified type deduction.
                TransformGlsl(file);
                file.Save(outFile);
            }
            else {
                Environment.Exit(-1);
            }

            var sb = NewStringBuilder();
            var lineList = new List<string>();
            lineList.Add("#include \"shaderlib/glsl.h\"");
            if (fromShaderToy) {
                lineList.Add("static SamplerState s_linear_clamp_sampler;");
            }
            foreach (var pair in s_GlslStructInfos) {
                string struName = pair.Key;
                var struInfo = pair.Value;

                sb.Length = 0;
                sb.AppendFormat("struct {0};", struName);
                lineList.Add(sb.ToString());

                sb.Length = 0;
                sb.AppendFormat("{0} glsl_{0}_ctor(", struName);
                string prestr = string.Empty;
                foreach (var fi in struInfo.Fields) {
                    sb.Append(prestr);
                    sb.Append(TypeToShaderType(fi.Type, out var arraySuffix));
                    sb.Append(arraySuffix);
                    prestr = ", ";
                }
                sb.Append(");");
                lineList.Add(sb.ToString());
            }
            foreach(var pair in s_ArrayInits) {
                string fn = pair.Key;
                var arrInitInfo = pair.Value;

                sb.Length = 0;
                sb.AppendFormat("typedef {0} td_{0}_x{1}[{1}];", arrInitInfo.Type, arrInitInfo.Size);
                lineList.Add(sb.ToString());

                sb.Length = 0;
                sb.AppendFormat("td_{0}_x{1} {2}(", arrInitInfo.Type, arrInitInfo.Size, fn);
                string prestr = string.Empty;
                for (int ct = 0; ct < arrInitInfo.Size; ++ct) {
                    sb.Append(prestr);
                    sb.Append(arrInitInfo.Type);
                    prestr = ", ";
                }
                sb.Append(");");
                lineList.Add(sb.ToString());
            }
            var lines = File.ReadAllLines(outFile);
            lineList.AddRange(lines);
            foreach (var pair in s_GlslStructInfos) {
                string struName = pair.Key;
                var struInfo = pair.Value;
                sb.Length = 0;
                sb.AppendFormat("{0} glsl_{0}_ctor(", struName);
                string prestr = string.Empty;
                foreach (var fi in struInfo.Fields) {
                    sb.Append(prestr);
                    sb.Append(TypeToShaderType(fi.Type, out var arraySuffix));
                    sb.Append(' ');
                    sb.Append('_');
                    sb.Append(fi.Name.ToLower());
                    sb.Append(arraySuffix);
                    prestr = ", ";
                }
                sb.Append(") {");
                lineList.Add(sb.ToString());

                sb.Length = 0;
                sb.AppendFormat("\t{0} __stru_tmp = {{", struName);
                prestr = string.Empty;
                foreach (var fi in struInfo.Fields) {
                    sb.Append(prestr);
                    sb.Append('_');
                    sb.Append(fi.Name.ToLower());
                    prestr = ", ";
                }
                sb.Append("};");
                lineList.Add(sb.ToString());

                lineList.Add("\treturn __stru_tmp;");
                lineList.Add("}");
            }
            foreach(var pair in s_ArrayInits) {
                string fn = pair.Key;
                var arrInitInfo = pair.Value;

                sb.Length = 0;
                sb.AppendFormat("td_{0}_x{1} {2}(", arrInitInfo.Type, arrInitInfo.Size, fn);
                string prestr = string.Empty;
                for (int ct = 0; ct < arrInitInfo.Size; ++ct) {
                    sb.Append(prestr);
                    sb.Append(arrInitInfo.Type);
                    sb.Append(" v");
                    sb.Append(ct);
                    prestr = ", ";
                }
                sb.Append(") {");
                lineList.Add(sb.ToString());

                sb.Length = 0;
                sb.AppendFormat("\t{0} __arr_tmp[{1}] = {{", arrInitInfo.Type, arrInitInfo.Size);
                prestr = string.Empty;
                for (int ct = 0; ct < arrInitInfo.Size; ++ct) {
                    sb.Append(prestr);
                    sb.Append('v');
                    sb.Append(ct);
                    prestr = ", ";
                }
                sb.Append("};");
                lineList.Add(sb.ToString());

                lineList.Add("\treturn __arr_tmp;");
                lineList.Add("}");
            }
            RecycleStringBuilder(sb);
            File.WriteAllLines(outFile, lineList.ToArray());
        }
        private static string PreprocessGlslCondExp(IList<string> glslLines)
        {
            //In C language, the ?: operator and the assignment operation have the same precedence, which is different
            //from MetaDSL. It is possible that an assignment expression may appear within a conditional expression. We
            //need to enclose these expressions in parentheses before performing DSL parsing.
            var sb = new StringBuilder();
            bool inCommentBlock = false;
            for (int ix = 0; ix < glslLines.Count; ++ix) {
                var line = glslLines[ix];
                int k = 0;
                if (inCommentBlock) {
                    for (int i = 0; i < line.Length; ++i) {
                        char ch = line[i];
                        if (ch == '*' && i < line.Length - 1 && line[i + 1] == '/') {
                            inCommentBlock = false;
                            k = i + 2;
                            break;
                        }
                    }
                    if (inCommentBlock)
                        continue;
                }
                var rline = line.Substring(k);
                if (rline.StartsWith("#line")) {
                    //ignore
                }
                else {
                    for (int i = k; i < line.Length; ++i) {
                        char c = line[i];
                        if (c == '\\') {
                            ++i;
                            sb.Append(c);
                            sb.Append(line[i]);
                        }
                        else if (s_GlslCondExpStack.Count > 0 && (c == '(' || c == '[' || c == '{')) {
                            s_GlslCondExpStack.Peek().IncParenthesisCount();
                            sb.Append(c);
                        }
                        else if (s_GlslCondExpStack.Count > 0 && (c == ')' || c == ']' || c == '}')) {
                            if (s_GlslCondExpStack.Peek().MaybeCompletePart(GlslCondExpEnum.Colon)) {
                                s_GlslCondExpStack.Pop();
                                sb.Append(')');
                            }
                            if (s_GlslCondExpStack.Count > 0) {
                                s_GlslCondExpStack.Peek().DecParenthesisCount();
                            }
                            sb.Append(c);
                        }
                        else if (c == '?') {
                            s_GlslCondExpStack.Push(new GlslCondExpInfo(GlslCondExpEnum.Question));
                            sb.Append(c);
                            sb.Append(' ');
                            sb.Append('(');
                        }
                        else if (c == ':') {
                            while (s_GlslCondExpStack.Count > 0 && s_GlslCondExpStack.Peek().MaybeCompletePart(GlslCondExpEnum.Colon)) {
                                s_GlslCondExpStack.Pop();
                                sb.Append(')');
                            }
                            if (s_GlslCondExpStack.Count > 0 && s_GlslCondExpStack.Peek().MaybeCompletePart(GlslCondExpEnum.Question)) {
                                s_GlslCondExpStack.Pop();
                                s_GlslCondExpStack.Push(new GlslCondExpInfo(GlslCondExpEnum.Colon));
                                sb.Append(')');
                                sb.Append(' ');
                                sb.Append(c);
                                sb.Append(' ');
                                sb.Append('(');
                            }
                            else {
                                sb.Append(c);
                            }
                        }
                        else if (c == ',' || c == ';') {
                            while (s_GlslCondExpStack.Count > 0 && s_GlslCondExpStack.Peek().MaybeCompletePart(GlslCondExpEnum.Colon)) {
                                s_GlslCondExpStack.Pop();
                                sb.Append(')');
                            }
                            sb.Append(c);
                        }
                        else if (c == '"') {
                            sb.Append(c);
                            for (int j = i + 1; j < line.Length; ++j) {
                                char ch = line[j];
                                sb.Append(ch);
                                if (ch == '\\') {
                                    ++j;
                                    sb.Append(line[j]);
                                }
                                else if (ch == '"') {
                                    i = j;
                                    break;
                                }
                            }
                        }
                        else {
                            char nc = i < line.Length - 1 ? line[i + 1] : '\0';
                            if (c == '/' && nc == '*') {
                                inCommentBlock = true;
                                for (int j = i + 2; j < line.Length; ++j) {
                                    char ch = line[j];
                                    if (ch == '*' && j < line.Length - 1 && line[j + 1] == '/') {
                                        inCommentBlock = false;
                                        i = j + 1;
                                        break;
                                    }
                                }
                                if (inCommentBlock)
                                    break;
                            }
                            else if (c == '/' && nc == '/') {
                                break;
                            }
                            else {
                                sb.Append(c);
                                char lc = i > 0 ? line[i - 1] : '\0';
                                if (c == '\'' && (!char.IsDigit(lc) || !char.IsDigit(nc))) {
                                    for (int j = i + 1; j < line.Length; ++j) {
                                        char ch = line[j];
                                        if (ch == '\\') {
                                            ++j;
                                            sb.Append(ch);
                                            sb.Append(line[j]);
                                        }
                                        else if (ch == '\'') {
                                            sb.Append(ch);
                                            i = j;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }
        private static void TransformGlsl(Dsl.DslFile file)
        {
            foreach (var dsl in file.DslInfos) {
                TransformGlslToplevelSyntax(dsl);
            }
        }
        private static void TransformGlslToplevelSyntax(Dsl.ISyntaxComponent syntax)
        {
            var valData = syntax as Dsl.ValueData;
            var funcData = syntax as Dsl.FunctionData;
            var stmData = syntax as Dsl.StatementData;
            if (null != valData) {
                TransformGlslVar(valData);
            }
            else if (null != funcData) {
                if (funcData.GetId() == "=") {
                    //In GLSL, the assignment statement corresponds directly to HLSL and is legal, so we do not need
                    //to transform it. Therefore, we can separate and process each part individually here.
                    var p = funcData.GetParam(0);
                    var vd = p as Dsl.ValueData;
                    var fd = p as Dsl.FunctionData;
                    var sd = p as Dsl.StatementData;
                    if (null != vd) {
                        TransformGlslVar(vd);
                    }
                    else if (null != fd) {
                        TransformGlslCall(fd);
                    }
                    else if (null != sd) {
                        //In DSL syntax, semicolons separate statements, while in GLSL, function definitions do not end
                        //with separators. During DSL parsing, the function definition is parsed to the left of the
                        //assignment statement's syntax part. Here, we need to split the function definition from the
                        //left part of the assignment statement to correctly analyze the function prototype and identify
                        //variable definitions within the assignment statement (but the overall representation should
                        //not be changed, otherwise the syntax may be incorrect when outputting to HLSL).
                        Dsl.StatementData? left, right;
                        if (SplitGlslStatementsInExpression(sd, out left, out right)) {
                            Debug.Assert(null != left && null != right);
                            string id = left.GetId();
                            int index = 0;
                            while (index < left.GetFunctionNum()) {
                                bool existsFunc = ParseGlslStatement(left, ref index, out Dsl.FunctionData? f, out List<string> modifiers);
                                if (existsFunc) {
                                    Debug.Assert(null != f);
                                    TransformGlslFuncDef(f, modifiers);
                                }
                                else {
                                    break;
                                }
                            }
                            //When splitting, "left" is the end of a compound statement, and by this point, "left" should
                            //have been processed completely.
                            TransformGlslVar(right, 0, true, sd);
                        }
                        else {
                            TransformGlslVar(sd, 0, true);
                        }
                    }
                    var v = funcData.GetParam(1);
                    TransformGlslSyntax(v, false, out var addStm);
                }
                else {
                    TransformGlslFunction(funcData);
                }
            }
            else if (null != stmData) {
                //In DSL syntax, semicolons separate statements, while in GLSL, function definitions do not end with
                //separators. During DSL parsing, the function definition is concatenated with subsequent function
                //definitions or struct/buffer/variable definitions to form a single large statement. Here, they need
                //to be separated for analysis (to ensure the correctness of the output HLSL, the overall representation
                //should not be modified).
                int index = 0;
                bool handled = false;
                while (index < stmData.GetFunctionNum()) {
                    int startIndex = index;
                    string id = stmData.GetFunctionId(startIndex);
                    bool existsFunc = ParseGlslToplevelStatement(stmData, ref index, out var f, out var layout, out var modifiers, out var varNamePart);
                    if (existsFunc) {
                        Debug.Assert(null != f);
                        if (id == "struct") {
                            TransformGlslStruct(f, varNamePart);
                            if (null != varNamePart) {
                                ++index;
                                var sta = new Dsl.ValueData("static");
                                stmData.Functions.Insert(startIndex, sta);
                                ++index;
                            }
                            else if (stmData.GetSeparator() == Dsl.AbstractSyntaxComponent.SEPARATOR_COMMA) {
                                var sta = new Dsl.ValueData("static");
                                stmData.Functions.Insert(startIndex, sta);
                                ++index;
                                stmData.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_NOTHING);
                            }
                        }
                        else if (modifiers.Contains("uniform") || modifiers.Contains("buffer")) {
                            TransformGlslBuffer(f, layout, varNamePart);
                            if (null != varNamePart) {
                                ++index;
                            }
                        }
                        else {
                            TransformGlslFuncDef(f, modifiers);
                        }
                        handled = true;
                    }
                    else {
                        handled = false;
                        break;
                    }
                }
                if (!handled) {
                    TransformGlslVar(stmData, index, true);
                }
            }
        }
        private static void TransformGlslStruct(Dsl.FunctionData structFunc, Dsl.ValueOrFunctionData? varNamePart)
        {
            string name = structFunc.GetId();
            var struInfo = new GlslStructInfo();
            struInfo.Name = name;
            GlslVarInfo? last = null;
            foreach (var p in structFunc.Params) {
                var stm = p as Dsl.StatementData;
                if (null != stm) {
                    var varInfo = ParseGlslVarInfo(stm);
                    struInfo.Fields.Add(varInfo);
                    last = varInfo;
                }
                else if (null != last) {
                    var f = p as Dsl.FunctionData;
                    if (null != f) {
                        var varInfo = new GlslVarInfo();
                        varInfo.CopyFrom(last);
                        varInfo.Name = f.GetId();
                        string arrTag = BuildGlslTypeWithTypeArgs(f).Substring(varInfo.Name.Length);
                        varInfo.Type = varInfo.Type + arrTag;
                        struInfo.Fields.Add(varInfo);
                    }
                    else {
                        var varInfo = new GlslVarInfo();
                        varInfo.CopyFrom(last);
                        varInfo.Name = p.GetId();
                        struInfo.Fields.Add(varInfo);
                    }
                }
            }
            if (null != varNamePart) {
                var vinfo = new GlslVarInfo();
                var fd = varNamePart.AsFunction;
                if (null != fd) {
                    List<string> arrTags = new List<string>();
                    vinfo.Name = BuildGlslTypeWithArrTags(fd, arrTags);
                    var sb = new StringBuilder();
                    sb.Append(struInfo.Name);
                    for (int ix = arrTags.Count - 1; ix >= 0; --ix) {
                        sb.Append(arrTags[ix]);
                    }
                    vinfo.Type = sb.ToString();
                }
                else {
                    vinfo.Name = varNamePart.GetId();
                    vinfo.Type = struInfo.Name;
                }
                AddGlslVar(vinfo);
            }
            else {
                var vinfo = new GlslVarInfo();
                vinfo.Type = struInfo.Name;
                SetLastGlslVarType(vinfo);
            }
            if (s_GlslStructInfos.ContainsKey(struInfo.Name)) {
                Console.WriteLine("duplicated glsl struct define '{0}', line: {1}", struInfo.Name, structFunc.GetLine());
            }
            else {
                s_GlslStructInfos.Add(struInfo.Name, struInfo);
            }
        }
        private static void TransformGlslBuffer(Dsl.FunctionData info, string layout, Dsl.ValueOrFunctionData? varNamePart)
        {
            var cbufInfo = new GlslBufferInfo();
            cbufInfo.Name = info.GetId();
            cbufInfo.Layout = layout;
            if (null != varNamePart) {
                Debug.Assert(varNamePart.IsValue);
                cbufInfo.instName = varNamePart.GetId();
            }
            foreach (var p in info.Params) {
                var stm = p as Dsl.StatementData;
                if (null != stm) {
                    var varInfo = ParseGlslVarInfo(stm);
                    cbufInfo.Variables.Add(varInfo);

                    AddGlslVar(varInfo);
                }
            }
            if (s_GlslBufferInfos.ContainsKey(cbufInfo.Name)) {
                Console.WriteLine("duplicated glsl buffer define '{0}', line: {1}", cbufInfo.Name, info.GetLine());
            }
            else {
                s_GlslBufferInfos.Add(cbufInfo.Name, cbufInfo);
            }
        }
        private static void TransformGlslFuncDef(Dsl.FunctionData func, List<string> modifiers)
        {
            Debug.Assert(func.IsHighOrder);
            var paramsPart = func.LowerOrderFunction;
            var funcInfo = new GlslFuncInfo();
            string funcName = paramsPart.GetId();
            funcInfo.Name = funcName;
            PushGlslFuncInfo(funcInfo);
            PushGlslBlock();

            var retInfo = new GlslVarInfo();
            if (modifiers.Count > 0) {
                retInfo.Name = funcName;
                retInfo.Type = modifiers[modifiers.Count - 1];
                modifiers.RemoveAt(modifiers.Count - 1);
                retInfo.Modifiers = modifiers;
            }
            SetGlslRetInfo(retInfo);

            List<string> argTypes = new List<string>();
            for (int ix = 0; ix < paramsPart.GetParamNum(); ++ix) {
                var pcomp = paramsPart.GetParam(ix);
                var pFunc = pcomp as Dsl.FunctionData;
                var pStm = pcomp as Dsl.StatementData;
                Dsl.ISyntaxComponent? pDef = null;
                if (null != pFunc && pFunc.GetId() == "=") {
                    pcomp = pFunc.GetParam(0);
                    pStm = pcomp as Dsl.StatementData;
                    pDef = pFunc.GetParam(1);
                    TransformGlslSyntax(pDef, false, out var addStm);
                }
                if (null != pStm) {
                    var paramInfo = ParseGlslVarInfo(pStm);
                    paramInfo.DefaultValue = pDef;
                    AddGlslParamInfo(paramInfo);
                    argTypes.Add(paramInfo.Type);
                }
            }
            bool first = false;
            string signature = GetFullTypeFuncSig(funcName, argTypes);
            funcInfo.Signature = signature;
            if (!s_GlslFuncInfos.ContainsKey(signature)) {
                s_GlslFuncInfos.Add(signature, funcInfo);
                first = true;
            }
            if (first) {
                if (!s_GlslFuncOverloads.TryGetValue(funcName, out var overloads)) {
                    overloads = new HashSet<string>();
                    s_GlslFuncOverloads.Add(funcName, overloads);
                }
                if (!overloads.Contains(signature)) {
                    overloads.Add(signature);
                }
            }
            for (int stmIx = 0; stmIx < func.GetParamNum(); ++stmIx) {
                Dsl.ISyntaxComponent? syntax = null;
                for (; ; ) {
                    //Remove consecutive semicolons
                    syntax = func.GetParam(stmIx);
                    if (syntax.IsValid()) {
                        break;
                    }
                    else {
                        func.Params.Remove(syntax);
                        if (stmIx < func.GetParamNum()) {
                            syntax = func.GetParam(stmIx);
                        }
                        else {
                            syntax = null;
                            break;
                        }
                    }
                }
                //Process statement
                if (stmIx < func.GetParamNum() && null != syntax && TransformGlslSyntax(syntax, true, out var addStm)) {
                    func.Params.Insert(stmIx + 1, addStm);
                    ++stmIx;
                }
            }
            PopGlslBlock();
            PopGlslFuncInfo();
        }
        private static bool TransformGlslSyntax(Dsl.ISyntaxComponent syntax, bool isStatement, out Dsl.ISyntaxComponent? addStm)
        {
            addStm = null;

            var valData = syntax as Dsl.ValueData;
            var funcData = syntax as Dsl.FunctionData;
            var stmData = syntax as Dsl.StatementData;
            if (null != valData) {
                TransformGlslVar(valData);
            }
            else if (null != funcData) {
                if (funcData.GetId() == "=") {
                    //In GLSL, the assignment statement corresponds directly to HLSL and is legal, so we do not need
                    //to transform it. Therefore, we can separate and process each part individually here.
                    var p = funcData.GetParam(0);
                    var vd = p as Dsl.ValueData;
                    var fd = p as Dsl.FunctionData;
                    var sd = p as Dsl.StatementData;
                    bool handled = false;
                    if (null != vd) {
                        var vinfo = GetGlslVarInfo(vd.GetId());
                        if (null != vinfo) {
                            string baseType = GetGlslTypeRemoveArrTag(vinfo.Type, out var arrNums);
                            if (null != arrNums && arrNums.Count == 1) {
                                var v = funcData.GetParam(1) as Dsl.FunctionData;
                                if (null != v && v.IsHighOrder
                                    && v.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS
                                    && v.LowerOrderFunction.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                                    //In GLSL, arrays can be assigned using initialization syntax, but HLSL does not support this. In such
                                    //cases, a temporary variable needs to be defined and initialized, and then assigned to the array variable
                                    //that needs to be assigned.
                                    handled = true;

                                    TransformGlslSyntax(v, false, out var caddStm);
                                    Debug.Assert(isStatement);

                                    int uid = GenUniqueNumber();
                                    string tempVar = string.Format("_glsl_tmp_{0}", uid);
                                    var lhs = new Dsl.StatementData();
                                    var type = new Dsl.ValueData(baseType, Dsl.ValueData.ID_TOKEN);
                                    var vname = new Dsl.FunctionData();
                                    vname.Name = new Dsl.ValueData(tempVar, Dsl.ValueData.ID_TOKEN);
                                    vname.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET);
                                    vname.AddParam(new Dsl.ValueData(arrNums[0].ToString(), Dsl.ValueData.NUM_TOKEN));
                                    lhs.AddFunction(type);
                                    lhs.AddFunction(vname);
                                    funcData.SetParam(0, lhs);

                                    var newStm = new Dsl.FunctionData();
                                    newStm.Name = new Dsl.ValueData("=", Dsl.ValueData.ID_TOKEN);
                                    newStm.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                                    newStm.AddParam(vd);
                                    newStm.AddParam(new Dsl.ValueData(tempVar, Dsl.ValueData.ID_TOKEN));
                                    newStm.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);

                                    addStm = newStm;
                                }
                            }
                        }
                    }
                    if (!handled) {
                        if (null != vd) {
                            TransformGlslVar(vd);
                        }
                        else if (null != fd) {
                            TransformGlslCall(fd);
                        }
                        else if (null != sd) {
                            //In DSL syntax, semicolons separate statements, while in GLSL, compound statements do not have
                            //separators. During DSL parsing, the compound statement before the assignment statement is
                            //parsed to the left of the assignment statement. Here, we need to split the compound statement
                            //from the left part of the assignment statement to correctly identify variable definitions
                            //within the assignment statement (but the overall representation should not be changed,
                            //otherwise the syntax may be incorrect when outputting to HLSL).
                            Dsl.StatementData? left, right;
                            if (SplitGlslStatementsInExpression(sd, out left, out right)) {
                                Debug.Assert(null != left && null != right);
                                string id = left.GetId();
                                int funcIndex = 0;
                                while (funcIndex < left.GetFunctionNum()) {
                                    bool existsFunc = ParseGlslStatement(left, ref funcIndex, out Dsl.FunctionData? f, out List<string> modifiers);
                                    if (existsFunc) {
                                        Debug.Assert(null != f);
                                        TransformGlslFunction(f);
                                    }
                                    else {
                                        break;
                                    }
                                }
                                if (funcIndex < left.GetFunctionNum())
                                    TransformGlslVar(left, funcIndex, false, sd);
                                TransformGlslVar(right, 0, false, sd);
                            }
                            else {
                                TransformGlslVar(sd, 0, false);
                            }
                        }
                        var v = funcData.GetParam(1);
                        TransformGlslSyntax(v, false, out var caddStm);
                    }
                }
                else if (funcData.GetId() == "<-") {
                    var p = funcData.GetParam(0);
                    var v = funcData.GetParam(1);
                    var vd = p as Dsl.ValueData;
                    var fd = p as Dsl.FunctionData;
                    var sd = p as Dsl.StatementData;
                    if (null != sd) {
                        //In DSL syntax, semicolons separate statements, while in GLSL, compound statements do not
                        //have separators.During DSL parsing, the compound statement before the assignment statement
                        //is parsed to the left of the return statement.Here, we need to split the compound statement
                        //from the left part of the return statement to handle the return statement correctly.The
                        //return statement has already been changed to a<-expression during parsing. Here, it needs
                        //to be changed to a return function form.
                        Dsl.StatementData? left, right;
                        if (SplitGlslStatementsInExpression(sd, out left, out right)) {
                            Debug.Assert(null != left && null != right);
                            string id = left.GetId();
                            int funcIndex = 0;
                            while (funcIndex < left.GetFunctionNum()) {
                                bool existsFunc = ParseGlslStatement(left, ref funcIndex, out Dsl.FunctionData? f, out List<string> modifiers);
                                if (existsFunc) {
                                    Debug.Assert(null != f);
                                    TransformGlslFunction(f);
                                }
                                else {
                                    break;
                                }
                            }
                            if (funcIndex < left.GetFunctionNum()) {
                                //TransformGlslVar(left, 0, false, sd);
                            }
                            //TransformGlslVar(right, 0, false, sd);
                        }
                        else {
                            //TransformGlslVar(sd, 0, false);
                        }
                    }
                    var vfd = v as Dsl.FunctionData;
                    if (null != vfd && vfd.GetId() == "*") {
                        string lhs = vfd.GetParamId(0);
                        string rhs = vfd.GetParamId(1);
                    }
                    TransformGlslSyntax(v, false, out var caddStm);
                    if (null != fd) {
                        funcData.LowerOrderFunction = fd;
                    }
                    else {
                        funcData.Name.SetId("return");
                    }
                    funcData.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);
                    funcData.Params.Clear();
                    funcData.AddParam(v);
                }
                else {
                    TransformGlslFunction(funcData);
                }
            }
            else if (null != stmData) {
                int funcIndex = 0;
                while (funcIndex < stmData.GetFunctionNum()) {
                    bool existsFunc = ParseGlslStatement(stmData, ref funcIndex, out Dsl.FunctionData? f, out List<string> modifiers);
                    if (existsFunc) {
                        Debug.Assert(null != f);
                        TransformGlslFunction(f);
                    }
                    else {
                        break;
                    }
                }
                var firstValOrFunc = stmData.GetFunction(funcIndex);
                string funcId = firstValOrFunc.GetId();
                if (funcId == "return") {
                    var retFunc = firstValOrFunc.AsFunction;
                    if (null != retFunc) {
                        TransformGlslFunction(retFunc);
                    }
                    else {
                        for (int ix = funcIndex + 1; ix < stmData.GetFunctionNum(); ++ix) {
                            var fd = stmData.GetFunction(ix).AsFunction;
                            if (null != fd)
                                TransformGlslFunction(fd);
                        }
                    }
                }
                else if (funcId == "hlsl_attr" || funcId == "switch" || funcId == "if" || funcId == "while" || funcId == "for" || funcId == "do") {
                    for (int ix = funcIndex; ix < stmData.GetFunctionNum(); ++ix) {
                        var fd = stmData.GetFunction(ix).AsFunction;
                        if (null != fd)
                            TransformGlslFunction(fd);
                    }
                }
                else if (funcId == "?") {
                    var tfunc = stmData.First.AsFunction;
                    var ffunc = stmData.Second.AsFunction;
                    Debug.Assert(null != tfunc && null != ffunc);
                    TransformGlslFunction(tfunc);
                    TransformGlslFunction(ffunc);
                }
                else {
                    TransformGlslVar(stmData, funcIndex, false);
                }
            }
            return null != addStm;
        }
        private static void TransformGlslFunction(Dsl.FunctionData funcData)
        {
            if (null != funcData) {
                if (funcData.IsHighOrder) {
                    var lowerFunc = funcData.LowerOrderFunction;
                    if (lowerFunc.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET && funcData.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS) {
                        //arr init to call arr init func
                        int size = funcData.GetParamNum();
                        if (lowerFunc.GetParamNum() == 1) {
                            if (int.TryParse(lowerFunc.GetParamId(0), out var num)) {
                                Debug.Assert(num == size);
                            }
                        }
                        string ty = lowerFunc.GetId();
                        string fn = "glsl_" + ty + "_x" + size + "_ctor";
                        if(!s_ArrayInits.TryGetValue(fn, out var arrInitInfo)) {
                            arrInitInfo = new ArrayInitInfo { Type = ty, Size = size };
                            s_ArrayInits.Add(fn, arrInitInfo);
                        }
                        funcData.Name = new Dsl.ValueData(fn, Dsl.ValueData.ID_TOKEN);
                        TransformGlslFunction(funcData);
                        return;
                    }
                    TransformGlslFunction(lowerFunc);
                    if (funcData.HaveStatement()) {
                        PushGlslBlock();
                        for (int stmIx = 0; stmIx < funcData.GetParamNum(); ++stmIx) {
                            var syntax = funcData.GetParam(stmIx);
                            if (TransformGlslSyntax(syntax, true, out var addStm)) {
                                funcData.Params.Insert(stmIx + 1, addStm);
                                ++stmIx;
                            }
                        }
                        PopGlslBlock();
                    }
                    else if (funcData.HaveParam()) {
                        for (int paramIx = 0; paramIx < funcData.GetParamNum(); ++paramIx) {
                            var syntax = funcData.GetParam(paramIx);
                            if (TransformGlslSyntax(syntax, false, out var addStm)) {
                                funcData.Params.Insert(paramIx + 1, addStm);
                                ++paramIx;
                            }
                        }
                    }
                }
                else if (funcData.HaveStatement()) {
                    PushGlslBlock();
                    for (int stmIx = 0; stmIx < funcData.GetParamNum(); ++stmIx) {
                        Dsl.ISyntaxComponent? syntax = null;
                        for (; ; ) {
                            //consecutive
                            syntax = funcData.GetParam(stmIx);
                            if (syntax.IsValid()) {
                                break;
                            }
                            else {
                                funcData.Params.Remove(syntax);
                                if (stmIx < funcData.GetParamNum()) {
                                    syntax = funcData.GetParam(stmIx);
                                }
                                else {
                                    syntax = null;
                                    break;
                                }
                            }
                        }
                        //Process statement
                        if (stmIx < funcData.GetParamNum() && null != syntax && TransformGlslSyntax(syntax, true, out var addStm)) {
                            funcData.Params.Insert(stmIx + 1, addStm);
                            ++stmIx;
                        }
                    }
                    PopGlslBlock();
                }
                else if (funcData.HaveParam()) {
                    TransformGlslCall(funcData);
                }
            }
        }
        private static void TransformGlslCall(Dsl.FunctionData call)
        {
            if (!call.HaveId() && call.GetParamNum() == 1 && call.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS) {
                var pp = call.GetParam(0);
                var innerCall = pp as Dsl.FunctionData;
                if (null != innerCall) {
                    if (!innerCall.HaveId() && innerCall.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS) {
                        call.ClearParams();
                        call.Params.AddRange(innerCall.Params);
                        TransformGlslCall(call);
                        return;
                    }
                }
            }
            for (int ix = 0; ix < call.GetParamNum(); ++ix) {
                var syntax = call.GetParam(ix);
                if (TransformGlslSyntax(syntax, false, out var addStm)) {
                    call.Params.Insert(ix + 1, addStm);
                    ++ix;
                }
            }
            string funcName = call.GetId();
            if (call.Params.Count == 1 && (
                funcName == "vec2" || funcName == "vec3" || funcName == "vec4" ||
                funcName == "dvec2" || funcName == "dvec3" || funcName == "dvec4" ||
                funcName == "ivec2" || funcName == "ivec3" || funcName == "ivec4" ||
                funcName == "uvec2" || funcName == "uvec3" || funcName == "uvec4" ||
                funcName == "bvec2" || funcName == "bvec3" || funcName == "bvec4" ||
                funcName == "mat2" || funcName == "mat3" || funcName == "mat4" ||
                funcName == "dmat2" || funcName == "dmat3" || funcName == "dmat4")) {
                bool isSimpleType = false;
                var p = call.GetParam(0);
                string t = GlslTypeInference(p);
                if (t == "float" || t == "double" || t == "int" || t == "uint" || t == "bool") {
                    isSimpleType = true;
                }
                if (isSimpleType)
                    call.Name.SetId("glsl_" + funcName);
                var val = p as Dsl.ValueData;
                string strVal = p.GetId();
                if (null != val && val.GetIdType() == Dsl.ValueData.NUM_TOKEN && strVal.IndexOf('.') < 0 && strVal.IndexOfAny(s_eOrE) < 0) {
                    if (funcName.StartsWith("vec") || funcName.StartsWith("dvec") || funcName.StartsWith("mat") || funcName.StartsWith("dmat"))
                        val.SetId(strVal + ".0");
                }
            }
            /*
            * When the column-major matrix of glsl is translated into hlsl, it is directly translated into a row-major matrix (columns become rows), so that the operations of matrices and vectors and array access to rows and columns are consistent.
            * However, the order of matrix and matrix operations needs to be exchanged, and the parameter order is not adjusted during construction (the parameter order directly corresponds to hlsl, which is the row-major matrix order)
            else if (funcName == "mat2" || funcName == "mat3" || funcName == "mat4" ||
                    funcName == "dmat2" || funcName == "dmat3" || funcName == "dmat4") {
                call.Name.SetId("glsl_" + funcName + "_ctor");
            }
            */
            else if (s_GlslStructInfos.TryGetValue(funcName, out var struInfo)) {
                call.Name.SetId("glsl_" + funcName + "_ctor");
            }
            else if (funcName == "inverse") {
                call.Name.SetId("glsl_" + funcName);
            }
            else if (call.Params.Count == 2 && funcName == "atan") {
                call.Name.SetId("atan2");
            }
            else if (funcName == "texture") {
                var param1 = call.GetParam(0) as Dsl.ValueData;
                if (null != param1) {
                    string texName = param1.GetId();

                    var param2 = call.GetParam(1);
                    CheckTextureType(texName, param2);

                    if (call.GetParamNum() == 3) {
                        call.Name.SetId(texName + ".SampleBias");
                    }
                    else {
                        call.Name.SetId(texName + ".Sample");
                    }
                    param1.SetId("s_linear_clamp_sampler");
                }
            }
            else if (funcName == "textureOffset") {
                var param1 = call.GetParam(0) as Dsl.ValueData;
                if (null != param1) {
                    string texName = param1.GetId();

                    var param2 = call.GetParam(1);
                    CheckTextureType(texName, param2);

                    if (call.GetParamNum() == 4) {
                        call.Name.SetId(texName + ".SampleBias");
                        var param3 = call.GetParam(2);
                        var param4 = call.GetParam(3);
                        int sep3 = param3.GetSeparator();
                        int sep4 = param4.GetSeparator();
                        param3.SetSeparator(sep4);
                        param4.SetSeparator(sep3);
                        call.SetParam(2, param4);
                        call.SetParam(3, param3);
                    }
                    else {
                        call.Name.SetId(texName + ".Sample");
                    }
                    param1.SetId("s_linear_clamp_sampler");
                }
            }
            else if (funcName == "textureLod" || funcName == "textureLodOffset") {
                var param1 = call.GetParam(0) as Dsl.ValueData;
                if (null != param1) {
                    string texName = param1.GetId();

                    var param2 = call.GetParam(1);
                    CheckTextureType(texName, param2);

                    call.Name.SetId(texName + ".SampleLevel");
                    param1.SetId("s_linear_clamp_sampler");
                }
            }
            else if (funcName == "textureGrad" || funcName == "textureGradOffset") {
                var param1 = call.GetParam(0) as Dsl.ValueData;
                if (null != param1) {
                    string texName = param1.GetId();

                    var param2 = call.GetParam(1);
                    CheckTextureType(texName, param2);

                    call.Name.SetId(texName + ".SampleGrad");
                    param1.SetId("s_linear_clamp_sampler");
                }
            }
            else if (funcName == "textureGather" || funcName == "textureGatherOffset") {
                var param1 = call.GetParam(0) as Dsl.ValueData;
                if (null != param1) {
                    string texName = param1.GetId();

                    var param2 = call.GetParam(1);
                    CheckTextureType(texName, param2);

                    call.Name.SetId(texName + ".Gather");
                    param1.SetId("s_linear_clamp_sampler");
                }
            }
            else if (funcName == "texelFetch") {
                var param1 = call.GetParam(0) as Dsl.ValueData;
                if (null != param1) {
                    string texName = param1.GetId();

                    var param2 = call.GetParam(1);
                    CheckTextureType(texName, param2);

                    call.Name.SetId(texName + ".Load");
                    if (call.GetParamNum() == 3) {
                        var param3 = call.GetParam(2);

                        var newParam = new Dsl.FunctionData();
                        newParam.Name = new Dsl.ValueData("ivec3", Dsl.ValueData.ID_TOKEN);
                        newParam.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);
                        newParam.AddParam(param2);
                        newParam.AddParam(param3);
                        call.Params.RemoveAt(2);
                        call.SetParam(1, newParam);
                    }
                    call.Params.RemoveAt(0);
                }
            }
            else if (call.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR) {
                if (funcName == "*") {
                    bool hasMat = false;
                    bool hasScalar = false;
                    foreach (var p in call.Params) {
                        string t = GlslTypeInference(p);
                        if (t == "mat2" || t == "mat3" || t == "mat4" ||
                            t == "dmat2" || t == "dmat3" || t == "dmat4") {
                            hasMat = true;
                        }
                        else if(t == "float" || t == "double" || 
                            t == "int" || t == "uint") {
                            hasScalar = true;
                        }
                    }
                    if (hasMat && !hasScalar) {
                        call.Name.SetId("glsl_mul");
                        call.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);
                        call.GetParam(0).SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_COMMA);
                    }
                }
                else if (funcName == "*=") {
                    bool hasMat = false;
                    bool hasScalar = false;
                    foreach (var p in call.Params) {
                        string t = GlslTypeInference(p);
                        if (t == "mat2" || t == "mat3" || t == "mat4" ||
                            t == "dmat2" || t == "dmat3" || t == "dmat4") {
                            hasMat = true;
                        }
                        else if (t == "float" || t == "double" ||
                            t == "int" || t == "uint") {
                            hasScalar = true;
                        }
                    }
                    if (hasMat && !hasScalar) {
                        call.Name.SetId(" = ");

                        var newCall = new Dsl.FunctionData();
                        newCall.Name = new Dsl.ValueData("glsl_mul");
                        newCall.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);

                        Dsl.ISyntaxComponent nv;
                        var v = call.GetParam(0);
                        var p = call.GetParam(1);
                        var vd = v as Dsl.ValueData;
                        var vf = v as Dsl.FunctionData;
                        if (null != vd) {
                            nv = new Dsl.ValueData(vd.GetId(), vd.GetIdType());
                        }
                        else {
                            Debug.Assert(null != vf);
                            var f = new Dsl.FunctionData();
                            f.CopyFrom(vf);
                            nv = f;
                        }
                        nv.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_COMMA);

                        newCall.AddParam(nv);
                        newCall.AddParam(p);
                        newCall.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_NOTHING);

                        call.SetParam(1, newCall);
                    }
                }
            }
        }
        private static void TransformGlslVar(Dsl.ValueData valData)
        {
            if (valData.IsId()) {
                string vid = valData.GetId();
                if (vid != "true" && vid != "false" && vid.IndexOf(' ') < 0) {
                    var vinfo = GetGlslVarInfo(vid);
                    if (null == vinfo) {
                        var lastVarInfo = GetLastGlslVarType();
                        if (null != lastVarInfo) {
                            var varInfo = new GlslVarInfo();
                            varInfo.CopyFrom(lastVarInfo);
                            varInfo.Name = vid;
                            AddGlslVar(varInfo);
                        }
                    }
                }
            }
        }
        private static void TransformGlslVar(Dsl.StatementData stmData, int index, bool toplevel)
        {
            TransformGlslVar(stmData, index, toplevel, null);
        }
        private static void TransformGlslVar(Dsl.StatementData stmData, int index, bool toplevel, Dsl.StatementData? oriStmData)
        {
            var lastId = stmData.Last.GetId();
            var last = stmData.Last.AsFunction;
            if (null == last || (last.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET && !last.HaveStatement())) {
                bool isType = true;
                bool needAdjustArrTag = false;
                int funcNum = stmData.GetFunctionNum();
                for (int ix = index; ix < funcNum; ++ix) {
                    var valOrFunc = stmData.GetFunction(ix);
                    var val = valOrFunc.AsValue;
                    var func = valOrFunc.AsFunction;
                    if (null == val) {
                        if (ix < funcNum - 2) {
                            isType = false;
                            break;
                        }
                        else {
                            if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                                if (ix == funcNum - 2) {
                                    needAdjustArrTag = true;
                                }
                            }
                            else {
                                isType = false;
                                break;
                            }
                        }
                    }
                }
                if (isType) {
                    GlslVarInfo vinfo = ParseGlslVarInfo(stmData);
                    AddGlslVar(vinfo);
                    if (needAdjustArrTag) {
                        var typeFunc = stmData.GetFunction(funcNum - 2).AsFunction;
                        var nameValOrFunc = stmData.GetFunction(funcNum - 1);
                        var nameVal = nameValOrFunc.AsValue;
                        var nameFunc = nameValOrFunc.AsFunction;
                        if (null != nameVal) {
                            nameFunc = new Dsl.FunctionData();
                            nameFunc.Name = nameVal;
                            stmData.SetFunction(funcNum - 1, nameFunc);
                        }
                        else {
                            Debug.Assert(null != nameFunc);
                            while (nameFunc.IsHighOrder) {
                                nameFunc = nameFunc.LowerOrderFunction;
                            }
                        }
                        Debug.Assert(null != nameFunc);
                        while (null != typeFunc) {
                            var newFunc = new Dsl.FunctionData();
                            newFunc.Name = nameFunc.Name;
                            newFunc.SetParamClass(typeFunc.GetParamClass());
                            newFunc.Params = typeFunc.Params;
                            newFunc.SetSeparator(typeFunc.GetSeparator());
                            nameFunc.LowerOrderFunction = newFunc;
                            nameFunc = newFunc;
                            if (typeFunc.IsHighOrder) {
                                typeFunc = typeFunc.LowerOrderFunction;
                            }
                            else {
                                stmData.SetFunction(funcNum - 2, typeFunc.Name);
                                typeFunc = null;
                            }
                        }
                    }
                    if (toplevel) {
                        var sta = new Dsl.ValueData("static");
                        if (null != oriStmData) {
                            int ix = oriStmData.Functions.IndexOf(stmData.First);
                            if (ix >= 0) {
                                oriStmData.Functions.Insert(ix, sta);
                            }
                        }
                        else {
                            stmData.Functions.Insert(index, sta);
                        }
                    }
                }
            }
        }

        private static void CheckTextureType(string texName, Dsl.ISyntaxComponent coord)
        {
            string ty = GlslTypeInference(coord);
            if (ty.Length > 0) {
                var texTypes = s_MainShaderInfo.TexTypes;
                string channel = texName;
                var vinfo = GetGlslVarInfo(texName);
                if (null != vinfo) {
                    texTypes = new SortedDictionary<string, string> { { texName, vinfo.Type } };
                }
                else if (!s_ShaderToyChannels.Contains(channel)) {
                    int ix = texName.IndexOf('_');
                    if (ix >= 0) {
                        string bufId = texName.Substring(0, ix);
                        channel = texName.Substring(ix + 1);
                        bool find = false;
                        foreach (var bufInfo in s_ShaderBufferInfos) {
                            if (bufInfo.BufferId == bufId) {
                                texTypes = bufInfo.TexTypes;
                                find = true;
                                break;
                            }
                        }
                        if (!find) {
                            texTypes = s_EmptyDictionarys;
                        }
                    }
                }
                string suffix = GetGlslTypeSuffix(ty);
                if (suffix == "3") {
                    if (texTypes.TryGetValue(channel, out var cty)) {
                        if (cty == "sampler2D") {
                            Console.WriteLine("Channel '{0}' type {1} can't sample by 3d coord, line: {2}.", texName, cty, coord.GetLine());
                        }
                    }
                    else {
                        Console.WriteLine("Undefined 3d channel '{0}', line: {1}.", texName, coord.GetLine());
                    }
                }
                else if(suffix == "2") {
                    if (texTypes.TryGetValue(channel, out var cty)) {
                        if (cty != "sampler2D") {
                            Console.WriteLine("Channel '{0}' type {1} can't sample by 2d coord, line: {2}.", texName, cty, coord.GetLine());
                        }
                    }
                    else {
                        Console.WriteLine("Undefined 2d channel '{0}', line: {1}.", texName, coord.GetLine());
                    }
                }
            }
        }
        private static string GlslTypeInference(Dsl.ISyntaxComponent syntax)
        {
            var valData = syntax as Dsl.ValueData;
            var funcData = syntax as Dsl.FunctionData;
            var stmData = syntax as Dsl.StatementData;
            if (null != valData) {
                var varInfo = GetGlslVarInfo(valData.GetId());
                if (null != varInfo) {
                    return varInfo.Type;
                }
                else {
                    int idType = valData.GetIdType();
                    string val = valData.GetId();
                    switch (idType) {
                        case Dsl.ValueData.NUM_TOKEN:
                            if (val.Contains('.'))
                                return "float";
                            else
                                return "int";
                        default:
                            if (val == "true" || val == "false")
                                return "bool";
                            return string.Empty;
                    }
                }
            }
            else if (null != funcData) {
                string funcName = funcData.GetId();
                if (funcName == "vec2" || funcName == "vec3" || funcName == "vec4" ||
                    funcName == "ivec2" || funcName == "ivec3" || funcName == "ivec4" ||
                    funcName == "uvec2" || funcName == "uvec3" || funcName == "uvec4" ||
                    funcName == "bvec2" || funcName == "bvec3" || funcName == "bvec4") {
                    return funcName;
                }
                else if (funcName == "float" || funcName == "double" || funcName == "int" ||
                    funcName == "uint" || funcName == "bool" ||
                    funcName == "mat2" || funcName == "mat3" || funcName == "mat4" ||
                    funcName == "dmat2" || funcName == "dmat3" || funcName == "dmat4") {
                    return funcName;
                }
                else if (funcName == "texture") {
                    return "vec4";
                }
                switch (funcData.GetParamClassUnmasked()) {
                    case (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR: {
                            int pnum = funcData.GetParamNum();
                            if (pnum == 1) {
                                string op = funcData.GetId();
                                string p1 = GlslTypeInference(funcData.GetParam(0));
                                return GlslOperatorTypeInference(op, p1);
                            }
                            else {
                                string op = funcData.GetId();
                                string p1 = GlslTypeInference(funcData.GetParam(0));
                                string p2 = GlslTypeInference(funcData.GetParam(1));
                                return GlslOperatorTypeInference(op, p1, p2);
                            }
                        }
                    case (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PERIOD: {
                            if (funcData.IsHighOrder) {
                                string objType = GlslTypeInference(funcData.LowerOrderFunction);
                                string mname = funcData.GetParamId(0);
                                return GlslMemberTypeInference(".", objType, string.Empty, mname);
                            }
                            else {
                                var vinfo = GetGlslVarInfo(funcName);
                                if (null != vinfo) {
                                    string objType = vinfo.Type;
                                    string mname = funcData.GetParamId(0);
                                    return GlslMemberTypeInference(".", objType, string.Empty, mname);
                                }
                                else {
                                    string objType = funcName;
                                    string mname = funcData.GetParamId(0);
                                    return GlslMemberTypeInference(".", objType, string.Empty, mname);
                                }
                            }
                        }
                    case (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET: {
                            if (funcData.IsHighOrder) {
                                string objType = GlslTypeInference(funcData.LowerOrderFunction);
                                string mname = funcData.GetParamId(0);
                                return GlslMemberTypeInference("[]", objType, string.Empty, mname);
                            }
                            else {
                                var vinfo = GetGlslVarInfo(funcName);
                                if (null != vinfo) {
                                    string objType = vinfo.Type;
                                    string mname = funcData.GetParamId(0);
                                    return GlslMemberTypeInference("[]", objType, string.Empty, mname);
                                }
                                else {
                                    string objType = funcName;
                                    string mname = funcData.GetParamId(0);
                                    return GlslMemberTypeInference("[]", objType, string.Empty, mname);
                                }
                            }
                        }
                    case (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS: {
                            List<string> argTypes = new List<string>();
                            foreach (var p in funcData.Params) {
                                string type = GlslTypeInference(p);
                                argTypes.Add(type);
                            }
                            if (string.IsNullOrEmpty(funcName)) {
                                return (argTypes.Count > 0 ? argTypes[argTypes.Count - 1] : string.Empty);
                            }
                            else {
                                return GlslFunctionTypeInference(funcName, argTypes, out GlslFuncInfo? funcInfo);
                            }
                        }
                }
            }
            else if (null != stmData) {
                if (stmData.GetId() == "?") {
                    var tfunc = stmData.First.AsFunction;
                    var ffunc = stmData.Second.AsFunction;
                    Debug.Assert(null != tfunc && null != ffunc);
                    var texp = tfunc.GetParam(0);
                    var fexp = ffunc.GetParam(0);
                    var t1 = GlslTypeInference(texp);
                    var t2 = GlslTypeInference(fexp);
                    if (!string.IsNullOrEmpty(t1))
                        return t1;
                    else if (!string.IsNullOrEmpty(t2))
                        return t2;
                }
            }
            return string.Empty;
        }

        private static string GlslOperatorTypeInference(string op, string opd)
        {
            string resultType = string.Empty;
            if (op == "+" || op == "-" || op == "~")
                resultType = opd;
            else if (op == "!")
                resultType = "bool";
            return resultType;
        }
        private static string GlslOperatorTypeInference(string op, string opd1, string opd2)
        {
            string resultType = string.Empty;
            if (op == "*") {
                if (opd1 == opd2) {
                    resultType = opd1;
                }
                else {
                    return GetGlslMatmulType(opd1, opd2);
                }
            }
            else if (op == "+" || op == "-" || op == "/" || op == "%") {
                if (opd1.StartsWith("mat") || opd1.StartsWith("dmat"))
                    resultType = opd1;
                else if (opd2.StartsWith("mat") || opd2.StartsWith("dmat"))
                    resultType = opd2;
                else if (opd1.StartsWith("vec") || opd1.StartsWith("dvec"))
                    resultType = opd1;
                else if (opd2.StartsWith("vec") || opd2.StartsWith("dvec"))
                    resultType = opd2;
                else
                    resultType = opd1.Length >= opd2.Length ? opd1 : opd2;
            }
            else if (op == "&&" || op == "||" || op == ">=" || op == "==" || op == "!=" || op == "<=" || op == ">" || op == "<") {
                if (opd1.StartsWith("vec") || opd1.StartsWith("dvec"))
                    resultType = "bool" + GetGlslTypeSuffix(opd1);
                else if (opd2.StartsWith("vec") || opd2.StartsWith("dvec"))
                    resultType = "bool" + GetGlslTypeSuffix(opd2);
                else
                    resultType = "bool";
            }
            else if (op == "&" || op == "|" || op == "^" || op == "<<" || op == ">>") {
                if (opd1.StartsWith("vec") || opd1.StartsWith("dvec"))
                    resultType = "int" + GetGlslTypeSuffix(opd1);
                else if (opd2.StartsWith("vec") || opd2.StartsWith("dvec"))
                    resultType = "int" + GetGlslTypeSuffix(opd2);
                else
                    resultType = "int";
            }
            return resultType;
        }
        private static string GlslFunctionTypeInference(string func, IList<string> args, out GlslFuncInfo? funcInfo)
        {
            funcInfo = null;
            string callSig = func + "_" + string.Join("_", args);
            if (s_GlslFuncOverloads.TryGetValue(func, out var overloads)) {
                foreach (var sig in overloads) {
                    if (sig.StartsWith(callSig)) {
                        if (s_GlslFuncInfos.TryGetValue(sig, out var tmpInfo)) {
                            if (args.Count == tmpInfo.Params.Count && sig == callSig || args.Count < tmpInfo.Params.Count && null != tmpInfo.Params[args.Count].DefaultValue) {
                                funcInfo = tmpInfo;
                                return null == funcInfo.RetInfo ? "void" : funcInfo.RetInfo.Type;
                            }
                        }
                    }
                }
                //find nearst match
                int curScore = -1;
                foreach (var sig in overloads) {
                    if (s_GlslFuncInfos.TryGetValue(sig, out var tmpInfo)) {
                        if (IsGlslArgsMatch(args, tmpInfo, out int newScore) && curScore < newScore) {
                            curScore = newScore;
                            funcInfo = tmpInfo;
                        }
                    }
                }
                if (null != funcInfo) {
                    return null == funcInfo.RetInfo ? "void" : funcInfo.RetInfo.Type;
                }
            }
            else {
                //built-in function
                if (s_GlslBuiltInFuncs.TryGetValue(func, out var resultType)) {
                    string ret = GetFuncResultType(resultType, func, args, args, false, true);
                    return ret;
                }
                else if (func.StartsWith("glsl_")) {
                    string oriName = func.Substring("glsl_".Length);
                    if (oriName.EndsWith("_ctor"))
                        oriName = oriName.Substring(0, oriName.Length - "_ctor".Length);
                    if (s_GlslBuiltInFuncs.TryGetValue(oriName, out var resultType2)) {
                        string ret = GetFuncResultType(resultType2, oriName, args, args, false, true);
                        return ret;
                    }
                    else if(s_GlslStructInfos.TryGetValue(oriName, out var struInfo)) {
                        string ret = oriName;
                        return ret;
                    }
                    else {
                        string ret = oriName;
                        return ret;
                    }
                }
            }
            return string.Empty;
        }
        private static string GlslMemberTypeInference(string op, string objType, string resultType, string memberOrType)
        {
            if (op == ".") {
                if (s_GlslStructInfos.TryGetValue(objType, out var info)) {
                    foreach (var field in info.Fields) {
                        if (field.Name == memberOrType) {
                            resultType = field.Type;
                            break;
                        }
                    }
                }
                else {
                    string baseType = GetGlslTypeRemoveSuffix(objType);
                    string suffix = GetGlslTypeSuffix(objType);
                    if (string.IsNullOrEmpty(resultType)) {
                        if (memberOrType.Length == 1) {
                            string stype = GlslVectorMatrixTypeToScalarType(objType);
                            resultType = stype;
                        }
                        else if (baseType == "mat") {
                            int ct = GetMemberCount(memberOrType);
                            if (ct == 1)
                                resultType = "float";
                            else
                                resultType = "vec" + ct.ToString();
                        }
                        else if (baseType == "dmat") {
                            int ct = GetMemberCount(memberOrType);
                            if (ct == 1)
                                resultType = "double";
                            else
                                resultType = "dvec" + ct.ToString();
                        }
                        else {
                            resultType = baseType + memberOrType.Length.ToString();
                        }
                    }
                }
            }
            else if (op == "[]") {
                resultType = objType;
            }
            return resultType;
        }
        private static string GetGlslMaxType(params string[] argTypes)
        {
            IList<string> ats = argTypes;
            return GetGlslMaxType(ats);
        }
        private static string GetGlslMaxType(IList<string> argTypes)
        {
            string mat = string.Empty;
            string vec = string.Empty;
            string maxTy = string.Empty;
            string ty0 = argTypes[0];
            if (ty0.StartsWith("mat") || ty0.StartsWith("dmat"))
                mat = ty0;
            else if (ty0.StartsWith("vec") || ty0.StartsWith("dvec"))
                vec = ty0;
            else
                maxTy = ty0;
            for (int i = 1; i < argTypes.Count; ++i) {
                string ty = argTypes[i];
                if (string.IsNullOrEmpty(mat) && (ty.StartsWith("mat") || ty.StartsWith("dmat")))
                    mat = ty;
                else if (string.IsNullOrEmpty(vec) && (ty.StartsWith("vec") || ty.StartsWith("dvec")))
                    vec = ty;
                else
                    maxTy = maxTy.Length >= ty.Length ? maxTy : ty;
            }
            if (!string.IsNullOrEmpty(mat))
                return mat;
            else if (!string.IsNullOrEmpty(vec))
                return vec;
            else
                return maxTy;
        }
        private static string GetGlslMatmulType(string oriTypeA, string oriTypeB)
        {
            string ret;
            string bt1 = GetGlslTypeRemoveSuffix(oriTypeA);
            string s1 = GetGlslTypeSuffix(oriTypeA);
            string bt2 = GetGlslTypeRemoveSuffix(oriTypeB);
            string s2 = GetGlslTypeSuffix(oriTypeB);
            if (s1.Length == 0 && s2.Length == 0)
                ret = GetGlslMaxType(oriTypeA, oriTypeB);
            else if (s1.Length == 0)
                ret = oriTypeB;
            else if (s2.Length == 0)
                ret = oriTypeA;
            else if (s1.Length == 1 && s2.Length == 0)
                ret = oriTypeA;
            else if (s1.Length == 0 && s2.Length == 1)
                ret = oriTypeB;
            else if (bt1 == "vec" && bt2 == "vec")
                return "float";
            else if (bt1 == "dvec" && bt2 == "dvec")
                return "double";
            else if (bt1 == "vec" || bt1 == "dvec")
                return oriTypeA;
            else if (bt2 == "vec" || bt2 == "dvec")
                return oriTypeB;
            else if (s1.Length == 1 && s2.Length == 1) {
                return GetGlslMaxType(oriTypeA, oriTypeB);
            }
            else {
                string mt = GetGlslMaxType(bt1, bt2);
                ret = mt + s1[0] + "x" + s2[2];
            }
            return ret;
        }
        private static bool SplitGlslStatementsInExpression(Dsl.StatementData expParam, out Dsl.StatementData? left, out Dsl.StatementData? right)
        {
            left = null;
            right = null;
            int funcNum = expParam.GetFunctionNum();
            var lastFunc = expParam.Last.AsFunction;
            if (null != lastFunc && lastFunc.IsHighOrder) {
                //The statement curly braces are followed by a parenthesized expression.
                var innerFunc = lastFunc;
                while (innerFunc.IsHighOrder && innerFunc.HaveParam())
                    innerFunc = innerFunc.LowerOrderFunction;
                if (null != innerFunc && innerFunc.HaveStatement()) {
                    left = new Dsl.StatementData();
                    right = new Dsl.StatementData();
                    for (int i = 0; i < funcNum - 1; ++i) {
                        left.AddFunction(expParam.GetFunction(i));
                    }
                    left.AddFunction(innerFunc);
                    var func = new Dsl.FunctionData();
                    right.AddFunction(func);
                    var f = lastFunc;
                    while (f != innerFunc) {
                        func.Params.AddRange(f.Params);
                        f = f.LowerOrderFunction;
                        if (f != innerFunc) {
                            func.LowerOrderFunction = new Dsl.FunctionData();
                            func = func.LowerOrderFunction;
                        }
                    }
                    return true;
                }
            }
            for (int ix = funcNum - 2; ix >= 0; --ix) {
                var func = expParam.GetFunction(ix);
                string id = func.GetId();
                var f_ = func.AsFunction;
                if (null != f_) {
                    if (ix < funcNum - 1 && f_.HaveStatement()) {
                        left = new Dsl.StatementData();
                        right = new Dsl.StatementData();
                        for (int i = 0; i <= ix; ++i) {
                            left.AddFunction(expParam.GetFunction(i));
                        }
                        for (int i = ix + 1; i < funcNum; ++i) {
                            right.AddFunction(expParam.GetFunction(i));
                        }
                        return true;
                    }
                }
            }
            return false;
        }
        private static bool ParseGlslToplevelStatement(Dsl.StatementData stmData, ref int index, out Dsl.FunctionData? f, out string layout, out List<string> modifiers, out Dsl.ValueOrFunctionData? varNamePart)
        {
            bool ret = false;
            f = null;
            layout = string.Empty;
            modifiers = new List<string>();
            varNamePart = null;
            for (int ix = index; ix < stmData.GetFunctionNum(); ++ix) {
                var func = stmData.GetFunction(ix);
                string id = func.GetId();
                var f_ = func.AsFunction;
                if (null != f_) {
                    if (id == "layout") {
                        layout = f_.GetParamId(0);
                    }
                    else if (f_.HaveId() && f_.HaveStatement()) {
                        f = f_;
                        ret = true;
                        index = ix + 1;
                        if (f_.IsHighOrder) {
                            break;
                        }
                    }
                    else if(f_.HaveId() && f_.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                        varNamePart = func;
                    }
                    else if (f_.HaveId() && f_.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS) {
                        varNamePart = func;
                    }
                    else {
                        Console.WriteLine("unknown glsl top level syntax, line: {0}", f_.GetLine());
                    }
                }
                else {
                    if (ret) {
                        varNamePart = func;
                    }
                    else if (func == stmData.Last) {
                        varNamePart = func;
                    }
                    else {
                        modifiers.Add(id);
                    }
                }
            }
            return ret;
        }
        private static bool ParseGlslStatement(Dsl.StatementData stmData, ref int index, out Dsl.FunctionData? f, out List<string> modifiers)
        {
            bool ret = false;
            f = null;
            modifiers = new List<string>();
            for (int ix = index; ix < stmData.GetFunctionNum(); ++ix) {
                var func = stmData.GetFunction(ix);
                string id = func.GetId();
                var f_ = func.AsFunction;
                if (null != f_ && f_.HaveStatement()) {
                    f = f_;
                    ret = true;
                    index = ix + 1;
                    break;
                }
                else {
                    modifiers.Add(id);
                }
            }
            return ret;
        }
        private static GlslVarInfo ParseGlslVarInfo(Dsl.StatementData varStm)
        {
            var lastId = varStm.Last.GetId();

            var nameInfoFunc = varStm.Last.AsFunction;
            string arrTag = string.Empty;
            if (null != nameInfoFunc) {
                arrTag = BuildGlslTypeWithTypeArgs(nameInfoFunc).Substring(lastId.Length);
            }

            GlslVarInfo vinfo = new GlslVarInfo();
            int funcNum = varStm.GetFunctionNum();
            for (int ix = 0; ix < funcNum - 1; ++ix) {
                var valOrFunc = varStm.GetFunction(ix);
                string id = valOrFunc.GetId();
                if (ix == funcNum - 2) {
                    var func = valOrFunc.AsFunction;
                    if (null != func)
                        vinfo.Type = BuildGlslTypeWithTypeArgs(func) + arrTag;
                    else
                        vinfo.Type = id + arrTag;
                }
                else {
                    if (id == "inout")
                        vinfo.IsInOut = true;
                    else if (id == "out")
                        vinfo.IsOut = true;

                    vinfo.Modifiers.Add(id);
                }
            }
            vinfo.Name = lastId;
            return vinfo;
        }
        private static string BuildGlslTypeWithTypeArgs(Dsl.FunctionData func)
        {
            var sb = new StringBuilder();
            if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                var arrTags = new List<string>();
                string baseType = BuildGlslTypeWithArrTags(func, arrTags);
                sb.Append(baseType);
                for (int ix = arrTags.Count - 1; ix >= 0; --ix) {
                    sb.Append(arrTags[ix]);
                }
            }
            else {
                if (func.IsHighOrder) {
                    sb.Append(BuildGlslTypeWithTypeArgs(func.LowerOrderFunction));
                }
                else {
                    sb.Append(func.GetId());
                }
                foreach (var p in func.Params) {
                    sb.Append('|');
                    sb.Append(DslToGlslNameString(p));
                }
            }
            return sb.ToString();
        }
        private static string BuildGlslTypeWithArrTags(Dsl.FunctionData func, List<string> arrTags)
        {
            string ret = string.Empty;
            if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                if (func.IsHighOrder) {
                    ret = BuildGlslTypeWithArrTags(func.LowerOrderFunction, arrTags);
                }
                else {
                    ret = func.GetId();
                }
                string arrTag = "_x";
                if (func.GetParamNum() > 0) {
                    arrTag += func.GetParamId(0);
                }
                arrTags.Add(arrTag);
            }
            else {
                ret = BuildGlslTypeWithTypeArgs(func);
            }
            return ret;
        }
        private static string DslToGlslNameString(Dsl.ISyntaxComponent syntax)
        {
            var valData = syntax as Dsl.ValueData;
            if (null != valData)
                return valData.GetId();
            else {
                var funcData = syntax as Dsl.FunctionData;
                if (null != funcData) {
                    var sb = new StringBuilder();
                    if (funcData.IsHighOrder) {
                        sb.Append(DslToGlslNameString(funcData.LowerOrderFunction));
                    }
                    else {
                        sb.Append(funcData.GetId());
                    }
                    switch (funcData.GetParamClassUnmasked()) {
                        case (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PERIOD:
                            sb.Append(".");
                            break;
                        case (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET:
                            sb.Append("_x");
                            break;
                        default:
                            if (funcData.GetParamNum() > 0)
                                sb.Append("_");
                            break;
                    }
                    foreach (var p in funcData.Params) {
                        sb.Append(DslToGlslNameString(p));
                    }
                    return sb.ToString();
                }
                else {
                    var stmData = syntax as Dsl.StatementData;
                    if (null != stmData) {
                        var sb = new StringBuilder();
                        for (int ix = 0; ix < stmData.GetFunctionNum(); ++ix) {
                            if (ix > 0)
                                sb.Append("__");
                            var func = stmData.GetFunction(ix);
                            sb.Append(DslToGlslNameString(func));
                        }
                        return sb.ToString();
                    }
                    else {
                        return string.Empty;
                    }
                }
            }
        }
        private static bool IsGlslArgsMatch(IList<string> args, GlslFuncInfo funcInfo, out int score)
        {
            bool ret = false;
            score = 0;
            if (args.Count <= funcInfo.Params.Count) {
                ret = true;
                for (int ix = 0; ix < args.Count; ++ix) {
                    int argScore;
                    if (!IsGlslTypeMatch(args[ix], funcInfo.Params[ix].Type, out argScore)) {
                        ret = false;
                        break;
                    }
                    score += argScore;
                }
                if (ret && args.Count < funcInfo.Params.Count) {
                    if (null == funcInfo.Params[args.Count].DefaultValue)
                        ret = false;
                }
            }
            return ret;
        }
        private static bool IsGlslTypeMatch(string argType, string paramType, out int score)
        {
            bool ret = false;
            score = 0;
            if (argType == paramType) {
                score = 2;
                ret = true;
            }
            else if ((argType == "bool" || argType == "int" || argType == "uint" || argType == "float" || argType == "double")
                && (paramType == "bool" || paramType == "int" || paramType == "uint" || paramType == "float" || paramType == "double")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bvec2" || argType == "ivec2" || argType == "uvec3" || argType == "vec2" || argType == "dvec2")
                && (paramType == "bvec2" || paramType == "ivec2" || paramType == "uvec2" || paramType == "vec2" || paramType == "dvec2")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bvec3" || argType == "ivec3" || argType == "uvec3" || argType == "vec3" || argType == "dvec3")
                && (paramType == "bvec3" || paramType == "ivec3" || paramType == "uvec3" || paramType == "vec3" || paramType == "dvec3")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "bvec4" || argType == "ivec4" || argType == "uvec4" || argType == "vec4" || argType == "dvec4")
                && (paramType == "bvec4" || paramType == "ivec4" || paramType == "uvec4" || paramType == "vec4" || paramType == "dvec4")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "mat2" || argType == "dmat2") || (paramType == "mat2" || paramType == "dmat2")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "mat3" || argType == "dmat3") || (paramType == "mat3" || paramType == "dmat3")) {
                score = 1;
                ret = true;
            }
            else if ((argType == "mat4" || argType == "dmat4") || (paramType == "mat4" || paramType == "dmat4")) {
                score = 1;
                ret = true;
            }
            else if (argType == "bool" || argType == "bvec2" || argType == "bvec3" || argType == "bvec4") {
                ret = true;
            }
            else if (argType == "int" || argType == "ivec2" || argType == "ivec3" || argType == "ivec4") {
                ret = true;
            }
            else if (argType == "uint" || argType == "uvec2" || argType == "uvec3" || argType == "uvec4") {
                ret = true;
            }
            else if (argType == "float" || argType == "vec2" || argType == "vec3" || argType == "vec4") {
                ret = true;
            }
            else if (argType == "double" || argType == "dvec2" || argType == "dvec3" || argType == "dvec4") {
                ret = true;
            }
            return ret;
        }
        private static string TypeToShaderType(string type, out string arraySuffix)
        {
            string shaderType = GetGlslTypeRemoveArrTag(type, out var arrNums);
            if (arrNums.Count > 0) {
                var sb = new StringBuilder();
                for(int ix = arrNums.Count - 1; ix >= 0; --ix) {
                    sb.Append("[");
                    sb.Append(arrNums[ix]);
                    sb.Append("]");
                }
                arraySuffix = sb.ToString();
            }
            else {
                arraySuffix = string.Empty;
            }
            return shaderType;
        }
        private static string GetGlslTypeRemoveSuffix(string type)
        {
            type = GetGlslTypeRemoveArrTag(type, out var arrNums);
            if (type.Length >= 3) {
                char last = type[type.Length - 1];
                if (last == '2' || last == '3' || last == '4') {
                    string last3 = type.Substring(type.Length - 3);
                    if (last3 == "2x2" || last3 == "3x3" || last3 == "4x4")
                        return type.Substring(0, type.Length - 3);
                    else if (last3 == "2x3" || last3 == "3x2" || last3 == "3x4" || last3 == "4x3" || last3 == "2x4" || last3 == "4x2")
                        return type.Substring(0, type.Length - 3);
                    else
                        return type.Substring(0, type.Length - 1);
                }
            }
            return type;
        }
        private static string GetGlslTypeSuffix(string type)
        {
            return GetGlslTypeSuffix(type, out var arrNums);
        }
        private static string GetGlslTypeSuffix(string type, out IList<int> arrNums)
        {
            type = GetGlslTypeRemoveArrTag(type, out arrNums);
            if (type.Length >= 3) {
                char last = type[type.Length - 1];
                if (last == '2' || last == '3' || last == '4') {
                    string last3 = type.Substring(type.Length - 3);
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
        private static string GetGlslTypeSuffixReverse(string type)
        {
            return GetGlslTypeSuffixReverse(type, out var arrNums);
        }
        private static string GetGlslTypeSuffixReverse(string type, out IList<int> arrNums)
        {
            type = GetGlslTypeRemoveArrTag(type, out arrNums);
            if (type.Length >= 3) {
                char last = type[type.Length - 1];
                if (last == '2' || last == '3' || last == '4') {
                    string last3 = type.Substring(type.Length - 3);
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
        private static string GetGlslTypeRemoveArrTag(string type, out IList<int> arrNums)
        {
            var list = new List<int>();
            var r = GetGlslTypeRemoveArrTagRecursively(type, list);
            arrNums = list;
            return r;
        }
        private static string GetGlslTypeRemoveArrTagRecursively(string type, List<int> arrNums)
        {
            int st = type.LastIndexOf("_x");
            if (st > 0) {
                string arrNumStr = type.Substring(st + 2);
                if (int.TryParse(arrNumStr, out int arrNum)) {
                    arrNums.Add(arrNum);
                    type = GetGlslTypeRemoveArrTagRecursively(type.Substring(0, st), arrNums);
                }
            }
            return type;
        }
        private static string GlslVectorMatrixTypeToScalarType(string vm)
        {
            if (vm.StartsWith("vec"))
                return "float";
            else if (vm.StartsWith("ivec"))
                return "int";
            else if (vm.StartsWith("uvec"))
                return "uint";
            else if (vm.StartsWith("dvec"))
                return "double";
            else if (vm.StartsWith("bvec"))
                return "bool";
            else if (vm.StartsWith("mat"))
                return "float";
            else if (vm.StartsWith("dmat"))
                return "double";
            else
                return GetGlslTypeRemoveSuffix(vm);
        }

        private static void PreprocessHlsl(string srcFile, string outFile, ref bool isCompute)
        {
            bool isShader = false;
            bool isBegin = false;
            int beginCount = 0;
            var lines = File.ReadAllLines(srcFile);
            for (int ix = 0; ix < lines.Length; ++ix) {
                string line = lines[ix];
                string trimLine = line.Trim();
                if (trimLine.StartsWith(s_Unity3dComputePlatformPrefix)) {
                    ++beginCount;
                }
                if (trimLine.StartsWith(s_Unity3dComputeKernelPrefix)) {
                    isCompute = true;
                    ++beginCount;
                }
                if (trimLine.StartsWith(s_Unity3dComputeKeywordsPrefix)) {
                    ++beginCount;
                }
                if (trimLine.StartsWith(s_Unity3dShaderPrefix)) {
                    isShader = true;
                    ++beginCount;
                }
                if (trimLine.StartsWith(s_Unity3dSubShaderPrefix)) {
                    isShader = true;
                    ++beginCount;
                }
                if (trimLine.StartsWith(s_Unity3dPassPrefix)) {
                    isShader = true;
                    ++beginCount;
                }
                if (trimLine.StartsWith(s_Unity3dShaderCodeBeginPrefix)) {
                    ++beginCount;
                }
                if (!isBegin) {
                    if (!trimLine.StartsWith("//")) {
                        line = "//" + line;
                    }
                    lines[ix] = line;

                    if (isCompute && beginCount >= 3) {
                        isBegin = true;
                    }
                    else if (isShader && beginCount >= 4) {
                        isBegin = true;
                    }
                }
            }
            if (isShader) {
                int removeCt = 3;
                for (int ix = lines.Length - 1; ix >= 0; --ix) {
                    string line = lines[ix];
                    string trimLine = line.Trim();
                    if (trimLine == "}") {
                        lines[ix] = "//" + line;
                        --removeCt;
                    }
                    if (removeCt <= 0)
                        break;
                }
            }
            File.WriteAllLines(outFile, lines);
        }

        private static void AddGlslVar(GlslVarInfo varInfo)
        {
            if (!s_GlslVarInfos.TryGetValue(varInfo.Name, out var varInfos)) {
                varInfos = new Dictionary<int, GlslVarInfo>();
                s_GlslVarInfos.Add(varInfo.Name, varInfos);
            }
            varInfos[CurGlslBlockId()] = varInfo;
            SetLastGlslVarType(varInfo);
        }
        private static GlslFuncInfo? CurGlslFuncInfo()
        {
            GlslFuncInfo? curFuncInfo = null;
            if (s_GlslFuncParseStack.Count > 0) {
                curFuncInfo = s_GlslFuncParseStack.Peek();
            }
            return curFuncInfo;
        }
        private static void PushGlslFuncInfo(GlslFuncInfo funcInfo)
        {
            s_GlslFuncParseStack.Push(funcInfo);
        }
        private static void PopGlslFuncInfo()
        {
            s_GlslFuncParseStack.Pop();
        }
        private static void AddGlslParamInfo(GlslVarInfo varInfo)
        {
            var funcInfo = CurGlslFuncInfo();
            if (null != funcInfo) {
                funcInfo.Params.Add(varInfo);
                if (varInfo.IsInOut || varInfo.IsOut) {
                    funcInfo.HasInOutOrOutParams = true;
                    funcInfo.InOutOrOutParams.Add(varInfo);
                }
            }
        }
        private static void SetGlslRetInfo(GlslVarInfo varInfo)
        {
            var funcInfo = CurGlslFuncInfo();
            if (null != funcInfo) {
                funcInfo.RetInfo = varInfo;
            }
        }
        private static int CurGlslBlockId()
        {
            if (s_GlslLexicalScopeStack.Count > 0) {
                return s_GlslLexicalScopeStack.Peek();
            }
            return 0;
        }
        private static void PushGlslBlock()
        {
            ++s_LastGlslBlockId;
            s_GlslLexicalScopeStack.Push(s_LastGlslBlockId);
            s_LastGlslVarTypeStack.Push(null);
        }
        private static void PopGlslBlock()
        {
            s_LastGlslVarTypeStack.Pop();
            s_GlslLexicalScopeStack.Pop();
        }
        private static void SetLastGlslVarType(GlslVarInfo info)
        {
            if (s_LastGlslVarTypeStack.Count > 0) {
                s_LastGlslVarTypeStack.Pop();
                s_LastGlslVarTypeStack.Push(info);
            }
            else {
                s_LastGlslToplevelVarType = info;
            }
        }
        private static GlslVarInfo? GetLastGlslVarType()
        {
            if (s_LastGlslVarTypeStack.Count > 0) {
                return s_LastGlslVarTypeStack.Peek();
            }
            else {
                return s_LastGlslToplevelVarType;
            }
        }
        private static GlslVarInfo? GetGlslVarInfo(string name)
        {
            GlslVarInfo? varInfo = null;
            if (s_GlslVarInfos.TryGetValue(name, out var varInfos)) {
                bool find = false;
                foreach (var blockId in s_GlslLexicalScopeStack) {
                    if (varInfos.TryGetValue(blockId, out varInfo)) {
                        find = true;
                        break;
                    }
                }
                if (!find) {
                    find = varInfos.TryGetValue(0, out varInfo);
                }
            }
            if (null == varInfo) {
                var curFunc = CurGlslFuncInfo();
                if (null != curFunc) {
                    foreach (var p in curFunc.Params) {
                        if (name == p.Name) {
                            varInfo = p;
                            break;
                        }
                    }
                }
            }
            return varInfo;
        }

        internal sealed class GlslVarInfo
        {
            internal string Name = string.Empty;
            internal string Type = string.Empty;
            internal bool IsInOut = false;
            internal bool IsOut = false;
            internal List<string> Modifiers = new List<string>();
            internal Dsl.ISyntaxComponent? DefaultValue = null;

            internal void CopyFrom(GlslVarInfo other)
            {
                Name = other.Name;
                Type = other.Type;
                IsInOut = other.IsInOut;
                IsOut = other.IsOut;
                Modifiers.AddRange(other.Modifiers);
                DefaultValue = other.DefaultValue;
            }
            internal string CalcTypeString()
            {
                return (Modifiers.Count > 0 ? string.Join(' ', Modifiers) + " " : string.Empty) + Type;
            }
        }
        internal sealed class GlslFuncInfo
        {
            internal string Name = string.Empty;
            internal string Signature = string.Empty;
            internal bool HasInOutOrOutParams = false;
            internal List<GlslVarInfo> Params = new List<GlslVarInfo>();
            internal GlslVarInfo? RetInfo = null;
            internal List<GlslVarInfo> InOutOrOutParams = new List<GlslVarInfo>();

            internal bool IsVoid()
            {
                return null == RetInfo || RetInfo.Type == "void";
            }
        }
        internal sealed class GlslStructInfo
        {
            internal string Name = string.Empty;
            internal List<GlslVarInfo> Fields = new List<GlslVarInfo>();
        }
        internal sealed class GlslBufferInfo
        {
            internal string Name = string.Empty;
            internal string Layout = string.Empty;
            internal string instName = string.Empty;
            internal List<GlslVarInfo> Variables = new List<GlslVarInfo>();
        }
        internal sealed class ArrayInitInfo
        {
            internal string Type = string.Empty;
            internal int Size = 0;
        }
        internal enum GlslCondExpEnum
        {
            Question = 0,
            Colon,
        }
        internal sealed class GlslCondExpInfo
        {
            internal GlslCondExpInfo(GlslCondExpEnum part)
            {
                m_CondExpPart = part;
                m_ParenthesisCount = 0;
            }
            internal void IncParenthesisCount()
            {
                ++m_ParenthesisCount;
            }
            internal void DecParenthesisCount()
            {
                --m_ParenthesisCount;
            }
            internal bool MaybeCompletePart(GlslCondExpEnum part)
            {
                return m_CondExpPart == part && m_ParenthesisCount == 0;
            }

            private GlslCondExpEnum m_CondExpPart;
            private int m_ParenthesisCount;
        }

        private static Stack<GlslCondExpInfo> s_GlslCondExpStack = new Stack<GlslCondExpInfo>();
        private static SortedDictionary<string, GlslStructInfo> s_GlslStructInfos = new SortedDictionary<string, GlslStructInfo>();
        private static SortedDictionary<string, GlslBufferInfo> s_GlslBufferInfos = new SortedDictionary<string, GlslBufferInfo>();
        private static SortedDictionary<string, ArrayInitInfo> s_ArrayInits = new SortedDictionary<string, ArrayInitInfo>();

        private static GlslVarInfo? s_LastGlslToplevelVarType = null;
        private static Stack<GlslVarInfo?> s_LastGlslVarTypeStack = new Stack<GlslVarInfo?>();
        private static Dictionary<string, Dictionary<int, GlslVarInfo>> s_GlslVarInfos = new Dictionary<string, Dictionary<int, GlslVarInfo>>();
        private static Stack<int> s_GlslLexicalScopeStack = new Stack<int>();
        private static int s_LastGlslBlockId = 0;

        private static Dictionary<string, GlslFuncInfo> s_GlslFuncInfos = new Dictionary<string, GlslFuncInfo>();
        private static Dictionary<string, HashSet<string>> s_GlslFuncOverloads = new Dictionary<string, HashSet<string>>();
        private static Stack<GlslFuncInfo> s_GlslFuncParseStack = new Stack<GlslFuncInfo>();

        private static char[] s_eOrE = new char[] { 'e', 'E' };

        private static Dictionary<string, string> s_GlslBuiltInFuncs = new Dictionary<string, string> {
            { "float", "@@" },
            { "vec2", "@@" },
            { "vec3", "@@" },
            { "vec4", "@@" },
            { "double", "@@" },
            { "dvec2", "@@" },
            { "dvec3", "@@" },
            { "dvec4", "@@" },
            { "uint", "@@" },
            { "uvec2", "@@" },
            { "uvec3", "@@" },
            { "uvec4", "@@" },
            { "int", "@@" },
            { "ivec2", "@@" },
            { "ivec3", "@@" },
            { "ivec4", "@@" },
            { "bool", "@@" },
            { "bvec2", "@@" },
            { "bvec3", "@@" },
            { "bvec4", "@@" },
            { "mat2", "@@" },
            { "mat3", "@@" },
            { "mat4", "@@" },
            { "dmat2", "@@" },
            { "dmat3", "@@" },
            { "dmat4", "@@" },
            { "radians", "@0" },
            { "degrees", "@0" },
            { "sin", "@0" },
            { "cos", "@0" },
            { "tan", "@0" },
            { "asin", "@0" },
            { "acos", "@0" },
            { "atan", "@0" },
            { "sinh", "@0" },
            { "cosh", "@0" },
            { "tanh", "@0" },
            { "asinh", "@0" },
            { "acosh", "@0" },
            { "atanh", "@0" },
            { "pow", "@0" },
            { "exp", "@0" },
            { "log", "@0" },
            { "exp2", "@0" },
            { "log2", "@0" },
            { "sqrt", "@0" },
            { "inversesqrt", "@0" },
            { "abs", "@0" },
            { "sign", "@0" },
            { "floor", "@0" },
            { "trunc", "@0" },
            { "round", "@0" },
            { "roundEven", "@0" },
            { "ceil", "@0" },
            { "fract", "@0" },
            { "mod", "@0" },
            { "modf", "@0" },
            { "min", "@0" },
            { "max", "@0" },
            { "clamp", "@m" },
            { "mix", "@m" },
            { "step", "@m" },
            { "smoothstep", "@m" },
            { "isnan", "bool" },
            { "isinf", "bool" },
            { "floatBitsToInt", "int" },
            { "floatBitsToUint", "uint" },
            { "intBitsToFloat", "float" },
            { "uintBitsToFloat", "float" },
            { "fma", "float" },
            { "frexp", "float" },
            { "ldexp", "float" },
            { "packUnorm2x16", "uint" },
            { "packSnorm2x16", "uint" },
            { "packUnorm4x8", "uint" },
            { "packSnorm4x8", "uint" },
            { "unpackUnorm2x16", "vec2" },
            { "unpackSnorm2x16", "vec2" },
            { "unpackUnorm4x8", "vec4" },
            { "unpackSnorm4x8", "vec4" },
            { "packHalf2x16", "uint" },
            { "unpackHalf2x16", "vec2" },
            { "length", "float" },
            { "distance", "float" },
            { "dot", "float" },
            { "cross", "vec3" },
            { "normalize", "@0" },
            { "faceforward", "float" },
            { "reflect", "float" },
            { "refract", "float" },
            { "matrixCompMult", "@0" },
            { "outerProduct", "mat$1x$0" },
            { "transpose", "mat$R0" },
            { "determinant", "float" },
            { "inverse", "@0" },
            { "lessThan", "bvec" },
            { "lessThanEqual", "bvec" },
            { "greaterThan", "bvec" },
            { "greaterThanEqual", "bvec" },
            { "equal", "bvec" },
            { "notEqual", "bvec" },
            { "any", "bool" },
            { "all", "bool"},
            { "not", "bvec"},
            { "uaddCarry", "@0" },
            { "usubBorrow", "@0" },
            { "umulExtended", "void" },
            { "imulExtended", "void" },
            { "bitfieldExtrace", "@0" },
            { "bitfieldInsert", "@0" },
            { "bitfieldReverse", "@0" },
            { "bitCount", "@0" },
            { "findLSB", "@0" },
            { "findMSB", "@0" },
            { "textureSize", "ivec$0" },
            { "texture", "vec$0" },
            { "textureProj", "vec$0" },
            { "textureLod", "vec$0" },
            { "textureOffset", "vec$0" },
            { "texelFetch", "vec$0" },
            { "texelFetchOffset", "vec$0" },
            { "textureProjOffset", "vec$0" },
            { "textureLodOffset", "vec$0" },
            { "textureProjLod", "vec$0" },
            { "textureProjLodOffset", "vec$0" },
            { "textureGrad", "vec$0" },
            { "textureGradOffset", "vec$0" },
            { "textureProjGrad", "vec$0" },
            { "textureProjGradOffset", "vec$0" },
            { "textureGather", "vec$0" },
            { "textureGatherOffset", "vec$0" },
            { "atomicCounterIncrement", "uint" },
            { "atomicCounterDecrement", "uint" },
            { "atomicCounter", "uint" },
            { "atomicAdd", "@0" },
            { "atomicMin", "@0" },
            { "atomicMax", "@0" },
            { "atomicAnd", "@0" },
            { "atomicOr", "@0" },
            { "atomicXor", "@0" },
            { "atomicExchange", "@0" },
            { "aotmicCompSwap", "@0" },
            { "imageSize", "@0" },
            { "imageLoad", "vec4" },
            { "imageStore", "void" },
            { "imageAtomicAdd", "uint" },
            { "imageAtomicMin", "@0" },
            { "imageAtomicMax", "@0" },
            { "imageAtomicAnd", "@0" },
            { "imageAtomicOr", "@0" },
            { "imageAtomicXor", "@0" },
            { "imageAtomicExchange", "@0" },
            { "imageAtomicCompSwap", "@0" },
            { "dFdx", "@0" },
            { "dFdy", "@0" },
            { "fwidth", "@0" },
            { "interpolateAtCentroid", "@0" },
            { "interpolateAtSample", "@0" },
            { "interpolateAtOffset", "@0" },
            { "barrier", "void" },
            { "memoryBarrier", "void" },
            { "memoryBarrierAtomicCounter", "void" },
            { "memoryBarrierBuffer", "void" },
            { "memoryBarrierShared", "void" },
            { "memoryBarrierImage", "void" },
            { "groupMemoryBarrier", "void" },
            { "EmitVertex", "void" },
            { "EndPrimitive", "void" },
            { "subpassLoad", "vec4" },
        };

        private static string s_Unity3dComputePlatformPrefix = "**** Platform ";
        private static string s_Unity3dComputeKernelPrefix = "Preprocessed code for kernel ";
        private static string s_Unity3dComputeKeywordsPrefix = "keywords: ";

        private static string s_Unity3dShaderPrefix = "Shader ";
        private static string s_Unity3dSubShaderPrefix = "SubShader ";
        private static string s_Unity3dPassPrefix = "Pass ";
        private static string s_Unity3dShaderCodeBeginPrefix = "Preprocessed source:";
    }
}