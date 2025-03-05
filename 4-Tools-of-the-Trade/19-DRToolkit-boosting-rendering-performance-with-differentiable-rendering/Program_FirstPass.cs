using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Hlsl2Python
{
    internal struct SyntaxStackInfo
    {
        internal Dsl.ISyntaxComponent Syntax;
        internal int Index;

        internal SyntaxStackInfo(Dsl.ISyntaxComponent syntax, int index)
        {
            Syntax = syntax;
            Index = index;
        }
    }
    internal static class SyntaxSearcher
    {
        internal delegate bool SyntaxPredDelegation(Dsl.ISyntaxComponent info, int index, IEnumerable<SyntaxStackInfo> syntaxStack);

        internal static bool Search(Dsl.ISyntaxComponent info, SyntaxPredDelegation pred)
        {
            var syntaxStack = new Stack<SyntaxStackInfo>();
            return IterateSyntax(info, 0, syntaxStack, pred);
        }

        private static bool IterateSyntax(Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, SyntaxPredDelegation pred)
        {
            bool ret = false;
            ret = pred(info, index, syntaxStack);
            if (!ret) {
                var valData = info as Dsl.ValueData;
                var funcData = info as Dsl.FunctionData;
                var stmData = info as Dsl.StatementData;
                if (null != valData) {
                    ret = IterateValue(valData, index, syntaxStack, pred);
                }
                else if (null != funcData) {
                    ret = IterateFunction(funcData, index, syntaxStack, pred);
                }
                else if (null != stmData) {
                    ret = IterateStatement(stmData, index, syntaxStack, pred);
                }
            }
            return ret;
        }
        private static bool IterateValue(Dsl.ValueData valData, int index, Stack<SyntaxStackInfo> syntaxStack, SyntaxPredDelegation pred)
        {
            return false;
        }
        private static bool IterateFunction(Dsl.FunctionData funcData, int index, Stack<SyntaxStackInfo> syntaxStack, SyntaxPredDelegation pred)
        {
            bool ret = false;
            syntaxStack.Push(new SyntaxStackInfo(funcData, index));
            if (funcData.IsHighOrder) {
                ret = IterateSyntax(funcData.LowerOrderFunction, -1, syntaxStack, pred);
            }
            else {
                ret = IterateSyntax(funcData.Name, -1, syntaxStack, pred);
            }
            if (!ret) {
                int cindex = 0;
                foreach (var p in funcData.Params) {
                    ret = IterateSyntax(p, cindex, syntaxStack, pred);
                    if (ret)
                        break;
                    ++cindex;
                }
            }
            syntaxStack.Pop();
            return ret;
        }
        private static bool IterateStatement(Dsl.StatementData stmData, int index, Stack<SyntaxStackInfo> syntaxStack, SyntaxPredDelegation pred)
        {
            bool ret = false;
            int cindex = 0;
            syntaxStack.Push(new SyntaxStackInfo(stmData, index));
            foreach (var func in stmData.Functions) {
                ret = IterateSyntax(func, cindex, syntaxStack, pred);
                if (ret)
                    break;
                ++cindex;
            }
            syntaxStack.Pop();
            return ret;
        }
    }
    internal partial class Program
    {
        private static void ParseStruct(Dsl.ISyntaxComponent info)
        {
            var structFunc = info as Dsl.FunctionData;
            Debug.Assert(null != structFunc);
            string name = string.Empty;
            if (structFunc.IsHighOrder) {
                name = structFunc.LowerOrderFunction.GetParamId(0);
            }
            else {
                //forward declaration
                return;
            }
            var struInfo = new StructInfo();
            struInfo.Name = name;
            foreach (var p in structFunc.Params) {
                var func = p as Dsl.FunctionData;
                var stm = p as Dsl.StatementData;
                VarInfo varInfo;
                if (null != stm) {
                    func = stm.First.AsFunction;
                    Debug.Assert(null != func);
                    varInfo = ParseVarInfo(func, stm);
                }
                else {
                    Debug.Assert(null != func);
                    varInfo = ParseVarInfo(func, null);
                }
                struInfo.Fields.Add(varInfo);
                struInfo.FieldName2Indexes.Add(varInfo.Name, struInfo.Fields.Count - 1);
            }
            if (s_StructInfos.ContainsKey(struInfo.Name)) {
                Console.WriteLine("duplicated struct define '{0}', line: {1}", struInfo.Name, info.GetLine());
            }
            else {
                s_StructInfos.Add(struInfo.Name, struInfo);
            }
        }
        private static void ParseCBuffer(Dsl.ISyntaxComponent info)
        {
            var cbufFunc = info as Dsl.FunctionData;
            var cbufStm = info as Dsl.StatementData;
            Dsl.FunctionData? body = null;
            string name = string.Empty;
            string reg = string.Empty;
            if (null != cbufStm) {
                foreach (var f in cbufStm.Functions) {
                    var func = f.AsFunction;
                    Debug.Assert(null != func);
                    if (func.GetId() == "register") {
                        if (func.IsHighOrder)
                            reg = func.LowerOrderFunction.GetParamId(0);
                        else
                            reg = func.GetParamId(0);
                    }
                }
                body = cbufStm.Last.AsFunction;
                Debug.Assert(null != body);
            }
            else {
                Debug.Assert(null != cbufFunc);
                if (cbufFunc.IsHighOrder)
                    name = cbufFunc.LowerOrderFunction.GetParamId(0);
                body = cbufFunc;
            }
            var cbufInfo = new CBufferInfo();
            cbufInfo.Name = name;
            cbufInfo.Register = reg;
            foreach (var p in body.Params) {
                var func = p as Dsl.FunctionData;
                var stm = p as Dsl.StatementData;
                VarInfo varInfo;
                if (null != stm) {
                    func = stm.First.AsFunction;
                    Debug.Assert(null != func);
                    varInfo = ParseVarInfo(func, stm);
                }
                else {
                    Debug.Assert(null != func);
                    varInfo = ParseVarInfo(func, null);
                }
                cbufInfo.Variables.Add(varInfo);

                AddVar(varInfo);
            }
            if (s_CBufferInfos.ContainsKey(cbufInfo.Name)) {
                Console.WriteLine("duplicated cbuffer define '{0}', line: {1}", cbufInfo.Name, info.GetLine());
            }
            else {
                s_CBufferInfos.Add(cbufInfo.Name, cbufInfo);
            }
        }
        private static void ParseTypeDef(Dsl.ISyntaxComponent info)
        {
            var func = info as Dsl.FunctionData;
            if (null != func) {
                var pair = ParseTypeDef(func, out bool isConst);
                AddTypeDef(pair.Key, pair.Value);
            }
            else {
                var stm = info as Dsl.StatementData;
                if (null != stm) {
                    func = stm.First.AsFunction;
                    Debug.Assert(null != func);
                    var pair = ParseTypeDef(func, out bool isConst);
                    AddTypeDef(pair.Key, pair.Value);
                }
            }
        }
        private static VarInfo? ParseVarDecl(Dsl.ISyntaxComponent info)
        {
            VarInfo? ret = null;
            //Declare variables without initial values in Python and assign them initial values, otherwise, these variables may not be
            //introduced into the correct lexical scope (for example, accessing an undefined variable for an 'out' variable).
            var func = info as Dsl.FunctionData;
            if (null != func) {
                var varInfo = ParseVarInfo(func, null);
                AddVar(varInfo);
                ret = varInfo;
            }
            else {
                var stm = info as Dsl.StatementData;
                if (null != stm) {
                    func = stm.First.AsFunction;
                    Debug.Assert(null != func);
                    var varInfo = ParseVarInfo(func, stm);
                    AddVar(varInfo);
                    ret = varInfo;
                }
            }
            return ret;
        }
        private static string ParseFuncSignature(string funcName, Dsl.FunctionData paramsPart)
        {
            return ParseFuncSignature(funcName, paramsPart, null);
        }
        private static string ParseFuncSignature(string funcName, Dsl.FunctionData paramsPart, List<VarInfo>? paramInfos)
        {
            List<string> paramTypes = new List<string>();
            for (int ix = 0; ix < paramsPart.GetParamNum(); ++ix) {
                var pcomp = paramsPart.GetParam(ix);
                var pFunc = pcomp as Dsl.FunctionData;
                var pStm = pcomp as Dsl.StatementData;
                Dsl.ISyntaxComponent? defVal = null;
                if (null != pFunc && pFunc.GetId() == "=") {
                    defVal = pFunc.GetParam(1);
                    pcomp = pFunc.GetParam(0);
                    pFunc = pcomp as Dsl.FunctionData;
                    pStm = pcomp as Dsl.StatementData;
                }
                if (null != pFunc) {
                    var paramInfo = ParseVarInfo(pFunc, null);
                    if (null != paramInfos) {
                        paramInfo.DefaultValueSyntax = defVal;
                        paramInfos.Add(paramInfo);
                    }
                    paramTypes.Add(paramInfo.Type);
                }
                else if (null != pStm) {
                    var paramInfo = ParseVarInfo(pStm.First.AsFunction, pStm);
                    if (null != paramInfos) {
                        paramInfo.DefaultValueSyntax = defVal;
                        paramInfos.Add(paramInfo);
                    }
                    paramTypes.Add(paramInfo.Type);
                }
            }
            string signature = GetFullTypeFuncSig(funcName, paramTypes);
            return signature;
        }
        private static void ParseAndPruneFuncDef(Dsl.ISyntaxComponent info, string mainEntryFunc, HashSet<string> additionalEntryFuncs)
        {
            var stmData = info as Dsl.StatementData;
            if (null != stmData) {
                var firstFunc = stmData.First.AsFunction;
                var secondFunc = stmData.Second.AsFunction;
                var lastFunc = stmData.Last.AsFunction;
                var retInfo = ParseVarInfo(firstFunc, stmData);
                string funcName = retInfo.Name;
                if (secondFunc.IsHighOrder)
                    secondFunc = secondFunc.LowerOrderFunction;

                List<VarInfo> paramInfos = new List<VarInfo>();
                string signature = ParseFuncSignature(funcName, secondFunc, paramInfos);
                if (!s_FuncInfos.TryGetValue(signature, out var funcInfo)) {
                    funcInfo = new FuncInfo();

                    funcInfo.Name = funcName;
                    var blockInfo = new BlockInfo();
                    blockInfo.OwnerFunc = funcInfo;
                    blockInfo.Syntax = info;
                    blockInfo.FuncSyntaxIndex = stmData.GetFunctionNum() - 1;
                    funcInfo.ToplevelBlock = blockInfo;

                    SetRetInfo(retInfo, funcInfo);
                    foreach (var pinfo in paramInfos) {
                        AddParamInfo(pinfo, funcInfo);
                    }

                    funcInfo.Signature = signature;
                    if (!s_FuncInfos.ContainsKey(signature)) {
                        s_FuncInfos.Add(signature, funcInfo);
                    }
                    if (!s_FuncOverloads.TryGetValue(funcName, out var overloads)) {
                        overloads = new HashSet<string>();
                        s_FuncOverloads.Add(funcName, overloads);
                    }
                    if (!overloads.Contains(signature)) {
                        overloads.Add(signature);
                    }
                }

                Debug.Assert(null != funcInfo && null != funcInfo.ToplevelBlock);

                if (!lastFunc.HaveStatement()) {
                    //forward declaration
                    return;
                }
                else {
                    ClearParamInfo(funcInfo);
                    foreach (var pinfo in paramInfos) {
                        AddParamInfo(pinfo, funcInfo);
                    }
                }

                PushBlock(funcInfo.ToplevelBlock);

                int uid = GenUniqueNumber();
                string retValVar = string.Format("_func_ret_val_{0}", uid);
                string retFlagVar = string.Format("_func_ret_flag_{0}", uid);
                var retBlockInfo = new RetBlockInfo { RetValVar = retValVar, RetFlagVar = retFlagVar };
                PushRetBlockInfo(retBlockInfo);
                if (!funcInfo.ParseAndPruned) {
                    funcInfo.ParseAndPruned = true;

                    if (lastFunc.HaveStatement() && !s_AllFuncDsls.ContainsKey(signature)) {
                        s_AllFuncDsls.Add(signature, stmData);
                        
                        if(mainEntryFunc == funcName) {
                            s_MainEntryFuncSignature = signature;
                            s_EntryFuncs.Add(signature, stmData);
                        }
                        else if (additionalEntryFuncs.Contains(funcName)) {
                            s_EntryFuncs.Add(signature, stmData);
                        }
                    }
                    var syntaxStack = new Stack<SyntaxStackInfo>();
                    syntaxStack.Push(new SyntaxStackInfo(info, 0));
                    syntaxStack.Push(new SyntaxStackInfo(lastFunc, stmData.GetFunctionNum() - 1));
                    retBlockInfo.StatementNum = lastFunc.GetParamNum();
                    if (lastFunc.HaveStatement()) {
                        for (int ix = 0; ix < lastFunc.GetParamNum(); ++ix) {
                            var stm = lastFunc.GetParam(ix);
                            int newIx = ix;
                            newIx = ParseAndPruneStatement(ref stm, ix, syntaxStack, out var binfo);
                            lastFunc.SetParam(newIx, stm);

                            if (binfo.ExistsReturn) {
                                if (newIx < lastFunc.GetParamNum() - 1) {
                                    retBlockInfo.NeedCheckRetFlag = true;

                                    var newIf = BuildIfNotStatement("if", retBlockInfo.RetFlagVar);
                                    for (int leftIx = newIx + 1; leftIx < lastFunc.GetParamNum(); ++leftIx) {
                                        newIf.AddParam(lastFunc.GetParam(leftIx));
                                    }
                                    lastFunc.Params.RemoveRange(newIx + 1, lastFunc.GetParamNum() - newIx - 1);
                                    lastFunc.AddParam(newIf);
                                }
                                else {
                                    string lid = stm.GetId();
                                    if (lid == "return" || lid == "<-") {
                                    }
                                    else {
                                        retBlockInfo.NeedCheckRetFlag = true;
                                    }
                                }
                            }
                            ix = newIx;
                        }
                    }
                    if (retBlockInfo.NeedCheckRetFlag) {
                        bool needRewrite = true;
                        if (lastFunc.GetParamNum() == 1) {
                            string lid = lastFunc.GetParamId(0);
                            if (lid == "return" || lid == "<-") {
                                needRewrite = false;
                            }
                        }
                        if (needRewrite) {
                            var retFlagDef = BuildVarDefStatement("bool", retFlagVar, "false");
                            lastFunc.Params.Insert(0, retFlagDef);
                            if (!funcInfo.IsVoid()) {
                                Debug.Assert(null != funcInfo.RetInfo);
                                var retValDef = BuildVarDefStatement(funcInfo.RetInfo.Type, retValVar);
                                lastFunc.Params.Insert(0, retValDef);
                                lastFunc.AddParam(BuildReturnStatement(retValVar));
                            }
                        }
                    }
                    syntaxStack.Pop();
                    syntaxStack.Pop();
                }
                PopRetBlockInfo();
                PopBlock();
            }
        }
        private static void ParseAndPruneBlock(ref Dsl.FunctionData func, Stack<SyntaxStackInfo> syntaxStack, RetBlockInfo retBlockInfo, BreakBlockInfo? brBlockInfo, ContinueBlockInfo? contBlockInfo, out FlowBreakInfo breakInfo, out bool needCheckRetFlag, out bool needCheckBrFlag, out bool needCheckContFlag)
        {
            breakInfo = new FlowBreakInfo();
            needCheckRetFlag = false;
            needCheckBrFlag = false;
            needCheckContFlag = false;

            for (int ix = 0; ix < func.GetParamNum(); ++ix) {
                var stm = func.GetParam(ix);
                int newIx = ix;
                if (func.HaveStatement()) {
                    newIx = ParseAndPruneStatement(ref stm, ix, syntaxStack, out var binfo);
                    func.SetParam(newIx, stm);
                    breakInfo.ExistsReturn |= binfo.ExistsReturn;
                    breakInfo.ExistsBreak |= binfo.ExistsBreak;
                    breakInfo.ExistsContinue |= binfo.ExistsContinue;

                    if (binfo.ExistsReturn || binfo.ExistsContinue || binfo.ExistsBreak) {
                        var varNames = new List<string>();
                        if (binfo.ExistsReturn) {
                            needCheckRetFlag = true;
                            varNames.Add(retBlockInfo.RetFlagVar);
                        }
                        if (binfo.ExistsBreak) {
                            Debug.Assert(null != brBlockInfo);
                            needCheckBrFlag = true;
                            varNames.Add(brBlockInfo.BreakFlagVar);
                        }
                        if (binfo.ExistsContinue) {
                            Debug.Assert(null != contBlockInfo);
                            needCheckContFlag = true;
                            varNames.Add(contBlockInfo.ContinueFlagVar);
                        }
                        if (newIx < func.GetParamNum() - 1) {
                            var newIf = BuildIfAndNotStatement("if", varNames);
                            for (int leftIx = newIx + 1; leftIx < func.GetParamNum(); ++leftIx) {
                                newIf.AddParam(func.GetParam(leftIx));
                            }
                            func.Params.RemoveRange(newIx + 1, func.GetParamNum() - newIx - 1);
                            func.AddParam(newIf);
                        }
                    }
                }
                ix = newIx;
            }
        }
        private static int ParseAndPruneBlock(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            Dsl.FunctionData? func = info as Dsl.FunctionData;
            Debug.Assert(null != func);

            var retBlockInfo = CurRetBlockInfo();
            var brBlockInfo = CurBreakBlockInfo();
            var contBlockInfo = CurContinueBlockInfo();
            Debug.Assert(null != retBlockInfo);

            syntaxStack.Push(new SyntaxStackInfo(info, index));
            ParseAndPruneBlock(ref func, syntaxStack, retBlockInfo, brBlockInfo, contBlockInfo, out breakInfo, out var needCheckRetFlag, out var needCheckBrGlag, out var needCheckContFlag);
            info = func;
            syntaxStack.Pop();

            return index;
        }
        private static int ParseAndPruneIf(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            var retBlockInfo = CurRetBlockInfo();
            Debug.Assert(null != retBlockInfo);
            var brBlockInfo = CurBreakBlockInfo();
            var contBlockInfo = CurContinueBlockInfo();

            syntaxStack.Push(new SyntaxStackInfo(info, index));

            var ifFunc = info as Dsl.FunctionData;
            var ifStm = info as Dsl.StatementData;
            if (null != ifFunc) {
                ParseAndPruneBlock(ref ifFunc, syntaxStack, retBlockInfo, brBlockInfo, contBlockInfo, out breakInfo, out var needCheckRetFlag, out var needCheckBrFlag, out var needCheckContFlag);
            }
            else if (null != ifStm) {
                breakInfo = new FlowBreakInfo();
                for (int ix = 0; ix < ifStm.GetFunctionNum(); ++ix) {
                    var f = ifStm.GetFunction(ix);
                    var func = f.AsFunction;
                    if (null != func) {
                        syntaxStack.Push(new SyntaxStackInfo(func, ix));

                        ParseAndPruneBlock(ref func, syntaxStack, retBlockInfo, brBlockInfo, contBlockInfo, out var binfo, out var needCheckRetFlag, out var needCheckBrFlag, out var needCheckContFlag);
                        breakInfo.ExistsReturn |= binfo.ExistsReturn;
                        breakInfo.ExistsBreak |= binfo.ExistsBreak;
                        breakInfo.ExistsContinue |= binfo.ExistsContinue;

                        syntaxStack.Pop();
                    }
                }
            }
            else {
                breakInfo = new FlowBreakInfo();
            }

            syntaxStack.Pop();
            return index;
        }
        private static int ParseAndPruneSwitch(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            breakInfo = new FlowBreakInfo();

            var switchFunc = info as Dsl.FunctionData;
            if (null != switchFunc) {
                syntaxStack.Push(new SyntaxStackInfo(info, index));

                int uid = GenUniqueNumber();
                string brFlagVar = string.Format("_br_flag_{0}", uid);
                var brBlockInfo = new BreakBlockInfo { BreakFlagVar = brFlagVar };

                var retBlockInfo = CurRetBlockInfo();
                Debug.Assert(null != retBlockInfo);
                PushBreakBlockInfo(brBlockInfo);
                var contBlockInfo = CurContinueBlockInfo();
                brBlockInfo.StatementNum = switchFunc.GetParamNum();

                for (int ix = 0; ix < switchFunc.GetParamNum(); ++ix) {
                    var p = switchFunc.GetParam(ix);
                    var func = p as Dsl.FunctionData;
                    if (null != func) {
                        string curId = p.GetId();
                        if (curId == "case" || curId == "default") {
                            syntaxStack.Push(new SyntaxStackInfo(p, ix));

                            ParseAndPruneBlock(ref func, syntaxStack, retBlockInfo, brBlockInfo, contBlockInfo, out var myBreakInfo, out var needCheckRetFlag, out var needCheckBrFlag, out var needCheckContFlag);
                            breakInfo.ExistsReturn = myBreakInfo.ExistsReturn || breakInfo.ExistsReturn;
                            breakInfo.ExistsContinue = myBreakInfo.ExistsContinue || breakInfo.ExistsContinue;
                            if (needCheckBrFlag) {
                                var brFlagDef = BuildVarDefStatement("bool", brBlockInfo.BreakFlagVar, "false");
                                func.Params.Insert(0, brFlagDef);
                            }

                            syntaxStack.Pop();
                        }
                    }
                }

                PopBreakBlockInfo();

                syntaxStack.Pop();
            }
            return index;
        }
        private static int ParseAndPruneLoop(ref Dsl.FunctionData loopFunc, ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            int uid = GenUniqueNumber();
            string brFlagVar = string.Format("_br_flag_{0}", uid);
            string contFlagVar = string.Format("_cont_flag_{0}", uid);
            var brBlockInfo = new BreakBlockInfo { BreakFlagVar = brFlagVar };
            var contBlockInfo = new ContinueBlockInfo { ContinueFlagVar = contFlagVar };

            var retBlockInfo = CurRetBlockInfo();
            Debug.Assert(null != retBlockInfo);
            PushBreakBlockInfo(brBlockInfo);
            PushContinueBlockInfo(contBlockInfo);
            brBlockInfo.StatementNum = loopFunc.GetParamNum();
            contBlockInfo.StatementNum = loopFunc.GetParamNum();

            syntaxStack.Push(new SyntaxStackInfo(info, index));
            ParseAndPruneBlock(ref loopFunc, syntaxStack, retBlockInfo, brBlockInfo, contBlockInfo, out breakInfo, out var needCheckRetFlag, out var needCheckBrFlag, out var needCheckContFlag);
            syntaxStack.Pop();

            if (needCheckRetFlag || needCheckBrFlag || needCheckContFlag) {
                var outerFunc = GetOuterSyntax(syntaxStack) as Dsl.FunctionData;
                Debug.Assert(null != outerFunc);
                if (needCheckBrFlag) {
                    var brFlagDef = BuildVarDefStatement("bool", brBlockInfo.BreakFlagVar, "false");
                    outerFunc.Params.Insert(index, brFlagDef);
                    ++index;
                }
                if (needCheckContFlag) {
                    var contFlagDef = BuildVarDefStatement("bool", contBlockInfo.ContinueFlagVar, "false");
                    outerFunc.Params.Insert(index, contFlagDef);
                    var assignFunc = BuildAssignmentStatement(contBlockInfo.ContinueFlagVar, "false");
                    loopFunc.Params.Insert(0, assignFunc);
                    ++index;
                }
                var varNames = new List<string>();
                if (needCheckRetFlag) {
                    varNames.Add(retBlockInfo.RetFlagVar);
                }
                if (needCheckBrFlag) {
                    varNames.Add(brBlockInfo.BreakFlagVar);
                }

                if (varNames.Count > 0) {
                    var ifStm = new Dsl.StatementData();
                    var ifFunc = BuildIfAndNotStatement("if", varNames, Dsl.AbstractSyntaxComponent.SEPARATOR_NOTHING);
                    var elseFunc = BuildElseStatement("else_if_has_branches");
                    var breakStm = new Dsl.ValueData("break");
                    breakStm.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                    elseFunc.AddParam(breakStm);
                    ifStm.AddFunction(ifFunc);
                    ifStm.AddFunction(elseFunc);
                    ifFunc.Params.AddRange(loopFunc.Params);
                    ifStm.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);

                    loopFunc.Params.Clear();
                    loopFunc.AddParam(ifStm);
                }
            }

            PopContinueBlockInfo();
            PopBreakBlockInfo();

            return index;
        }
        private static int ParseAndPruneFor(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            var loopFunc = info as Dsl.FunctionData;
            Debug.Assert(null != loopFunc);
            var forFunc = loopFunc.LowerOrderFunction;
            Debug.Assert(null != forFunc);

            int newIndex = ParseAndPruneLoop(ref loopFunc, ref info, index, syntaxStack, out var myBreakInfo);
            breakInfo = new FlowBreakInfo();
            breakInfo.ExistsReturn = myBreakInfo.ExistsReturn;
            info = loopFunc;
            return newIndex;
        }
        private static int ParseAndPruneWhile(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            var loopFunc = info as Dsl.FunctionData;
            Debug.Assert(null != loopFunc);

            int newIndex = ParseAndPruneLoop(ref loopFunc, ref info, index, syntaxStack, out var myBreakInfo);
            breakInfo = new FlowBreakInfo();
            breakInfo.ExistsReturn = myBreakInfo.ExistsReturn;
            info = loopFunc;
            return newIndex;
        }
        private static int ParseAndPruneDoWhile(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            var loopStm = info as Dsl.StatementData;
            Debug.Assert(null != loopStm);
            var loopFunc0 = loopStm.First.AsFunction;
            Debug.Assert(null != loopFunc0);
            var loopFunc = loopFunc0;

            int newIndex = ParseAndPruneLoop(ref loopFunc, ref info, index, syntaxStack, out var myBreakInfo);
            breakInfo = new FlowBreakInfo();
            breakInfo.ExistsReturn = myBreakInfo.ExistsReturn;
            if (loopFunc != loopFunc0) {
                loopStm.SetFunction(0, loopFunc);
            }
            return newIndex;
        }
        private static int ParseAndPruneStatement(ref Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            breakInfo = new FlowBreakInfo();
            string id = info.GetId();
            if (id == "if") {
                if (s_EnableVectorization && !s_AutoVectorizeBranch) {
                    Console.WriteLine("[Info]: Cant vectorize statement '{0}', line: {1}", info.GetId(), info.GetLine());
                    MarkBranch();
                }
                ParseAndPruneIf(ref info, index, syntaxStack, out breakInfo);
            }
            else if (id == "switch") {
                if (s_EnableVectorization && !s_AutoVectorizeBranch) {
                    Console.WriteLine("[Info]: Cant vectorize statement '{0}', line: {1}", info.GetId(), info.GetLine());
                    MarkBranch();
                }
                ParseAndPruneSwitch(ref info, index, syntaxStack, out breakInfo);
            }
            else if (id == "for") {
                if (s_EnableVectorization && !s_AutoVectorizeBranch) {
                    Console.WriteLine("[Info]: Cant vectorize statement '{0}', line: {1}", info.GetId(), info.GetLine());
                }
                MarkBranch();
                return ParseAndPruneFor(ref info, index, syntaxStack, out breakInfo);
            }
            else if (id == "while") {
                if (s_EnableVectorization && !s_AutoVectorizeBranch) {
                    Console.WriteLine("[Info]: Cant vectorize statement '{0}', line: {1}", info.GetId(), info.GetLine());
                }
                MarkBranch();
                return ParseAndPruneWhile(ref info, index, syntaxStack, out breakInfo);
            }
            else if (id == "do") {
                if (s_EnableVectorization && !s_AutoVectorizeBranch) {
                    Console.WriteLine("[Info]: Cant vectorize statement '{0}', line: {1}", info.GetId(), info.GetLine());
                }
                MarkBranch();
                return ParseAndPruneDoWhile(ref info, index, syntaxStack, out breakInfo);
            }
            else if (id == "block") {
                return ParseAndPruneBlock(ref info, index, syntaxStack, out breakInfo);
            }
            else if (id == "return") {
                breakInfo.ExistsReturn = true;
                var funcBlockInfo = CurRetBlockInfo();
                Debug.Assert(null != funcBlockInfo);

                bool needRewrite = true;
                var body = GetOuterSyntaxs2(syntaxStack, out var func);
                if (null != func && func.GetId() == "func") {
                    //If there is only one return statement in the function or no return statement before this one, there is no need to rewrite this return statement.
                    if (funcBlockInfo.StatementNum == 1 || !funcBlockInfo.NeedCheckRetFlag) {
                        needRewrite = false;
                    }
                }

                if (needRewrite) {
                    var retFunc = info as Dsl.FunctionData;
                    Debug.Assert(null != retFunc);

                    var assignFunc = BuildAssignmentStatement(funcBlockInfo.RetFlagVar, "true");
                    if (retFunc.GetParamNum() > 0) {
                        var p = GetOuterSyntax(syntaxStack) as Dsl.FunctionData;
                        Debug.Assert(null != p);
                        p.Params.Insert(index, assignFunc);

                        retFunc.Name.SetId("=");
                        retFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                        retFunc.Params.Insert(0, new Dsl.ValueData(funcBlockInfo.RetValVar));

                        ++index;
                    }
                    else {
                        info = assignFunc;
                    }
                }
            }
            else if (id == "<-") {
                breakInfo.ExistsReturn = true;
                var funcBlockInfo = CurRetBlockInfo();
                Debug.Assert(null != funcBlockInfo);

                bool needRewrite = true;
                var body = GetOuterSyntaxs2(syntaxStack, out var func);
                if (null != func && func.GetId() == "func") {
                    //If there is only one return statement in the function or no return statement before this one, there is no need to rewrite this return statement.
                    if (funcBlockInfo.StatementNum == 1 || !funcBlockInfo.NeedCheckRetFlag) {
                        needRewrite = false;
                    }
                }

                if (needRewrite) {
                    var retFunc = info as Dsl.FunctionData;
                    Debug.Assert(null != retFunc);

                    var assignFunc = BuildAssignmentStatement(funcBlockInfo.RetFlagVar, "true");
                    var p = GetOuterSyntax(syntaxStack) as Dsl.FunctionData;
                    Debug.Assert(null != p);
                    p.Params.Insert(index, assignFunc);

                    retFunc.Name.SetId("=");
                    retFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                    retFunc.SetParam(0, new Dsl.ValueData(funcBlockInfo.RetValVar));

                    ++index;
                }
            }
            else if (id == "break") {
                breakInfo.ExistsBreak = true;
                var brBlockInfo = CurBreakBlockInfo();
                Debug.Assert(null != brBlockInfo);

                var assignFunc = BuildAssignmentStatement(brBlockInfo.BreakFlagVar, "true");
                info = assignFunc;
            }
            else if (id == "continue") {
                breakInfo.ExistsContinue = true;
                var contBlockInfo = CurContinueBlockInfo();
                Debug.Assert(null != contBlockInfo);

                var assignFunc = BuildAssignmentStatement(contBlockInfo.ContinueFlagVar, "true");
                info = assignFunc;
            }
            else {
                IterateSyntax(info, index, syntaxStack, out breakInfo);
            }
            return index;
        }
        private static void IterateSyntax(Dsl.ISyntaxComponent info, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            var funcData = info as Dsl.FunctionData;
            var stmData = info as Dsl.StatementData;
            if (null != funcData) {
                IterateFunction(funcData, index, syntaxStack, out breakInfo);
            }
            else if (null != stmData) {
                IterateStatement(stmData, index, syntaxStack, out breakInfo);
            }
            else {
                breakInfo = new FlowBreakInfo();
            }
        }
        private static void IterateFunction(Dsl.FunctionData funcData, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            breakInfo = new FlowBreakInfo();
            syntaxStack.Push(new SyntaxStackInfo(funcData, index));
            if (funcData.IsHighOrder) {
                IterateFunction(funcData.LowerOrderFunction, -1, syntaxStack, out var binfo);
                breakInfo.ExistsReturn |= binfo.ExistsReturn;
                breakInfo.ExistsBreak |= binfo.ExistsBreak;
                breakInfo.ExistsContinue |= binfo.ExistsContinue;
            }
            for (int ix = 0; ix < funcData.GetParamNum(); ++ix) {
                var p = funcData.GetParam(ix);
                int newIx = ix;
                if (funcData.HaveStatement()) {
                    newIx = ParseAndPruneStatement(ref p, ix, syntaxStack, out var binfo);
                    funcData.SetParam(newIx, p);
                    breakInfo.ExistsReturn |= binfo.ExistsReturn;
                    breakInfo.ExistsBreak |= binfo.ExistsBreak;
                    breakInfo.ExistsContinue |= binfo.ExistsContinue;
                }
                else {
                    IterateSyntax(p, newIx, syntaxStack, out var binfo);
                    breakInfo.ExistsReturn |= binfo.ExistsReturn;
                    breakInfo.ExistsBreak |= binfo.ExistsBreak;
                    breakInfo.ExistsContinue |= binfo.ExistsContinue;
                }
                ix = newIx;
            }
            syntaxStack.Pop();
        }
        private static void IterateStatement(Dsl.StatementData stmData, int index, Stack<SyntaxStackInfo> syntaxStack, out FlowBreakInfo breakInfo)
        {
            breakInfo = new FlowBreakInfo();
            syntaxStack.Push(new SyntaxStackInfo(stmData, index));
            for (int ix = 0; ix < stmData.GetFunctionNum(); ++ix) {
                var func = stmData.GetFunction(ix);
                var f = func.AsFunction;
                if (null != f) {
                    IterateFunction(f, ix, syntaxStack, out var binfo);
                    breakInfo.ExistsReturn |= binfo.ExistsReturn;
                    breakInfo.ExistsBreak |= binfo.ExistsBreak;
                    breakInfo.ExistsContinue |= binfo.ExistsContinue;
                }
            }
            syntaxStack.Pop();
        }

        private static Dsl.FunctionData BuildVarDefStatement(string varType, string varName)
        {
            return BuildVarDefStatement(varType, varName, string.Empty);
        }
        private static Dsl.FunctionData BuildVarDefStatement(string varType, string varName, string initVal)
        {
            var varFunc = new Dsl.FunctionData();
            var vSpec = new Dsl.FunctionData();
            var vType = new Dsl.ValueData(varType);
            var vName = new Dsl.ValueData(varName);
            vSpec.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_COMMA);
            vType.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_COMMA);

            varFunc.Name = new Dsl.ValueData("var");
            varFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);
            vSpec.Name = new Dsl.ValueData("spec");
            vSpec.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET);
            vSpec.AddParam("nothing");
            varFunc.AddParam(vSpec);
            varFunc.AddParam(vType);
            varFunc.AddParam(vName);

            if (string.IsNullOrEmpty(initVal)) {
                varFunc.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                return varFunc;
            }
            else {
                var newVarDef = new Dsl.FunctionData();
                var iVal = new Dsl.ValueData(initVal);
                newVarDef.Name = new Dsl.ValueData("=", Dsl.ValueData.ID_TOKEN);
                newVarDef.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                newVarDef.AddParam(varFunc);
                newVarDef.AddParam(iVal);

                newVarDef.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                return newVarDef;
            }
        }
        private static Dsl.FunctionData BuildIfNotStatement(string ifId, string boolVarName, int sep = Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON)
        {
            var newIf = BuildIfAndNotStatement(ifId, new string[] { boolVarName }, sep);
            return newIf;
        }
        private static Dsl.FunctionData BuildIfAndNotStatement(string ifId, IList<string> boolVarNames, int sep = Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON)
        {
            var newIf = new Dsl.FunctionData();
            var ifHead = new Dsl.FunctionData();

            ifHead.Name = new Dsl.ValueData(ifId);
            ifHead.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);

            var andNotExp = BuildAndNotExpression(boolVarNames);
            ifHead.AddParam(andNotExp);
            newIf.LowerOrderFunction = ifHead;
            newIf.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_STATEMENT);

            newIf.SetSeparator(sep);
            return newIf;
        }
        private static Dsl.FunctionData BuildIfStatement(string ifId, string boolVarName, int sep = Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON)
        {
            var newIf = BuildIfOrStatement(ifId, new string[] { boolVarName }, sep);
            return newIf;
        }
        private static Dsl.FunctionData BuildIfOrStatement(string ifId, IList<string> boolVarNames, int sep = Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON)
        {
            var newIf = new Dsl.FunctionData();
            var ifHead = new Dsl.FunctionData();

            ifHead.Name = new Dsl.ValueData(ifId);
            ifHead.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);

            var orExp = BuildOrExpression(boolVarNames);
            ifHead.AddParam(orExp);
            newIf.LowerOrderFunction = ifHead;
            newIf.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_STATEMENT);

            newIf.SetSeparator(sep);
            return newIf;
        }
        private static Dsl.FunctionData BuildElseStatement(string elseId)
        {
            var newIf = new Dsl.FunctionData();
            newIf.Name = new Dsl.ValueData(elseId);
            newIf.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_STATEMENT);

            newIf.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
            return newIf;
        }

        private static Dsl.FunctionData BuildAssignmentStatement(string varName, string val)
        {
            var assignFunc = new Dsl.FunctionData();
            var varV = new Dsl.ValueData(varName);
            var valV = new Dsl.ValueData(val);
            assignFunc.Name = new Dsl.ValueData("=", Dsl.ValueData.ID_TOKEN);
            assignFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
            assignFunc.AddParam(varV);
            assignFunc.AddParam(valV);

            assignFunc.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
            return assignFunc;
        }
        private static Dsl.FunctionData BuildReturnStatement(string varName)
        {
            var retFunc = new Dsl.FunctionData();
            var retKey = new Dsl.ValueData("return");
            var retVal = new Dsl.ValueData(varName);
            retFunc.Name = new Dsl.ValueData("<-", Dsl.ValueData.ID_TOKEN);
            retFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
            retFunc.AddParam(retKey);
            retFunc.AddParam(retVal);

            retFunc.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
            return retFunc;
        }
        private static Dsl.FunctionData BuildAndNotExpression(IList<string> boolVarNames)
        {
            Debug.Assert(boolVarNames.Count > 0);

            var notExps = new List<Dsl.FunctionData>();
            foreach (var boolVarName in boolVarNames) {
                var notExp = new Dsl.FunctionData();
                notExp.Name = new Dsl.ValueData("!", Dsl.ValueData.ID_TOKEN);
                notExp.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                notExp.AddParam(boolVarName);
                notExps.Add(notExp);
            }

            Dsl.FunctionData retFunc = notExps[0];
            if (boolVarNames.Count > 1) {
                var andExp = new Dsl.FunctionData();
                andExp.Name = new Dsl.ValueData("&&", Dsl.ValueData.ID_TOKEN);
                andExp.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);

                andExp.AddParam(notExps[0]);
                for (int i = 1; i < notExps.Count; ++i) {
                    andExp.AddParam(notExps[i]);

                    if (i < notExps.Count - 1) {
                        var newAndExp = new Dsl.FunctionData();
                        newAndExp.Name = new Dsl.ValueData("&&", Dsl.ValueData.ID_TOKEN);
                        newAndExp.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                        newAndExp.AddParam(andExp);

                        andExp = newAndExp;
                    }
                }
                retFunc = andExp;
            }

            return retFunc;
        }
        private static Dsl.ValueOrFunctionData BuildOrExpression(IList<string> boolVarNames)
        {
            Debug.Assert(boolVarNames.Count > 0);

            Dsl.ValueOrFunctionData retFunc = new Dsl.ValueData(boolVarNames[0], Dsl.ValueData.ID_TOKEN);
            if (boolVarNames.Count > 1) {
                var orExp = new Dsl.FunctionData();
                orExp.Name = new Dsl.ValueData("||", Dsl.ValueData.ID_TOKEN);
                orExp.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);

                orExp.AddParam(boolVarNames[0]);
                for (int i = 1; i < boolVarNames.Count; ++i) {
                    orExp.AddParam(boolVarNames[i]);

                    if (i < boolVarNames.Count - 1) {
                        var newOrExp = new Dsl.FunctionData();
                        newOrExp.Name = new Dsl.ValueData("||", Dsl.ValueData.ID_TOKEN);
                        newOrExp.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                        newOrExp.AddParam(orExp);

                        orExp = newOrExp;
                    }
                }
                retFunc = orExp;
            }

            return retFunc;
        }

        private static Dsl.ISyntaxComponent? GetOuterSyntax(IEnumerable<SyntaxStackInfo> syntaxStack)
        {
            foreach (var p in syntaxStack) {
                return p.Syntax;
            }
            return null;
        }
        private static Dsl.ISyntaxComponent? GetOuterSyntaxs2(IEnumerable<SyntaxStackInfo> syntaxStack, out Dsl.ISyntaxComponent? pp)
        {
            Dsl.ISyntaxComponent? first = null;
            Dsl.ISyntaxComponent? second = null;
            foreach (var p in syntaxStack) {
                if (null == first)
                    first = p.Syntax;
                else if (null == second)
                    second = p.Syntax;
                else
                    break;
            }
            pp = second;
            return first;
        }
        private static Dsl.ISyntaxComponent? GetOuterSyntaxs3(IEnumerable<SyntaxStackInfo> syntaxStack, out Dsl.ISyntaxComponent? pp, out Dsl.ISyntaxComponent? ppp)
        {
            Dsl.ISyntaxComponent? first = null;
            Dsl.ISyntaxComponent? second = null;
            Dsl.ISyntaxComponent? third = null;
            foreach (var p in syntaxStack) {
                if (null == first)
                    first = p.Syntax;
                else if (null == second)
                    second = p.Syntax;
                else if (null == third)
                    third = p.Syntax;
                else
                    break;
            }
            pp = second;
            ppp = third;
            return first;
        }
        private static SyntaxStackInfo? GetOuterSyntaxInfo(IEnumerable<SyntaxStackInfo> syntaxStack)
        {
            foreach (var p in syntaxStack) {
                return p;
            }
            return null;
        }
        private static SyntaxStackInfo? GetOuterSyntaxInfos2(IEnumerable<SyntaxStackInfo> syntaxStack, out SyntaxStackInfo? pp)
        {
            SyntaxStackInfo? first = null;
            SyntaxStackInfo? second = null;
            foreach (var p in syntaxStack) {
                if (null == first)
                    first = p;
                else if (null == second)
                    second = p;
                else
                    break;
            }
            pp = second;
            return first;
        }
        private static SyntaxStackInfo? GetOuterSyntaxInfos3(IEnumerable<SyntaxStackInfo> syntaxStack, out SyntaxStackInfo? pp, out SyntaxStackInfo? ppp)
        {
            SyntaxStackInfo? first = null;
            SyntaxStackInfo? second = null;
            SyntaxStackInfo? third = null;
            foreach (var p in syntaxStack) {
                if (null == first)
                    first = p;
                else if (null == second)
                    second = p;
                else if (null == third)
                    third = p;
                else
                    break;
            }
            pp = second;
            ppp = third;
            return first;
        }

        private static RetBlockInfo? CurRetBlockInfo()
        {
            RetBlockInfo? ret = null;
            if (s_RetBlockInfoStack.Count > 0) {
                ret = s_RetBlockInfoStack.Peek();
            }
            return ret;
        }
        private static BreakBlockInfo? CurBreakBlockInfo()
        {
            BreakBlockInfo? ret = null;
            if (s_BreakBlockInfoStack.Count > 0) {
                ret = s_BreakBlockInfoStack.Peek();
            }
            return ret;
        }
        private static ContinueBlockInfo? CurContinueBlockInfo()
        {
            ContinueBlockInfo? ret = null;
            if (s_ContinueBlockInfoStack.Count > 0) {
                ret = s_ContinueBlockInfoStack.Peek();
            }
            return ret;
        }
        private static void PushRetBlockInfo(RetBlockInfo info)
        {
            s_RetBlockInfoStack.Push(info);
        }
        private static void PopRetBlockInfo()
        {
            s_RetBlockInfoStack.Pop();
        }
        private static void PushBreakBlockInfo(BreakBlockInfo info)
        {
            s_BreakBlockInfoStack.Push(info);
        }
        private static void PopBreakBlockInfo()
        {
            s_BreakBlockInfoStack.Pop();
        }
        private static void PushContinueBlockInfo(ContinueBlockInfo info)
        {
            s_ContinueBlockInfoStack.Push(info);
        }
        private static void PopContinueBlockInfo()
        {
            s_ContinueBlockInfoStack.Pop();
        }

        internal ref struct FlowBreakInfo
        {
            internal bool ExistsReturn;
            internal bool ExistsBreak;
            internal bool ExistsContinue;

            public FlowBreakInfo()
            {
                ExistsReturn = false;
                ExistsBreak = false;
                ExistsContinue = false;
            }
        }
        internal sealed class RetBlockInfo
        {
            internal int StatementNum = 0;
            internal string RetValVar = string.Empty;
            internal string RetFlagVar = string.Empty;
            internal bool NeedCheckRetFlag = false;
        }
        internal sealed class BreakBlockInfo
        {
            internal int StatementNum = 0;
            internal string BreakFlagVar = string.Empty;
            internal bool NeedCheckBreakFlag = false;
        }
        internal sealed class ContinueBlockInfo
        {
            internal int StatementNum = 0;
            internal string ContinueFlagVar = string.Empty;
            internal bool NeedCheckContinueFlag = false;
        }
        private static Stack<RetBlockInfo> s_RetBlockInfoStack = new Stack<RetBlockInfo>();
        private static Stack<BreakBlockInfo> s_BreakBlockInfoStack = new Stack<BreakBlockInfo>();
        private static Stack<ContinueBlockInfo> s_ContinueBlockInfoStack = new Stack<ContinueBlockInfo>();
    }
}