using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Hlsl2Python
{
    internal partial class Program
    {
        private static void Vectorizing(StringBuilder sb)
        {
            //s_AllUsingFuncOrApis.Clear();
            //s_AutoGenCodes.Clear();
            //s_CalledScalarFuncs.Clear();
            SwapScalarCallInfo();
            MergeGlobalCallInfo();
            s_CalledVecFuncQueue.Clear();
            s_VecFuncCodeStack.Clear();
            foreach (var pair in s_EntryFuncs) {
                var entryFunc = pair.Key;
                if (s_FuncInfos.TryGetValue(entryFunc, out var funcInfo)) {
                    var argTypes = new List<string>();
                    foreach (var p in funcInfo.Params) {
                        argTypes.Add(GetTypeVec(p.Type));
                    }
                    VectorizeFunc(funcInfo, argTypes);
                }
            }
            while (s_CalledVecFuncQueue.Count > 0) {
                var vecInfo = s_CalledVecFuncQueue.Dequeue();
                if (s_AllFuncDsls.TryGetValue(vecInfo.FuncSignature, out var stmData)) {
                    vecInfo.ModifyFuncInfo();
                    TransformFunc(stmData);
                }
            }
            foreach (var sig in s_AllFuncSigs) {
                if (s_CalledScalarFuncs.Contains(sig)) {
                    if (s_AllFuncCodes.TryGetValue(sig, out var fsb)) {
                        sb.Append(fsb);
                        if (s_FuncInfos.TryGetValue(sig, out var fi)) {
                            sb.AppendLine();
                        }
                    }
                }
            }
            while (s_VecFuncCodeStack.Count > 0) {
                var vfc = s_VecFuncCodeStack.Pop();
                sb.AppendLine();
                sb.Append(vfc.VecFuncStringBuilder);
            }
            foreach (var fn in s_CalledScalarFuncs) {
                if (s_FuncInfos.TryGetValue(fn, out var fi)) {
                    foreach (var f in fi.UsingFuncOrApis) {
                        if (s_ScalarAutoGenCodes.TryGetValue(f, out var fsb)) {
                            if (!s_AutoGenCodes.ContainsKey(f))
                                s_AutoGenCodes.Add(f, fsb);
                        }
                    }
                }
            }
            s_IsVectorizing = false;
        }
        private static void VectorizeFunc(FuncInfo funcInfo, IList<string> args)
        {
            bool find = false;
            foreach (var vinfo in funcInfo.Vectorizations) {
                var vf = vinfo.VecFuncInfo;
                Debug.Assert(null != vf);
                if (vinfo.VecArgTypes.Count == args.Count) {
                    bool same = true;
                    for (int ix = 0; ix < vinfo.VecArgTypes.Count; ++ix) {
                        string t1 = vinfo.VecArgTypes[ix];
                        string t2 = args[ix];
                        var p = vf.Params[ix];
                        bool isOut = p.IsOut;
                        if (isOut) {
                            //The type of the out parameter is not used as a judgment basis.
                            continue;
                        }
                        if (t1 != t2) {
                            same = false;
                            break;
                        }
                    }
                    if (same) {
                        find = true;
                        vinfo.ModifyFuncInfo();
                        break;
                    }
                }
            }
            if (find)
                return;
            bool isVec = false;
            var vecInfo = new VectorialFuncInfo();
            for (int ix = 0; ix < funcInfo.Params.Count; ++ix) {
                var p = funcInfo.Params[ix];
                if (ix < args.Count) {
                    string realType = args[ix];
                    bool isOut = p.IsOut;
                    bool realTypeIsVec = false;
                    if (s_StructInfos.TryGetValue(GetTypeNoVecPrefix(realType), out var struInfo)) {
                        if (realType != p.Type)
                            isVec = true;
                    }
                    else if (IsTypeVec(realType)) {
                        realTypeIsVec = true;
                        isVec = true;
                    }
                    if (!isOut) {
                        if (realTypeIsVec) {
                            p.Type = GetTypeVec(p.Type);
                        }
                        else if (realType != p.Type) {
                            p.Type = realType;
                        }
                    }
                }
                else if (null != p.DefaultValueSyntax) {

                }
                else {
                    Debug.Assert(false);
                }
                vecInfo.VecArgTypes.Add(p.Type);
            }
            if (isVec) {
                vecInfo.FuncSignature = funcInfo.Signature;
                vecInfo.VecFuncInfo = funcInfo;
                vecInfo.VectorizeNo = funcInfo.Vectorizations.Count;

                if (!funcInfo.IsVoid()) {
                    Debug.Assert(null != funcInfo.RetInfo);
                    string retType = funcInfo.RetInfo.Type;
                    funcInfo.RetInfo.Type = GetTypeVec(retType);
                    vecInfo.VecRetType = funcInfo.RetInfo.Type;
                }
                if (funcInfo.HasBranches) {
                    MarkCalledScalarFunc(funcInfo, true);
                }
                else {
                    funcInfo.ClearForReTransform();
                    funcInfo.VectorizeNo = vecInfo.VectorizeNo;
                    funcInfo.Vectorizations.Add(vecInfo);
                    s_CalledVecFuncQueue.Enqueue(vecInfo);
                }
            }
            else {
                MarkCalledScalarFunc(funcInfo, false);
            }
        }
        private static bool VectorizeVar(Dsl.ISyntaxComponent info, out string broadcastVarName, out bool needBroadcastObj)
        {
            // Array vectorization is the vectorization of array elements, while structure vectorization is the
            // vectorization of structure fields.
            // When struct is vectorized, all fields are vectorized together, which is consistent with ordinary
            // variables (arrays are also vectorized together with all elements, otherwise it is not an array).
            // This makes the processing much simpler.
            // (When used as a function parameter, there will be many vectorized versions of the function. If it
            // is also used as a function return value, the return value can only be vectorized all together,
            // otherwise the vectorization signature of the function cannot be deduced before the function body
            // is processed. The conversion between different versions of local vectorization will also be very
            // complicated. Moreover, there may be nested structures, which may need to be avoided in actual use.)
            bool ret = false;
            needBroadcastObj = false;
            broadcastVarName = string.Empty;
            var vd = info as Dsl.ValueData;
            if (null != vd) {
                var varInfo = GetVarInfo(vd.GetId(), VarUsage.Find);
                if (null != varInfo) {
                    varInfo.Type = GetTypeVec(varInfo.Type, out var isTuple, out var isStruct, out var isVecBefore);
                    if (!isVecBefore) {
                        broadcastVarName = varInfo.Name;
                        ret = true;

                        if (null == varInfo.OwnerBlock) {
                            Console.WriteLine("[Error]: vectorize global var '{0}', please change it to a local var, line: {1}", info.GetId(), info.GetLine());
                        }
                    }
                }
                else {
                    Console.WriteLine("[Error]: can't vectorize var '{0}', line: {1}", info.GetId(), info.GetLine());
                }
            }
            else {
                var func = info as Dsl.FunctionData;
                if (null != func) {
                    if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PERIOD) {
                        //Object vectorization.
                        needBroadcastObj = true;
                        if (func.IsHighOrder) {
                            ret = VectorizeVar(func.LowerOrderFunction, out broadcastVarName, out var _);
                        }
                        else {
                            ret = VectorizeVar(func.Name, out broadcastVarName, out var _);
                        }
                    }
                    else if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                        //Array vectorization.
                        needBroadcastObj = true;
                        if (func.IsHighOrder) {
                            ret = VectorizeVar(func.LowerOrderFunction, out broadcastVarName, out var _);
                        }
                        else {
                            ret = VectorizeVar(func.Name, out broadcastVarName, out var _);
                        }
                    }
                    else {
                        var varInfo = ParseVarInfo(func, null);
                        if (!string.IsNullOrEmpty(varInfo.Name)) {
                            varInfo = GetVarInfo(varInfo.Name, VarUsage.Find);
                            if (null != varInfo) {
                                varInfo.Type = GetTypeVec(varInfo.Type, out var isTuple, out var isStruct, out var isVecBefore);
                                if (!isVecBefore) {
                                    broadcastVarName = varInfo.Name;
                                    ret = true;

                                    if (null == varInfo.OwnerBlock) {
                                        Console.WriteLine("[Error]: vectorize global var '{0}', please change it to a local var, line: {1}", info.GetId(), info.GetLine());
                                    }
                                }
                            }
                        }
                        else {
                            Console.WriteLine("[Error]: can't vectorize var '{0}', line: {1}", info.GetId(), info.GetLine());
                        }
                    }
                }
                else {
                    var stm = info as Dsl.StatementData;
                    if (null != stm) {
                        func = stm.First.AsFunction;
                        Debug.Assert(null != func);
                        var varInfo = ParseVarInfo(func, stm);
                        if (!string.IsNullOrEmpty(varInfo.Name)) {
                            varInfo = GetVarInfo(varInfo.Name, VarUsage.Find);
                            if (null != varInfo) {
                                varInfo.Type = GetTypeVec(varInfo.Type, out var isTuple, out var isStruct, out var isVecBefore);
                                if (!isVecBefore) {
                                    broadcastVarName = varInfo.Name;
                                    ret = true;

                                    if (null == varInfo.OwnerBlock) {
                                        Console.WriteLine("[Error]: vectorize global var '{0}', please change it to a local var, line: {1}", info.GetId(), info.GetLine());
                                    }
                                }
                            }
                        }
                        else {
                            Console.WriteLine("[Error]: can't vectorize var '{0}', line: {1}", info.GetId(), info.GetLine());
                        }
                    }
                    else {
                        Console.WriteLine("[Error]: can't vectorize var '{0}', line: {1}", info.GetId(), info.GetLine());
                    }
                }
            }
            return ret;
        }

        private static void GenerateScalarFuncCode()
        {
            foreach (var pair in s_FuncInfos) {
                GenerateScalarFuncCode(pair.Value);
            }
        }
        private static void GenerateScalarFuncCode(FuncInfo funcInfo)
        {
            string sig = funcInfo.Signature;
            if (s_AllFuncDsls.TryGetValue(sig, out var syntax)) {
                //reparse: calc const
                if(funcInfo.HasBranches && s_AutoVectorizeBranch) {
                    funcInfo.CodeGenerateEnabled = false;
                }
                else {
                    funcInfo.CodeGenerateEnabled = true;
                }
                funcInfo.ClearForReTransform();
                TransformFunc(syntax);
            }
        }
        private static void TryRemoveFuncBranches(int maxLoop)
        {
            if (s_AutoVectorizeBranch) {
                foreach (var pair in s_FuncInfos) {
                    TryRemoveFuncBranches(maxLoop, pair.Value);
                }
            }
        }
        private static bool TryRemoveFuncBranches(int maxLoop, FuncInfo funcInfo)
        {
            bool r = true;
            if (funcInfo.HasBranches) {
                var blockInfo = funcInfo.ToplevelBlock;
                Debug.Assert(null != blockInfo);

                string sig = funcInfo.Signature;
                if (s_AllFuncDsls.TryGetValue(sig, out var syntax)) {
                    r = TryRemoveBlockBranchesRecursively(maxLoop, blockInfo);
                    if (r) {
                        funcInfo.HasBranches = false;
                        //reparse1: construct new block info
                        if (!s_BranchRemovedFuncs.Contains(sig))
                            s_BranchRemovedFuncs.Add(sig);

                        funcInfo.ClearForReTransform();
                        funcInfo.ClearBlockInfo();
                        TransformFunc(syntax);

                        //reparse2: gen scalar code
                        funcInfo.ClearForReTransform();
                        funcInfo.CodeGenerateEnabled = true;
                        TransformFunc(syntax);
                    }
                    else {
                        funcInfo.HasBranches = true;
                        //reparse1: construct new block info
                        funcInfo.ClearForReTransform();
                        funcInfo.ClearBlockInfo();
                        TransformFunc(syntax);

                        //reparse2: gen scalar code
                        funcInfo.ClearForReTransform();
                        funcInfo.CodeGenerateEnabled = true;
                        TransformFunc(syntax);
                    }
                }
            }
            return r;
        }
        private static bool TryRemoveBlockBranchesRecursively(int maxLoop, BlockInfo blockInfo)
        {
            var syntax = blockInfo.Syntax;
            Debug.Assert(null != syntax);

            bool ret = true;
            foreach (var scb in blockInfo.SubsequentBlocks) {
                ret = TryRemoveBlockBranchesRecursively(maxLoop, scb) && ret;
            }
            foreach (var cb in blockInfo.ChildBlocks) {
                ret = TryRemoveBlockBranchesRecursively(maxLoop, cb) && ret;
            }
            string id = syntax.GetId();
            var tsyntax = syntax;
            bool cret = true;
            if (id == "for") {
                cret = TryUnrollFor(ref tsyntax, blockInfo, maxLoop);
            }
            else if (id == "while") {
                cret = TryUnrollWhile(ref tsyntax, blockInfo, maxLoop);
            }
            else if (id == "do") {
                cret = TryUnrollDoWhile(ref tsyntax, blockInfo, maxLoop);
            }
            if (cret && tsyntax != syntax) {
                var parent = blockInfo.Parent;
                blockInfo.Syntax = tsyntax;
                Debug.Assert(null != parent && blockInfo.FuncSyntaxIndex >= 0);
                var pfunc = parent.Syntax as Dsl.FunctionData;
                var pstm = parent.Syntax as Dsl.StatementData;
                if (null != pstm) {
                    pfunc = pstm.GetFunction(parent.FuncSyntaxIndex).AsFunction;
                    Debug.Assert(null != pfunc);
                }
                if (null != pfunc) {
                    int ix = pfunc.Params.IndexOf(syntax);
                    Debug.Assert(ix >= 0);
                    pfunc.SetParam(ix, tsyntax);
                }
            }
            return ret && cret;
        }
        private static bool TryUnrollFor(ref Dsl.ISyntaxComponent forSyntax, BlockInfo blockInfo, int maxLoop)
        {
            bool canUnroll = false;
            string tmp = string.Empty;
            var forBody = forSyntax as Dsl.FunctionData;
            Debug.Assert(null != forBody && forBody.IsHighOrder);
            var forFunc = forBody.LowerOrderFunction;
            if (forFunc.GetParamNum() == 3) {
                var forInits = forFunc.GetParam(0) as Dsl.FunctionData;
                Debug.Assert(null != forInits);
                var forConds = forFunc.GetParam(1) as Dsl.FunctionData;
                Debug.Assert(null != forConds);
                var forIncs = forFunc.GetParam(2) as Dsl.FunctionData;
                Debug.Assert(null != forIncs);
                if (forInits.GetParamNum() == 1 && forConds.GetParamNum() == 1 && forIncs.GetParamNum() == 1) {
                    var initFunc = forInits.GetParam(0) as Dsl.FunctionData;
                    var condFunc = forConds.GetParam(0) as Dsl.FunctionData;
                    var incFunc = forIncs.GetParam(0) as Dsl.FunctionData;
                    if (null != initFunc && null != condFunc && null != incFunc) {
                        var initLhs = initFunc.GetParam(0);
                        var initLhsFunc = initLhs as Dsl.FunctionData;
                        var initLhsStm = initLhs as Dsl.StatementData;
                        if (null != initLhsStm) {
                            initLhsFunc = initLhsStm.First.AsFunction;
                        }
                        string initVarType = string.Empty;
                        string initVar;
                        if (null != initLhsFunc) {
                            var varInfo = ParseVarInfo(initLhsFunc, initLhsStm);
                            initVar = varInfo.Name;
                            if (!string.IsNullOrEmpty(varInfo.Type)) {
                                initVarType = varInfo.Type;
                            }
                        }
                        else {
                            initVar = initLhs.GetId();
                        }
                        var initVal = initFunc.GetParam(1) as Dsl.ValueData;
                        string condOp = condFunc.GetId();
                        var condVal = condFunc.GetParam(1) as Dsl.ValueData;
                        string incOp = incFunc.GetId();
                        Dsl.ValueData? incVal = null;
                        int incNum = incFunc.GetParamNum();
                        if (incNum == 2) {
                            incVal = incFunc.GetParam(1) as Dsl.ValueData;
                        }
                        if (null != initVal && initVal.GetIdType() == Dsl.ValueData.NUM_TOKEN &&
                            null != condVal && condVal.GetIdType() == Dsl.ValueData.NUM_TOKEN &&
                            (null == incVal || incVal.GetIdType() == Dsl.ValueData.NUM_TOKEN)) {
                            if(string.IsNullOrEmpty(initVarType) && blockInfo.FindVarInfo(initVar, out var bi, out var bbi, out var bbsi, out var bvi)) {
                                Debug.Assert(null != bvi);
                                initVarType = bvi.Type;
                            }
                            int init = 0;
                            float fInit = 0.0f;
                            int cond = 0;
                            float fCond = 0.0f;
                            bool isFloat = initVarType == "float" || initVarType == "double";
                            bool isInt = initVarType == "int" || initVarType == "uint";
                            if (((isInt && int.TryParse(initVal.GetId(), out init)) ||
                                (isFloat && float.TryParse(initVal.GetId(), out fInit))) &&
                                ((isInt && int.TryParse(condVal.GetId(), out cond)) ||
                                (isFloat && float.TryParse(condVal.GetId(), out fCond))) &&
                                (condOp == "<" || condOp == "<=" || condOp == ">" || condOp == ">=") &&
                                (incOp == "++" || incOp == "+=" || incOp == "--" || incOp == "-=")) {
                                int inc = 0;
                                float fInc = 0.0f;
                                if (condOp == "<=" || condOp == ">=") {
                                    if (isInt) {
                                        if (incOp == "+=" || incOp == "++")
                                            ++cond;
                                        else
                                            --cond;
                                    }
                                    else if (isFloat) {
                                        if (incOp == "+=" || incOp == "++")
                                            ++fCond;
                                        else
                                            --fCond;
                                    }
                                }
                                if (null != incVal) {
                                    if (isInt) {
                                        int.TryParse(incVal.GetId(), out inc);
                                        if (incOp == "-=") {
                                            inc = -inc;
                                        }
                                    }
                                    else if (isFloat) {
                                        float.TryParse(incVal.GetId(), out fInc);
                                        if (incOp == "-=") {
                                            fInc = -fInc;
                                        }
                                    }
                                }
                                else if (isInt) {
                                    inc = incOp == "++" ? 1 : -1;
                                }
                                else if (isFloat) {
                                    fInc = incOp == "++" ? 1.0f : -1.0f;
                                }
                                if ((isInt && inc != 0) || (isFloat && Math.Abs(fInc) > float.Epsilon)) {
                                    int loopCount = isInt ? (cond - init) / inc : (isFloat ? (int)((fCond - fInit) / fInc) : 0);
                                    if (loopCount <= 0 || loopCount > 512) {
                                        Console.WriteLine("for loop {0} must be in (0, 512] !!! line: {1}", loopCount, forFunc.GetLine());
                                    }
                                    else {
                                        bool find = SyntaxSearcher.Search(forBody, (syntax, six, syntaxStack) => VarAssignmentPred(syntax, six, syntaxStack, initVar));
                                        if (!find) {
                                            canUnroll = true;
                                            int stmCt = forBody.GetParamNum();
                                            for (int i = 1; i < loopCount; ++i) {
                                                var assignStm = new Dsl.FunctionData();
                                                assignStm.Name = new Dsl.ValueData("=", Dsl.ValueData.ID_TOKEN);
                                                assignStm.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR);
                                                assignStm.AddParam(initVar);
                                                if (isInt) {
                                                    string val = (init + inc * i).ToString();
                                                    assignStm.AddParam(val, Dsl.ValueData.NUM_TOKEN);
                                                }
                                                else if (isFloat) {
                                                    string fVal = (fInit + fInc * i).ToString();
                                                    if (fVal.IndexOf('.') < 0)
                                                        fVal = fVal + ".0";
                                                    assignStm.AddParam(fVal, Dsl.ValueData.NUM_TOKEN);
                                                }
                                                else {
                                                    Debug.Assert(false);
                                                }
                                                assignStm.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                                                forBody.AddParam(assignStm);
                                                for (int ii = 0; ii < stmCt; ++ii) {
                                                    var stm = forBody.GetParam(ii);
                                                    forBody.AddParam(Dsl.Utility.CloneDsl(stm));
                                                }
                                            }
                                            initFunc.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                                            forBody.Name = new Dsl.ValueData("block");
                                            forBody.Params.Insert(0, initFunc);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!canUnroll) {
                    int loopCt = maxLoop;
                    var attrsFunc = blockInfo.Attribute as Dsl.FunctionData;
                    if (null != attrsFunc) {
                        foreach (var p in attrsFunc.Params) {
                            var attr = p as Dsl.FunctionData;
                            if (null != attr && attr.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                                var attrfd = attr.GetParam(0) as Dsl.FunctionData;
                                if (null != attrfd && attrfd.GetId() == "unroll") {
                                    if (int.TryParse(attrfd.GetParamId(0), out var ct) && ct >= 0) {
                                        loopCt = ct;
                                    }
                                }
                            }
                        }
                    }
                    if (loopCt > 0) {
                        canUnroll = true;
                        var newLoop = new Dsl.FunctionData();
                        newLoop.Name = new Dsl.ValueData("block");
                        newLoop.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_STATEMENT);
                        Dsl.ISyntaxComponent? last = null;
                        foreach (var p in forInits.Params) {
                            newLoop.AddParam(p);
                            last = p;
                        }
                        if (null != last) {
                            last.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                        }
                        var ifFunc = new Dsl.FunctionData();
                        ifFunc.Name = new Dsl.ValueData("if");
                        ifFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);
                        if (forConds.GetParamNum() == 1) {
                            ifFunc.AddParam(forConds.GetParam(0));
                        }
                        else {
                            ifFunc.AddParam(new Dsl.ValueData("true", Dsl.ValueData.ID_TOKEN));
                        }
                        forBody.LowerOrderFunction = ifFunc;
                        newLoop.AddParam(forBody);
                        for (int i = 1; i < loopCt; ++i) {
                            foreach (var p in forIncs.Params) {
                                newLoop.AddParam(p);
                                p.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                            }
                            var addBody = Dsl.Utility.CloneDsl(forBody);
                            newLoop.AddParam(addBody);
                        }
                        forSyntax = newLoop;
                    }
                }
            }
            if (!canUnroll) {
                Console.WriteLine("[Info]: Cant unroll statement '{0}', line: {1}", forSyntax.GetId(), forSyntax.GetLine());
            }
            return canUnroll;
        }
        private static bool TryUnrollWhile(ref Dsl.ISyntaxComponent whileSyntax, BlockInfo blockInfo, int maxLoop)
        {
            bool canUnroll = false;
            string tmp = string.Empty;
            var whileBody = whileSyntax as Dsl.FunctionData;
            Debug.Assert(null != whileBody && whileBody.IsHighOrder);
            var whileFunc = whileBody.LowerOrderFunction;
            if (whileFunc.GetParamNum() == 1) {
                var condFunc = whileFunc.GetParam(0) as Dsl.FunctionData;
                if (null != condFunc) {
                    string condOp = condFunc.GetId();
                    var condVar = condFunc.GetParam(0) as Dsl.ValueData;
                    var condVal = condFunc.GetParam(1) as Dsl.ValueData;
                    if (null != condVar && null != condVal && condVal.GetIdType() == Dsl.ValueData.NUM_TOKEN) {
                        string vname = condVar.GetId();
                        string vtype = string.Empty;
                        if (blockInfo.FindVarInfo(vname, out var bi, out var bbi, out var bbsi, out var bvi)) {
                            Debug.Assert(null != bvi);
                            vtype = bvi.Type;
                        }
                        int cond = 0;
                        float fCond = 0.0f;
                        bool isFloat = vtype == "float" || vtype == "double";
                        bool isInt = vtype == "int" || vtype == "uint";
                        if (((isInt && int.TryParse(condVal.GetId(), out cond)) || 
                            (isFloat && float.TryParse(condVal.GetId(), out fCond))) &&
                            (condOp == "<" || condOp == "<=" || condOp == ">" || condOp == ">=")) {
                            string incOp = string.Empty;
                            Dsl.ValueData? incVal = null;
                            var parent = blockInfo.Parent;
                            Debug.Assert(null != parent);
                            int init = 0;
                            float fInit = 0.0f;
                            int ix = parent.FindChildIndex(blockInfo, out var six);
                            if (parent.TryGetVarConstInBasicBlock(ix, -1, vname, out var vval) && !string.IsNullOrEmpty(vval) && 
                                ((isInt && int.TryParse(vval, out init)) || 
                                (isFloat && float.TryParse(vval, out fInit)))) {
                                int assignCt = 0;
                                Dsl.ISyntaxComponent? assignExp = null;
                                SyntaxSearcher.Search(whileBody, (syntax, six, syntaxStack) => VarAssignmentPredAndGetAssignExp(syntax, six, syntaxStack, vname, ref assignCt, out assignExp));
                                if (assignCt == 1 && null != assignExp) {
                                    var incFunc = assignExp as Dsl.FunctionData;
                                    Debug.Assert(null != incFunc);

                                    incOp = incFunc.GetId();
                                    int incNum = incFunc.GetParamNum();
                                    if (incNum == 2) {
                                        incVal = incFunc.GetParam(1) as Dsl.ValueData;
                                    }
                                }
                                int inc = 0;
                                float fInc = 0.0f;
                                if (condOp == "<=" || condOp == ">=") {
                                    if (isInt) {
                                        if (incOp == "+=" || incOp == "++")
                                            ++cond;
                                        else
                                            --cond;
                                    }
                                    else if (isFloat) {
                                        if (incOp == "+=" || incOp == "++")
                                            ++fCond;
                                        else
                                            --fCond;
                                    }
                                }
                                if ((null == incVal || incVal.GetIdType() == Dsl.ValueData.NUM_TOKEN) &&
                                    (incOp == "++" || incOp == "+=" || incOp == "--" || incOp == "-=")) {
                                    if (null != incVal) {
                                        if (isInt) {
                                            int.TryParse(incVal.GetId(), out inc);
                                            if (incOp == "-=") {
                                                inc = -inc;
                                            }
                                        }
                                        else if (isFloat) {
                                            float.TryParse(incVal.GetId(), out fInc);
                                            if (incOp == "-=") {
                                                fInc = -fInc;
                                            }
                                        }
                                    }
                                    else if (isInt) {
                                        inc = incOp == "++" ? 1 : -1;
                                    }
                                    else if (isFloat) {
                                        fInc = incOp == "++" ? 1.0f : -1.0f;
                                    }
                                }
                                if ((isInt && inc != 0) || (isFloat && Math.Abs(fInc) > float.Epsilon)) {
                                    int loopCount = isInt ? (cond - init) / inc : (isFloat ? (int)((fCond - fInit) / fInc) : 0);
                                    if (loopCount <= 0 || loopCount > 512) {
                                        Console.WriteLine("while loop {0} must be in (0, 512] !!! line: {1}", loopCount, whileFunc.GetLine());
                                    }
                                    else {
                                        canUnroll = true;
                                        int stmCt = whileBody.GetParamNum();
                                        for (int i = 1; i < loopCount; ++i) {
                                            for (int ii = 0; ii < stmCt; ++ii) {
                                                var stm = whileBody.GetParam(ii);
                                                whileBody.AddParam(Dsl.Utility.CloneDsl(stm));
                                            }
                                        }
                                        whileBody.Name = new Dsl.ValueData("block");
                                    }
                                }
                            }
                        }
                    }
                }
                if (!canUnroll) {
                    int loopCt = maxLoop;
                    var attrsFunc = blockInfo.Attribute as Dsl.FunctionData;
                    if (null != attrsFunc) {
                        foreach (var p in attrsFunc.Params) {
                            var attr = p as Dsl.FunctionData;
                            if (null != attr && attr.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                                var attrfd = attr.GetParam(0) as Dsl.FunctionData;
                                if (null != attrfd && attrfd.GetId() == "unroll") {
                                    if (int.TryParse(attrfd.GetParamId(0), out var ct) && ct >= 0) {
                                        loopCt = ct;
                                    }
                                }
                            }
                        }
                    }
                    if (loopCt > 0) {
                        canUnroll = true;
                        var newLoop = new Dsl.FunctionData();
                        newLoop.Name = new Dsl.ValueData("block");
                        newLoop.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_STATEMENT);
                        whileFunc.Name.SetId("if");
                        newLoop.AddParam(whileBody);
                        for (int i = 1; i < loopCt; ++i) {
                            var addBody = Dsl.Utility.CloneDsl(whileBody);
                            newLoop.AddParam(addBody);
                        }
                        whileSyntax = newLoop;
                    }
                }
            }
            if (!canUnroll) {
                Console.WriteLine("[Info]: Cant unroll statement '{0}', line: {1}", whileSyntax.GetId(), whileSyntax.GetLine());
            }
            return canUnroll;
        }
        private static bool TryUnrollDoWhile(ref Dsl.ISyntaxComponent dowhileSyntax, BlockInfo blockInfo, int maxLoop)
        {
            bool canUnroll = false;
            string tmp = string.Empty;
            var loopStm = dowhileSyntax as Dsl.StatementData;
            Debug.Assert(null != loopStm);
            var doBody = loopStm.First.AsFunction;
            var whileFunc = loopStm.Second.AsFunction;
            Debug.Assert(null != doBody && null != whileFunc);
            if (whileFunc.GetParamNum() == 1) {
                var condFunc = whileFunc.GetParam(0) as Dsl.FunctionData;
                if (null != condFunc) {
                    string condOp = condFunc.GetId();
                    var condVar = condFunc.GetParam(0) as Dsl.ValueData;
                    var condVal = condFunc.GetParam(1) as Dsl.ValueData;
                    if (null != condVar && null != condVal && condVal.GetIdType() == Dsl.ValueData.NUM_TOKEN) {
                        string vname = condVar.GetId();
                        string vtype = string.Empty;
                        if (blockInfo.FindVarInfo(vname, out var bi, out var bbi, out var bbsi, out var bvi)) {
                            Debug.Assert(null != bvi);
                            vtype = bvi.Type;
                        }
                        int cond = 0;
                        float fCond = 0.0f;
                        bool isFloat = vtype == "float" || vtype == "double";
                        bool isInt = vtype == "int" || vtype == "uint";
                        if (((isInt && int.TryParse(condVal.GetId(), out cond)) ||
                            (isFloat && float.TryParse(condVal.GetId(), out fCond))) &&
                            (condOp == "<" || condOp == "<=" || condOp == ">" || condOp == ">=")) {
                            string incOp = string.Empty;
                            Dsl.ValueData? incVal = null;
                            var parent = blockInfo.Parent;
                            Debug.Assert(null != parent);
                            int init = 0;
                            float fInit = 0.0f;
                            int ix = parent.FindChildIndex(blockInfo, out var six);
                            if (parent.TryGetVarConstInBasicBlock(ix, -1, vname, out var vval) && !string.IsNullOrEmpty(vval) && 
                                ((isInt && int.TryParse(vval, out init)) || 
                                (isFloat && float.TryParse(vval, out fInit)))) {
                                int assignCt = 0;
                                Dsl.ISyntaxComponent? assignExp = null;
                                SyntaxSearcher.Search(doBody, (syntax, six, syntaxStack) => VarAssignmentPredAndGetAssignExp(syntax, six, syntaxStack, vname, ref assignCt, out assignExp));
                                if (assignCt == 1 && null != assignExp) {
                                    var incFunc = assignExp as Dsl.FunctionData;
                                    Debug.Assert(null != incFunc);

                                    incOp = incFunc.GetId();
                                    int incNum = incFunc.GetParamNum();
                                    if (incNum == 2) {
                                        incVal = incFunc.GetParam(1) as Dsl.ValueData;
                                    }
                                }
                                int inc = 0;
                                float fInc = 0.0f;
                                if (condOp == "<=" || condOp == ">=") {
                                    if (isInt) {
                                        if (incOp == "+=" || incOp == "++")
                                            ++cond;
                                        else
                                            --cond;
                                    }
                                    else if(isFloat) {
                                        if (incOp == "+=" || incOp == "++")
                                            ++fCond;
                                        else
                                            --fCond;
                                    }
                                }
                                if ((null == incVal || incVal.GetIdType() == Dsl.ValueData.NUM_TOKEN) &&
                                    (incOp == "++" || incOp == "+=" || incOp == "--" || incOp == "-=")) {
                                    if (null != incVal) {
                                        if (isInt) {
                                            int.TryParse(incVal.GetId(), out inc);
                                            if (incOp == "-=") {
                                                inc = -inc;
                                            }
                                        }
                                        else if (isFloat) {
                                            float.TryParse(incVal.GetId(), out fInc);
                                            if (incOp == "-=") {
                                                fInc = -fInc;
                                            }
                                        }
                                    }
                                    else if (isInt) {
                                        inc = incOp == "++" ? 1 : -1;
                                    }
                                    else if (isFloat) {
                                        fInc = incOp == "++" ? 1.0f : -1.0f;
                                    }
                                }
                                if ((isInt && inc != 0) || (isFloat && Math.Abs(fInc) > float.Epsilon)) {
                                    int loopCount = isInt ? (cond - init) / inc : (isFloat ? (int)((fCond - fInit) / fInc) : 0);
                                    if (loopCount <= 0 || loopCount > 512) {
                                        Console.WriteLine("while loop {0} must be in (0, 512] !!! line: {1}", loopCount, whileFunc.GetLine());
                                    }
                                    else {
                                        canUnroll = true;
                                        int stmCt = doBody.GetParamNum();
                                        for (int i = 1; i < loopCount; ++i) {
                                            for (int ii = 0; ii < stmCt; ++ii) {
                                                var stm = doBody.GetParam(ii);
                                                doBody.AddParam(Dsl.Utility.CloneDsl(stm));
                                            }
                                        }
                                        doBody.Name = new Dsl.ValueData("block");
                                        doBody.SetSeparator(Dsl.AbstractSyntaxComponent.SEPARATOR_SEMICOLON);
                                        dowhileSyntax = doBody;
                                    }
                                }
                            }
                        }
                    }
                }
                if (!canUnroll) {
                    int loopCt = maxLoop;
                    var attrsFunc = blockInfo.Attribute as Dsl.FunctionData;
                    if (null != attrsFunc) {
                        foreach (var p in attrsFunc.Params) {
                            var attr = p as Dsl.FunctionData;
                            if (null != attr && attr.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                                var attrfd = attr.GetParam(0) as Dsl.FunctionData;
                                if (null != attrfd && attrfd.GetId() == "unroll") {
                                    if (int.TryParse(attrfd.GetParamId(0), out var ct) && ct >= 0) {
                                        loopCt = ct;
                                    }
                                }
                            }
                        }
                    }
                    if (loopCt > 0) {
                        canUnroll = true;
                        var newLoop = new Dsl.FunctionData();
                        newLoop.Name = new Dsl.ValueData("block");
                        newLoop.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_STATEMENT);
                        whileFunc.Name.SetId("if");
                        var ifFunc = new Dsl.FunctionData();
                        ifFunc.Name = new Dsl.ValueData("if");
                        ifFunc.SetParamClass((int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_PARENTHESIS);
                        ifFunc.AddParam(new Dsl.ValueData("true", Dsl.ValueData.ID_TOKEN));
                        var tmplBody = Dsl.Utility.CloneDsl(doBody) as Dsl.FunctionData;
                        Debug.Assert(null != tmplBody);
                        tmplBody.LowerOrderFunction = whileFunc;
                        doBody.LowerOrderFunction = ifFunc;
                        newLoop.AddParam(doBody);
                        for (int i = 1; i < loopCt; ++i) {
                            if (i == 1)
                                newLoop.AddParam(tmplBody);
                            else {
                                var addBody = Dsl.Utility.CloneDsl(tmplBody);
                                newLoop.AddParam(addBody);
                            }
                        }
                        dowhileSyntax = newLoop;
                    }
                }
            }
            if (!canUnroll) {
                Console.WriteLine("[Info]: Cant unroll statement '{0}', line: {1}", dowhileSyntax.GetId(), dowhileSyntax.GetLine());
            }
            return canUnroll;
        }
        private static bool VarAssignmentPred(Dsl.ISyntaxComponent syntax, int index, IEnumerable<SyntaxStackInfo> syntaxStack, string varName)
        {
            bool ret = false;
            if (IsVarAssignment(syntax, index, syntaxStack, varName, out var assignExp)) {
                ret = true;
                foreach (var p in syntaxStack) {
                    if (p.Syntax.GetId() == "for" && p.Index == -1) {
                        ret = false;
                        break;
                    }
                }
            }
            return ret;
        }
        private static bool VarAssignmentPredAndGetAssignExp(Dsl.ISyntaxComponent syntax, int index, IEnumerable<SyntaxStackInfo> syntaxStack, string varName, ref int ct, out Dsl.ISyntaxComponent? assignExp)
        {
            bool ret = false;
            if (IsVarAssignment(syntax, index, syntaxStack, varName, out assignExp)) {
                ++ct;
                ret = true;
            }
            return ret;
        }
        private static bool IsVarAssignment(Dsl.ISyntaxComponent syntax, int index, IEnumerable<SyntaxStackInfo> syntaxStack, string varName, out Dsl.ISyntaxComponent? assignExp)
        {
            bool ret = false;
            assignExp = null;
            if (syntax.GetId() == varName) {
                var ps = GetOuterSyntax(syntaxStack);
                if (null != ps) {
                    string pid = ps.GetId();
                    bool isLHS = false;
                    var assignFunc = ps as Dsl.FunctionData;
                    if (null != assignFunc) {
                        string leftId = assignFunc.GetParamId(0);
                        isLHS = leftId == varName;
                    }
                    if (isLHS && (pid == "++" || pid == "--" || pid == "=" || (pid[pid.Length - 1] == '=' && pid != ">=" && pid != "<=" && pid != "==" && pid != "!=" && pid != ">>=" && pid != "<<="))) {
                        assignExp = ps;
                        ret = true;
                    }
                }
            }
            return ret;
        }
        
        internal class VectorialFuncInfo
        {
            internal string FuncSignature = string.Empty;
            internal List<string> VecArgTypes = new List<string>();
            internal string VecRetType = string.Empty;
            internal FuncInfo? VecFuncInfo = null;
            internal int VectorizeNo = 0;

            internal void ModifyFuncInfo()
            {
                if (null != VecFuncInfo) {
                    for (int ix = 0; ix < VecFuncInfo.Params.Count; ++ix) {
                        var p = VecFuncInfo.Params[ix];
                        if (ix < VecArgTypes.Count) {
                            p.Type = VecArgTypes[ix];
                        }
                    }
                    if (!VecFuncInfo.IsVoid()) {
                        Debug.Assert(null != VecFuncInfo.RetInfo);
                        VecFuncInfo.RetInfo.Type = VecRetType;
                    }
                    VecFuncInfo.VectorizeNo = VectorizeNo;
                }
            }
        }
        internal class VecFuncCodeInfo
        {
            internal FuncInfo VecFuncInfo;
            internal StringBuilder VecFuncStringBuilder;

            internal VecFuncCodeInfo(FuncInfo funcInfo, StringBuilder sb)
            {
                VecFuncInfo = funcInfo;
                VecFuncStringBuilder = sb;
            }
        }

        private static HashSet<string> s_CalledScalarFuncs = new HashSet<string>();
        private static Queue<VectorialFuncInfo> s_CalledVecFuncQueue = new Queue<VectorialFuncInfo>();
        private static Stack<VecFuncCodeInfo> s_VecFuncCodeStack = new Stack<VecFuncCodeInfo>();
    }
}