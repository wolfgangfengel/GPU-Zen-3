using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Hlsl2Python
{
    internal partial class Program
    {
        internal static int GenUniqueNumber()
        {
            return ++s_UniqueNumber;
        }
        internal static string GetIndentString(int indent)
        {
            const string c_IndentString = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
            return c_IndentString.Substring(0, indent);
        }
        private static KeyValuePair<string, string> ParseTypeDef(Dsl.FunctionData typeDefFunc, out bool isConst)
        {
            var typeInfo = typeDefFunc.GetParam(0);
            var typeInfoVal = typeInfo as Dsl.ValueData;
            var typeInfoFunc = typeInfo as Dsl.FunctionData;
            var typeInfoStm = typeInfo as Dsl.StatementData;
            var typeNameInfo = typeDefFunc.GetParam(1);
            var typeNameFunc = typeNameInfo as Dsl.FunctionData;
            string newType = typeNameInfo.GetId();

            isConst = false;
            string oriType = string.Empty;
            if (null != typeInfoVal) {
                oriType = typeInfoVal.GetId();
            }
            else if (null != typeInfoFunc) {
                var pf = typeInfoFunc;
                if (null != pf && pf.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_ANGLE_BRACKET_COLON) {
                    oriType = BuildTypeWithTypeArgs(pf);
                }
            }
            else if (null != typeInfoStm) {
                foreach (var p in typeInfoStm.Functions) {
                    var pv = p as Dsl.ValueData;
                    if (null != pv) {
                        string key = pv.GetId();
                        if (key == "const")
                            isConst = true;
                        else
                            oriType = key;
                    }
                    else {
                        var pf = p as Dsl.FunctionData;
                        if (null != pf && pf.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_ANGLE_BRACKET_COLON) {
                            oriType = BuildTypeWithTypeArgs(pf);
                        }
                    }
                }
            }
            if (null != typeNameFunc) {
                List<string> arrTags = new List<string>();
                BuildTypeWithArrTags(typeNameFunc, arrTags);
                if(arrTags.Count> 0) {
                    var sb = NewStringBuilder();
                    sb.Append(oriType);
                    for(int ix = arrTags.Count - 1; ix >= 0; --ix) {
                        sb.Append(arrTags[ix]);
                    }
                    oriType = sb.ToString();
                    RecycleStringBuilder(sb);
                }
            }
            return new KeyValuePair<string, string>(newType, oriType);
        }
        private static bool AddTypeDef(string newType, string oriType)
        {
            bool isGlobal = false;
            var funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                var infos = funcInfo.LocalTypeDefs;
                infos[newType] = oriType;
            }
            else {
                var infos = s_GlobalTypeDefs;
                infos[newType] = oriType;
                isGlobal = true;
            }
            return isGlobal;
        }
        private static VarInfo ParseVarInfo(Dsl.FunctionData varFunc, Dsl.StatementData? varStm)
        {
            var varInfo = new VarInfo();
            string funcId = varFunc.GetId();
            if (funcId == "var" || funcId == "field" || funcId == "func") {
                var specFunc = varFunc.GetParam(0) as Dsl.FunctionData;
                Debug.Assert(null != specFunc);
                var typeInfo = varFunc.GetParam(1);
                var typeInfoVal = typeInfo as Dsl.ValueData;
                var typeInfoFunc = typeInfo as Dsl.FunctionData;
                var typeInfoStm = typeInfo as Dsl.StatementData;
                var nameInfo = varFunc.GetParam(2);
                var nameInfoFunc = nameInfo as Dsl.FunctionData;
                string varName = varFunc.GetParamId(2);
                string arrTag = string.Empty;
                if (null != nameInfoFunc) {
                    arrTag = BuildTypeWithTypeArgs(nameInfoFunc).Substring(varName.Length);
                }

                varInfo.Name = varName;
                if (null != typeInfoVal) {
                    varInfo.IsConst = false;
                    varInfo.Type = typeInfoVal.GetId() + arrTag;
                }
                else if (null != typeInfoFunc) {
                    var pf = typeInfoFunc;
                    if (null != pf && pf.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_ANGLE_BRACKET_COLON) {
                        varInfo.Type = BuildTypeWithTypeArgs(pf) + arrTag;
                    }
                }
                else if (null != typeInfoStm) {
                    foreach (var p in typeInfoStm.Functions) {
                        var pv = p as Dsl.ValueData;
                        if (null != pv) {
                            string key = pv.GetId();
                            if (key == "const")
                                varInfo.IsConst = true;
                            else
                                varInfo.Type = key + arrTag;
                        }
                        else {
                            var pf = p as Dsl.FunctionData;
                            if (null != pf && pf.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_ANGLE_BRACKET_COLON) {
                                varInfo.Type = BuildTypeWithTypeArgs(pf) + arrTag;
                            }
                        }
                    }
                }
                foreach (var p in specFunc.Params) {
                    var key = p.GetId();
                    varInfo.Modifiers.Add(key);

                    if (key == "inout")
                        varInfo.IsInOut = true;
                    else if (key == "out")
                        varInfo.IsOut = true;
                }
                if (null != varStm) {
                    for (int funcIx = 1; funcIx < varStm.GetFunctionNum(); ++funcIx) {
                        var func = varStm.GetFunction(funcIx);
                        string id = func.GetId();
                        if (id == "semantic") {
                            var adlFunc = func.AsFunction;
                            Debug.Assert(null != adlFunc);
                            if (adlFunc.IsHighOrder)
                                varInfo.Semantic = adlFunc.LowerOrderFunction.GetParamId(0);
                            else
                                varInfo.Semantic = adlFunc.GetParamId(0);
                        }
                        else if (id == "register") {
                            var adlFunc = func.AsFunction;
                            Debug.Assert(null != adlFunc);
                            varInfo.Register = adlFunc.GetParamId(0);
                        }
                    }
                }
                if(s_GlobalTypeDefs.TryGetValue(varInfo.Type, out var otype)) {
                    varInfo.Type = otype;
                }
                varInfo.OriType = varInfo.Type;
            }
            return varInfo;
        }
        private static string BuildTypeWithTypeArgs(Dsl.FunctionData func)
        {
            var sb = new StringBuilder();
            if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                var arrTags = new List<string>();
                string baseType = BuildTypeWithArrTags(func, arrTags);
                sb.Append(baseType);
                for (int ix = arrTags.Count - 1; ix >= 0; --ix) {
                    sb.Append(arrTags[ix]);
                }
            }
            else {
                if (func.IsHighOrder) {
                    sb.Append(BuildTypeWithTypeArgs(func.LowerOrderFunction));
                }
                else {
                    sb.Append(func.GetId());
                }
                foreach (var p in func.Params) {
                    sb.Append('|');
                    sb.Append(DslToNameString(p));
                }
            }
            return sb.ToString();
        }
        private static string BuildTypeWithArrTags(Dsl.FunctionData func, List<string> arrTags)
        {
            string ret = string.Empty;
            if (func.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_BRACKET) {
                if (func.IsHighOrder) {
                    ret = BuildTypeWithArrTags(func.LowerOrderFunction, arrTags);
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
                ret = BuildTypeWithTypeArgs(func);
            }
            return ret;
        }
        private static string DslToNameString(Dsl.ISyntaxComponent syntax)
        {
            var valData = syntax as Dsl.ValueData;
            if (null != valData)
                return valData.GetId();
            else {
                var funcData = syntax as Dsl.FunctionData;
                if (null != funcData) {
                    var sb = new StringBuilder();
                    if (funcData.IsHighOrder) {
                        sb.Append(DslToNameString(funcData.LowerOrderFunction));
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
                        sb.Append(DslToNameString(p));
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
                            sb.Append(DslToNameString(func));
                        }
                        return sb.ToString();
                    }
                    else {
                        return string.Empty;
                    }
                }
            }
        }
        private static FuncInfo? CurFuncInfo()
        {
            FuncInfo? curFuncInfo = null;
            if (s_LexicalScopeStack.Count > 0) {
                var curBlockInfo = s_LexicalScopeStack.Peek();
                curFuncInfo = curBlockInfo.OwnerFunc;
            }
            return curFuncInfo;
        }
        private static bool CurFuncBlockInfoConstructed()
        {
            bool ret = false;
            var funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                ret = funcInfo.BlockInfoConstructed;
            }
            return ret;
        }
        private static bool CurFuncCodeGenerateEnabled()
        {
            bool ret = true;
            var funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                ret = funcInfo.CodeGenerateEnabled;
            }
            return ret;
        }
        private static void ClearParamInfo(FuncInfo? funcInfo)
        {
            if (null == funcInfo)
                funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                funcInfo.Params.Clear();
                funcInfo.HasInOutOrOutParams = false;
                funcInfo.InOutOrOutParams.Clear();
            }
        }
        private static void AddParamInfo(VarInfo varInfo, FuncInfo? funcInfo)
        {
            if (null == funcInfo)
                funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                funcInfo.Params.Add(varInfo);
                if (varInfo.IsInOut || varInfo.IsOut) {
                    funcInfo.HasInOutOrOutParams = true;
                    funcInfo.InOutOrOutParams.Add(varInfo);
                }

                var curBlockInfo = funcInfo.ToplevelBlock;
                Debug.Assert(null != curBlockInfo);
                varInfo.OwnerBlock = curBlockInfo;
                varInfo.OwnerBasicBlockIndex = curBlockInfo.CurBasicBlockIndex;
                varInfo.OwnerStatementIndex = curBlockInfo.CurBasicBlock().CurStatementIndex;
            }
        }
        private static void SetRetInfo(VarInfo varInfo, FuncInfo? funcInfo)
        {
            if (null == funcInfo)
                funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                funcInfo.RetInfo = varInfo;
            }
        }
        private static bool AddVar(VarInfo varInfo)
        {
            bool isGlobal = false;
            var funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                if(funcInfo.LocalTypeDefs.TryGetValue(varInfo.Type, out var otype)) {
                    varInfo.Type = otype;
                    varInfo.OriType = otype;
                }
                var infos = funcInfo.LocalVarInfos;
                if (!infos.TryGetValue(varInfo.Name, out var varInfos)) {
                    varInfos = new Dictionary<int, VarInfo>();
                    infos.Add(varInfo.Name, varInfos);
                }
                varInfos[CurBlockId()] = varInfo;

                var curBlockInfo = CurBlockInfo();
                Debug.Assert(null != curBlockInfo);
                varInfo.OwnerBlock = curBlockInfo;
                varInfo.OwnerBasicBlockIndex = curBlockInfo.CurBasicBlockIndex;
                varInfo.OwnerStatementIndex = curBlockInfo.CurBasicBlock().CurStatementIndex;
                if (curBlockInfo.CurBasicBlockStatementDeclVars.TryGetValue(varInfo.Name, out var dvInfo)) {
                    dvInfo.Type = varInfo.Type;
                }
                else {
                    curBlockInfo.CurBasicBlockStatementDeclVars.Add(varInfo.Name, new DeclVarInfo { Name = varInfo.Name, Type = varInfo.Type });
                }
                if (curBlockInfo.CurBasicBlockStatementSetVars.TryGetValue(varInfo.Name, out var bvInfo)) {
                    bvInfo.Type = varInfo.Type;
                    bvInfo.OwnerBlock = varInfo.OwnerBlock;
                    bvInfo.OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex;
                    bvInfo.OwnerStatementIndex = varInfo.OwnerStatementIndex;
                }
                else {
                    curBlockInfo.CurBasicBlockStatementSetVars.Add(varInfo.Name, new BlockVarInfo { Name = varInfo.Name, Type = varInfo.Type, OwnerBlock = varInfo.OwnerBlock, OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex, OwnerStatementIndex = varInfo.OwnerStatementIndex });
                }
            }
            else {
                if (!s_GlobalVarInfos.TryGetValue(varInfo.Name, out var varInfos)) {
                    varInfos = new Dictionary<int, VarInfo>();
                    s_GlobalVarInfos.Add(varInfo.Name, varInfos);
                }
                varInfos[CurBlockId()] = varInfo;
                isGlobal = true;
            }
            return isGlobal;
        }
        private static BlockInfo? GetOrNewBlockInfo(Dsl.ISyntaxComponent syntax, int ix)
        {
            BlockInfo? blockInfo = null;
            if (CurFuncBlockInfoConstructed()) {
                if (s_LexicalScopeStack.Count > 0) {
                    var curBlockInfo = s_LexicalScopeStack.Peek();
                    blockInfo = curBlockInfo.FindChild(syntax, ix);
                }
            }
            else {
                blockInfo = new BlockInfo();
                blockInfo.Syntax = syntax;
                blockInfo.FuncSyntaxIndex = ix;
            }
            var curFunc = CurFuncInfo();
            if (null != curFunc && null != blockInfo) {
                blockInfo.Attribute = curFunc.LastAttribute;
            }
            return blockInfo;
        }
        private static int CurBlockId()
        {
            if (s_LexicalScopeStack.Count > 0) {
                var curBlockInfo = s_LexicalScopeStack.Peek();
                return curBlockInfo.BlockId;
            }
            return 0;
        }
        private static BlockInfo? CurBlockInfo()
        {
            BlockInfo? ret = null;
            if (s_LexicalScopeStack.Count > 0) {
                ret = s_LexicalScopeStack.Peek();
            }
            return ret;
        }
        private static void PushBlock(BlockInfo blockInfo)
        {
            blockInfo.CurBasicBlockIndex = 0;
            blockInfo.CurBasicBlock().CurStatementIndex = -1;
            if (!CurFuncBlockInfoConstructed()) {
                ++s_LastBlockId;
                blockInfo.BlockId = s_LastBlockId;
                if (s_LexicalScopeStack.Count > 0) {
                    var parent = s_LexicalScopeStack.Peek();
                    blockInfo.Parent = parent;
                    blockInfo.OwnerFunc = parent.OwnerFunc;
                    parent.AddChild(blockInfo);
                }
                else {
                    blockInfo.Parent = null;
                }
            }
            s_LexicalScopeStack.Push(blockInfo);
        }
        private static void PopBlock()
        {
            PopBlock(false);
        }
        private static void PopBlock(bool keepBasicBlock)
        {
            var blockInfo = s_LexicalScopeStack.Pop();
            if (s_LexicalScopeStack.Count > 0) {
                var parent = s_LexicalScopeStack.Peek();
                if (CurFuncBlockInfoConstructed() && !keepBasicBlock) {
                    int ix = parent.FindChildIndex(blockInfo, out var six);
                    if (ix >= 0) {
                        if (six >= 0) {
                            var firstBlock = parent.ChildBlocks[ix];
                            if (six == firstBlock.SubsequentBlocks.Count - 1) {
                                parent.CurBasicBlockIndex = ix + 1;
                                parent.CurBasicBlock().CurStatementIndex = -1;
                            }
                        }
                        else if (blockInfo.SubsequentBlocks.Count == 0) {
                            parent.CurBasicBlockIndex = ix + 1;
                            parent.CurBasicBlock().CurStatementIndex = -1;
                        }
                    }
                }
                else if(!keepBasicBlock) {
                    parent.CurBasicBlockIndex = parent.BasicBlocks.Count - 1;
                    parent.CurBasicBlock().CurStatementIndex = -1;
                }
            }
        }
        private static VarInfo? GetVarInfo(string name, VarUsage usage)
        {
            VarInfo? varInfo = null;
            bool hasLocalVar = false;
            var funcInfo = CurFuncInfo();
            if (null != funcInfo) {
                if (funcInfo.LocalVarInfos.TryGetValue(name, out var varInfos)) {
                    hasLocalVar = true;
                    foreach (var blockInfo in s_LexicalScopeStack) {
                        int blockId = blockInfo.BlockId;
                        if (varInfos.TryGetValue(blockId, out varInfo))
                            break;
                    }
                }
                if (null == varInfo) {
                    foreach (var p in funcInfo.Params) {
                        if (p.Name == name) {
                            varInfo = p;
                            if (hasLocalVar) {
                                Console.WriteLine("[error]: the param '{0}' of function '{1}' conflicts with local variable. please rename it !", name, funcInfo.Name);
                            }
                            break;
                        }
                    }
                }
            }
            if (null == varInfo) {
                if (s_GlobalVarInfos.TryGetValue(name, out var varInfos)) {
                    bool find = false;
                    foreach (var blockInfo in s_LexicalScopeStack) {
                        int blockId = blockInfo.BlockId;
                        if (varInfos.TryGetValue(blockId, out varInfo)) {
                            find = true;
                            break;
                        }
                    }
                    if (!find) {
                        varInfos.TryGetValue(0, out varInfo);
                    }
                    if (null != varInfo && hasLocalVar) {
                        Debug.Assert(null != funcInfo);
                        Console.WriteLine("[error]: the local variable '{0}' of function '{1}' conflicts with global variable. please rename it !", name, funcInfo.Name);
                    }
                }
                if (!s_IsVectorizing && null != varInfo && (usage == VarUsage.Read || usage == VarUsage.Write || usage == VarUsage.ObjSet)) {
                    if (null != funcInfo) {
                        if (!funcInfo.UsingGlobals.Contains(name))
                            funcInfo.UsingGlobals.Add(name);
                    }
                    else {
                        if (!s_InitGlobals.Contains(name))
                            s_InitGlobals.Add(name);
                    }
                }
            }
            if (!s_IsVectorizing && null != varInfo) {
                var curBlockInfo = CurBlockInfo();
                Debug.Assert(null == curBlockInfo && null == funcInfo || null != curBlockInfo);
                if (null != curBlockInfo) {
                    switch (usage) {
                        case VarUsage.Read:
                            if (curBlockInfo.CurBasicBlockStatementUsingVars.TryGetValue(name, out var bvrInfo)) {
                                bvrInfo.Type = varInfo.Type;
                                bvrInfo.OwnerBlock = varInfo.OwnerBlock;
                                bvrInfo.OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex;
                                bvrInfo.OwnerStatementIndex = varInfo.OwnerStatementIndex;
                            }
                            else {
                                curBlockInfo.CurBasicBlockStatementUsingVars.Add(name, new BlockVarInfo { Name = name, Type = varInfo.Type, OwnerBlock = varInfo.OwnerBlock, OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex, OwnerStatementIndex = varInfo.OwnerStatementIndex });
                            }
                            break;
                        case VarUsage.Write:
                            if (curBlockInfo.CurBasicBlockStatementSetVars.TryGetValue(name, out var bvwInfo)) {
                                bvwInfo.Type = varInfo.Type;
                                bvwInfo.OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex;
                                bvwInfo.OwnerStatementIndex = varInfo.OwnerStatementIndex;
                            }
                            else {
                                curBlockInfo.CurBasicBlockStatementSetVars.Add(name, new BlockVarInfo { Name = name, Type = varInfo.Type, OwnerBlock = varInfo.OwnerBlock, OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex, OwnerStatementIndex = varInfo.OwnerStatementIndex });
                            }
                            break;
                        case VarUsage.ObjSet:
                            if (curBlockInfo.CurBasicBlockStatementSetObjs.TryGetValue(name, out var bvoInfo)) {
                                bvoInfo.Type = varInfo.Type;
                                bvoInfo.OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex;
                                bvoInfo.OwnerStatementIndex = varInfo.OwnerStatementIndex;
                            }
                            else {
                                curBlockInfo.CurBasicBlockStatementSetObjs.Add(name, new BlockVarInfo { Name = name, Type = varInfo.Type, OwnerBlock = varInfo.OwnerBlock, OwnerBasicBlockIndex = varInfo.OwnerBasicBlockIndex, OwnerStatementIndex = varInfo.OwnerStatementIndex });
                            }
                            if (null != funcInfo) {
                                foreach (var p in funcInfo.Params) {
                                    if (!p.IsInOut && !p.IsOut && p.Name == name) {
                                        var mps = funcInfo.ModifiedInParams;
                                        if (!mps.Contains(name))
                                            mps.Add(name);
                                        break;
                                    }
                                }
                            }
                            break;
                    }
                }
            }
            return varInfo;
        }

        private static StringBuilder NewStringBuilder()
        {
            return s_StringBuilderPool.Alloc();
        }
        private static void RecycleStringBuilder(StringBuilder sb)
        {
            s_StringBuilderPool.Recycle(sb);
        }

        public static string FloatToString(float v)
        {
            if (v > -1e28 && v < 1e28)
                return v.ToString(s_FloatFormat);
            else
                return string.Format("{0:}", v);
        }
        public static string DoubleToString(double v)
        {
            if (v > -1e28 && v < 1e28)
                return v.ToString(s_DoubleFormat);
            else
                return string.Format("{0}", v);
        }

        internal enum VarUsage
        {
            Find = 0,
            Read,
            Write,
            ObjSet,
        }
        internal sealed class VarInfo
        {
            internal string Name = string.Empty;
            internal string Type = string.Empty;
            internal string OriType = string.Empty;
            internal bool IsConst = false;
            internal bool IsInOut = false;
            internal bool IsOut = false;
            internal List<string> Modifiers = new List<string>();
            internal string Semantic = string.Empty;
            internal string Register = string.Empty;
            internal Dsl.ISyntaxComponent? DefaultValueSyntax = null;
            internal string InitOrDefValueConst = string.Empty;

            internal BlockInfo? OwnerBlock = null;
            internal int OwnerBasicBlockIndex = -1;
            internal int OwnerStatementIndex = -1;
        }
        internal sealed class FuncInfo
        {
            internal string Name = string.Empty;
            internal string Signature = string.Empty;
            internal bool HasInOutOrOutParams = false;
            internal bool HasBranches = false;

            internal bool ParseAndPruned = false;
            internal bool Transformed = false;
            internal bool BlockInfoConstructed = false;
            internal bool CodeGenerateEnabled = false;

            internal List<VarInfo> Params = new List<VarInfo>();
            internal VarInfo? RetInfo = null;
            internal List<VarInfo> InOutOrOutParams = new List<VarInfo>();
            internal BlockInfo? ToplevelBlock = null;
            internal List<VectorialFuncInfo> Vectorizations = new List<VectorialFuncInfo>();
            internal int VectorizeNo = 0;
            internal List<string> UsingGlobals = new List<string>();
            internal HashSet<string> ModifiedInParams = new HashSet<string>();

            internal Dictionary<string, string> LocalTypeDefs = new Dictionary<string, string>();
            internal Dictionary<string, Dictionary<int, VarInfo>> LocalVarInfos = new Dictionary<string, Dictionary<int, VarInfo>>();
            internal HashSet<string> UsingFuncOrApis = new HashSet<string>();

            internal Dsl.ISyntaxComponent? LastAttribute = null;

            internal void ClearBlockInfo()
            {
                Debug.Assert(null != ToplevelBlock);
                BlockInfoConstructed = false;
                ToplevelBlock.ClearChildren();
            }
            internal void ClearForReTransform()
            {
                Transformed = false;
                LocalTypeDefs.Clear();
                LocalVarInfos.Clear();
                UsingFuncOrApis.Clear();
                LastAttribute = null;
            }
            internal void ResetScalarFuncInfo()
            {
                foreach (var p in Params) {
                    p.Type = p.OriType;
                }
                if (null != RetInfo) {
                    RetInfo.Type = RetInfo.OriType;
                }
            }

            internal bool IsVoid()
            {
                return null == RetInfo || RetInfo.Type == "void";
            }
            internal string GetVecNoTag()
            {
                if (VectorizeNo == 0)
                    return string.Empty;
                else
                    return VectorizeNo.ToString();
            }
            internal VectorialFuncInfo GetVectorialFuncInfo()
            {
                return Vectorizations[VectorizeNo];
            }
        }
        internal sealed class DeclVarInfo
        {
            internal string Name = string.Empty;
            internal string Type = string.Empty;
        }
        internal sealed class BlockVarInfo
        {
            internal string Name = string.Empty;
            internal string Type = string.Empty;
            internal BlockInfo? OwnerBlock = null;
            internal int OwnerBasicBlockIndex = -1;
            internal int OwnerStatementIndex = -1;

            internal string? CompileTimeConst = null;
            internal string GetCompileTimeConstDesc()
            {
                if (null == CompileTimeConst) {
                    return "[ ]";
                }
                else if (string.IsNullOrEmpty(CompileTimeConst)) {
                    return "[x]";
                }
                else {
                    return CompileTimeConst;
                }
            }
        }
        internal sealed class BasicBlockStatementInfo
        {
            internal Dsl.ISyntaxComponent? Statement = null;
            internal Dsl.ISyntaxComponent? Attribute = null;

            internal Dictionary<string, DeclVarInfo> DeclVars = new Dictionary<string, DeclVarInfo>();
            internal Dictionary<string, BlockVarInfo> UsingVars = new Dictionary<string, BlockVarInfo>();
            internal Dictionary<string, BlockVarInfo> SetVars = new Dictionary<string, BlockVarInfo>();
            internal Dictionary<string, BlockVarInfo> SetObjs = new Dictionary<string, BlockVarInfo>();

            internal bool SetVarConst(string name, string val)
            {
                bool ret = false;
                if (SetVars.TryGetValue(name, out var vinfo)) {
                    vinfo.CompileTimeConst = val;
                    ret = true;
                }
                return ret;
            }
            internal bool CacheVarConst(string name, string val)
            {
                bool ret = false;
                if (UsingVars.TryGetValue(name, out var vinfo)) {
                    vinfo.CompileTimeConst = val;
                    ret = true;
                }
                return ret;
            }
            internal bool TryGetVarSetConst(string name, out string val)
            {
                bool ret = false;
                val = string.Empty;
                if (SetVars.TryGetValue(name, out var vinfo)) {
                    if (null != vinfo.CompileTimeConst) {
                        val = vinfo.CompileTimeConst;
                        ret = true;
                    }
                }
                return ret;
            }
            internal bool TryGetVarCacheConst(string name, out string val)
            {
                bool ret = false;
                val = string.Empty;
                if (UsingVars.TryGetValue(name, out var vinfo)) {
                    if (null != vinfo.CompileTimeConst) {
                        val = vinfo.CompileTimeConst;
                        ret = true;
                    }
                }
                return ret;
            }
        }
        internal sealed class BasicBlockInfo
        {
            internal int CurStatementIndex = -1;
            internal List<BasicBlockStatementInfo> Statements = new List<BasicBlockStatementInfo>();

            internal BasicBlockStatementInfo CurStatementInfo()
            {
                Debug.Assert(CurStatementIndex >= 0 && CurStatementIndex < Statements.Count);
                return Statements[CurStatementIndex];
            }
            internal BasicBlockStatementInfo GetOrAddStatement(int index)
            {
                Debug.Assert(index >= 0 && index <= Statements.Count);
                BasicBlockStatementInfo stmInfo;
                if (index == Statements.Count) {
                    stmInfo = new BasicBlockStatementInfo();
                    Statements.Add(stmInfo);
                }
                else {
                    stmInfo = Statements[index];
                }
                return stmInfo;
            }

            internal bool SetVarConst(string name, string val)
            {
                bool ret = CurStatementInfo().SetVarConst(name, val);
                return ret;
            }
            internal bool CacheVarConst(int statementIndex, string name, string val)
            {
                if (statementIndex == -1)
                    statementIndex = Statements.Count - 1;
                bool ret = Statements[statementIndex].CacheVarConst(name, val);
                return ret;
            }
            internal bool TryGetVarSetConst(int endStatementIndex, string name, out string val)
            {
                if (endStatementIndex == -1)
                    endStatementIndex = Statements.Count - 1;
                bool ret = false;
                val = string.Empty;
                for (int ix = endStatementIndex; ix >= 0; --ix) {
                    var stmInfo = Statements[ix];
                    if (stmInfo.TryGetVarSetConst(name, out var v)) {
                        val = v;
                        ret = true;
                        break;
                    }
                }
                return ret;
            }
            internal bool TryGetVarSetOrCacheConst(int endStatementIndex, string name, out string val)
            {
                if (endStatementIndex == -1)
                    endStatementIndex = Statements.Count - 1;
                bool ret = false;
                val = string.Empty;
                for (int ix = endStatementIndex; ix >= 0; --ix) {
                    var stmInfo = Statements[ix];
                    if (stmInfo.SetVars.ContainsKey(name)) {
                        // If there is both a set and a read in the same statement and it is not a simple statement, then constant propagation
                        // cannot be performed (because our data structure is not further subdivided into the parts of the statement, and cannot
                        // be determined).
                        var funcData = stmInfo.Statement as Dsl.FunctionData;
                        if (null != funcData && funcData.GetParamClassUnmasked() == (int)Dsl.FunctionData.ParamClassEnum.PARAM_CLASS_OPERATOR) {
                            bool isVal = true;
                            foreach(var p in funcData.Params) {
                                if(p is Dsl.ValueData) {

                                }
                                else {
                                    isVal = false;
                                    break;
                                }
                            }
                            if (isVal) {
                                //For compound assignments, the value of the previous statement needs to be taken
                                //before performing the calculation and assignment.
                                if (ix < endStatementIndex && stmInfo.TryGetVarSetConst(name, out var v)) {
                                    val = v;
                                    ret = true;
                                    break;
                                }
                            }
                            else {
                                val = string.Empty;
                                ret = true;
                                break;
                            }
                        }
                        else {
                            val = string.Empty;
                            ret = true;
                            break;
                        }
                    }
                    else if (stmInfo.TryGetVarSetConst(name, out var v)) {
                        val = v;
                        ret = true;
                        break;
                    }
                    else if (stmInfo.TryGetVarCacheConst(name, out v)) {
                        val = v;
                        ret = true;
                        break;
                    }
                }
                return ret;
            }

            internal bool FindDeclVarInfo(string name, out BasicBlockStatementInfo? basicBlockStmInfo, out DeclVarInfo? vinfo)
            {
                bool ret = false;
                basicBlockStmInfo = null;
                vinfo = null;
                foreach (var bbsi in Statements) {
                    if (bbsi.DeclVars.TryGetValue(name, out vinfo)) {
                        basicBlockStmInfo = bbsi;
                        ret = true;
                        break;
                    }
                }
                return ret;
            }
            internal bool FindVarInfo(int endBasicBlockStatementIndex, string name, out BasicBlockStatementInfo? basicBlockStmInfo, out BlockVarInfo? vinfo)
            {
                bool ret = false;
                basicBlockStmInfo = null;
                vinfo = null;
                if (endBasicBlockStatementIndex < 0)
                    endBasicBlockStatementIndex = Statements.Count - 1;
                for (int ix = 0; ix <= endBasicBlockStatementIndex && ix < Statements.Count; ++ix) {
                    var bbsi = Statements[ix];
                    if (bbsi.UsingVars.TryGetValue(name, out vinfo)) {
                        basicBlockStmInfo = bbsi;
                        ret = true;
                        break;
                    }
                    else if (bbsi.SetVars.TryGetValue(name, out vinfo)) {
                        basicBlockStmInfo = bbsi;
                        ret = true;
                        break;
                    }
                    else if (bbsi.SetObjs.TryGetValue(name, out vinfo)) {
                        basicBlockStmInfo = bbsi;
                        ret = true;
                        break;
                    }
                }
                return ret;
            }

            internal void Print(int indent, int index)
            {
                Console.WriteLine("{0}===basic block[index:{1}]===", GetIndentString(indent), index);
                Console.WriteLine("{0}decl vars:", GetIndentString(indent));
                for (int ix = 0; ix < Statements.Count; ++ix) {
                    var stmInfo = Statements[ix];
                    foreach (var pair in stmInfo.DeclVars) {
                        var vinfo = pair.Value;
                        Console.WriteLine("{0}{1}:{2}[at {3}]", GetIndentString(indent + 1), vinfo.Name, vinfo.Type, ix);
                    }
                }
                Console.WriteLine("{0}using vars:", GetIndentString(indent));
                for (int ix = 0; ix < Statements.Count; ++ix) {
                    var stmInfo = Statements[ix];
                    foreach (var pair in stmInfo.UsingVars) {
                        var vinfo = pair.Value;
                        Console.WriteLine("{0}{1}:{2}[at {3}] = {4} <- {5}[{6}][{7}]", GetIndentString(indent + 1), vinfo.Name, vinfo.Type, ix, vinfo.GetCompileTimeConstDesc(), null != vinfo.OwnerBlock ? vinfo.OwnerBlock.BlockId : "MaybeGlobal", vinfo.OwnerBasicBlockIndex, vinfo.OwnerStatementIndex);
                    }
                }
                Console.WriteLine("{0}set vars:", GetIndentString(indent));
                for (int ix = 0; ix < Statements.Count; ++ix) {
                    var stmInfo = Statements[ix];
                    foreach (var pair in stmInfo.SetVars) {
                        var vinfo = pair.Value;
                        Console.WriteLine("{0}{1}:{2}[at {3}] = {4} <- {5}[{6}][{7}]", GetIndentString(indent + 1), vinfo.Name, vinfo.Type, ix, vinfo.GetCompileTimeConstDesc(), null != vinfo.OwnerBlock ? vinfo.OwnerBlock.BlockId : "MaybeGlobal", vinfo.OwnerBasicBlockIndex, vinfo.OwnerStatementIndex);
                    }
                }
                Console.WriteLine("{0}set objs:", GetIndentString(indent));
                for (int ix = 0; ix < Statements.Count; ++ix) {
                    var stmInfo = Statements[ix];
                    foreach (var pair in stmInfo.SetObjs) {
                        var vinfo = pair.Value;
                        Console.WriteLine("{0}{1}:{2}[at {3}] = {4} <- {5}[{6}][{7}]", GetIndentString(indent + 1), vinfo.Name, vinfo.Type, ix, vinfo.GetCompileTimeConstDesc(), null != vinfo.OwnerBlock ? vinfo.OwnerBlock.BlockId : "MaybeGlobal", vinfo.OwnerBasicBlockIndex, vinfo.OwnerStatementIndex);
                    }
                }
            }
        }
        internal sealed class BlockInfo
        {
            internal int BlockId = 0;
            internal int CurBasicBlockIndex = 0;

            internal Dictionary<string, string> VarTypesOnPrologue = new Dictionary<string, string>();
            internal Dictionary<string, string> VarCopyTemporaries = new Dictionary<string, string>();
            internal Dictionary<string, string> VarTemporaries = new Dictionary<string, string>();
            internal Dictionary<string, string> AllVarTemporaries = new Dictionary<string, string>();

            internal BlockInfo? Parent = null;
            internal List<BlockInfo> ChildBlocks = new List<BlockInfo>();
            internal List<BasicBlockInfo> BasicBlocks = new List<BasicBlockInfo>();

            internal FuncInfo? OwnerFunc = null;
            internal Dsl.ISyntaxComponent? Syntax = null;
            internal int FuncSyntaxIndex = 0;
            internal Dsl.ISyntaxComponent? Attribute = null;

            internal List<BlockInfo> SubsequentBlocks = new List<BlockInfo>();

            internal Dictionary<string, DeclVarInfo> CurBasicBlockStatementDeclVars
            {
                get {
                    return CurBasicBlockStatement().DeclVars;
                }
            }
            internal Dictionary<string, BlockVarInfo> CurBasicBlockStatementUsingVars
            {
                get {
                    return CurBasicBlockStatement().UsingVars;
                }
            }
            internal Dictionary<string, BlockVarInfo> CurBasicBlockStatementSetVars
            {
                get {
                    return CurBasicBlockStatement().SetVars;
                }
            }
            internal Dictionary<string, BlockVarInfo> CurBasicBlockStatementSetObjs
            {
                get {
                    return CurBasicBlockStatement().SetObjs;
                }
            }

            internal BlockInfo()
            {
                BasicBlocks.Add(new BasicBlockInfo());
                CurBasicBlockIndex = 0;
            }
            internal int FindChildIndex(BlockInfo blockInfo, out int sindex)
            {
                sindex = -1;
                int index = -1;
                for (int ix = 0; ix < ChildBlocks.Count; ++ix) {
                    var bi = ChildBlocks[ix];
                    if (bi == blockInfo) {
                        index = ix;
                        sindex = -1;
                        break;
                    }
                    else if (bi.Syntax == blockInfo.Syntax) {
                        int six = bi.SubsequentBlocks.IndexOf(blockInfo);
                        if (six >= 0) {
                            index = ix;
                            sindex = six;
                            break;
                        }
                    }
                }
                return index;
            }
            internal BlockInfo? FindChild(Dsl.ISyntaxComponent syntax, int ix)
            {
                BlockInfo? blockInfo = null;
                foreach (var bi in ChildBlocks) {
                    if (bi.Syntax == syntax) {
                        if (bi.FuncSyntaxIndex == ix) {
                            blockInfo = bi;
                            Debug.Assert(blockInfo.FuncSyntaxIndex == 0);
                        }
                        else {
                            int index = 1;
                            foreach (var sbi in bi.SubsequentBlocks) {
                                if (sbi.FuncSyntaxIndex == ix) {
                                    blockInfo = sbi;
                                    Debug.Assert(blockInfo.FuncSyntaxIndex == index);
                                    break;
                                }
                                ++index;
                            }
                        }
                        break;
                    }
                }
                return blockInfo;
            }
            internal void AddChild(BlockInfo child)
            {
                Debug.Assert(null != child.Syntax);
                bool isNewChild = true;
                if (ChildBlocks.Count > 0) {
                    var lastChild = ChildBlocks[ChildBlocks.Count - 1];
                    if (lastChild.Syntax == child.Syntax && child.FuncSyntaxIndex > 0) {
                        lastChild.SubsequentBlocks.Add(child);
                        isNewChild = false;
                    }
                }
                if (isNewChild) {
                    ChildBlocks.Add(child);
                    BasicBlocks.Add(new BasicBlockInfo());
                }
            }
            internal void ClearChildren()
            {
                ChildBlocks.Clear();
                BasicBlocks.Clear();
                BasicBlocks.Add(new BasicBlockInfo());
                CurBasicBlockIndex = 0;
            }
            internal void SetOrAddCurStatement(Dsl.ISyntaxComponent stm)
            {
                SetOrAddCurStatement(stm, null);
            }
            internal void SetOrAddCurStatement(Dsl.ISyntaxComponent stm, Dsl.ISyntaxComponent? attr)
            {
                var bbi = CurBasicBlock();
                int index = bbi.CurStatementIndex + 1;
                var stmInfo = bbi.GetOrAddStatement(index);
                stmInfo.Statement = stm;
                if(null != attr)
                    stmInfo.Attribute = attr;
                bbi.CurStatementIndex = index;
            }

            internal BasicBlockInfo CurBasicBlock()
            {
                Debug.Assert(BasicBlocks.Count > 0 && BasicBlocks.Count == ChildBlocks.Count + 1);
                Debug.Assert(CurBasicBlockIndex >= 0 && CurBasicBlockIndex < BasicBlocks.Count);
                return BasicBlocks[CurBasicBlockIndex];
            }
            internal BasicBlockStatementInfo CurBasicBlockStatement()
            {
                var basicBlockInfo = CurBasicBlock();
                return basicBlockInfo.CurStatementInfo();
            }

            internal bool SetVarConst(string name, string constVal)
            {
                return CurBasicBlock().SetVarConst(name, constVal);
            }
            internal bool TryGetVarConstInBlockScope(int endBasicBlockIndex, int endBasicBlockStatementIndex, string name, BlockInfo queryBlock, out string val)
            {
                if (!IsFuncBlockInfoConstructed()) {
                    //It is not possible to determine whether a variable has a constant value before the data structure for
                    //data flow analysis is constructed.
                    val = string.Empty;
                    return false;
                }
                if (this != queryBlock && ExistsDeclVarInCurBlock(name)) {
                    val = string.Empty;
                    return false;
                }
                if (endBasicBlockIndex == -1)
                    endBasicBlockIndex = BasicBlocks.Count - 1;
                Debug.Assert(endBasicBlockIndex >= 0 && endBasicBlockIndex < BasicBlocks.Count);

                bool ret = false;
                if (IsLoopOrInLoop()) {
                    val = string.Empty;
                    if (ExistsSetVarInBlockScope(0, 0, name, false, queryBlock)) {
                        ret = true;
                    }
                }
                else {
                    var lastBasicBlock = BasicBlocks[endBasicBlockIndex];
                    if (lastBasicBlock.TryGetVarSetConst(endBasicBlockStatementIndex, name, out val)) {
                        ret = true;
                    }
                    else {
                        //The search for sub-statement blocks and basic blocks is performed alternately.
                        for (int ix = endBasicBlockIndex - 1; ix >= 0; --ix) {
                            var blockInfo = ChildBlocks[ix];
                            Debug.Assert(null != blockInfo);
                            bool isLoop = blockInfo.IsLoop();
                            bool isIfBranches = blockInfo.IsIfBranches();
                            bool isSwitchBranches = blockInfo.IsSwitchBranches();
                            if (isLoop) {
                                val = string.Empty;
                                if (blockInfo.ExistsSetVarInBlockScope(0, 0, name, false, queryBlock)) {
                                    ret = true;
                                    break;
                                }
                            }
                            else if (isIfBranches) {
                                if (blockInfo.IsFullIfBranches()) {
                                    bool first = blockInfo.TryGetVarConstInBlockScope(-1, -1, name, queryBlock, out var val0);
                                    bool isSame = true;
                                    foreach (var sbi in blockInfo.SubsequentBlocks) {
                                        if (sbi.TryGetVarConstInBlockScope(-1, -1, name, queryBlock, out var tval)) {
                                            if (first) {
                                                if (val0 != tval) {
                                                    isSame = false;
                                                    break;
                                                }
                                            }
                                            else {
                                                isSame = false;
                                                break;
                                            }
                                        }
                                    }
                                    if (isSame) {
                                        if (first) {
                                            val = val0;
                                            ret = true;
                                            break;
                                        }
                                    }
                                    else {
                                        val = string.Empty;
                                        ret = true;
                                        break;
                                    }
                                }
                                else {
                                    val = string.Empty;
                                    if (blockInfo.ExistsSetVarInBlockScope(0, 0, name, false, queryBlock)) {
                                        ret = true;
                                        break;
                                    }
                                    foreach (var sbi in blockInfo.SubsequentBlocks) {
                                        if (sbi.ExistsSetVarInBlockScope(0, 0, name, false, queryBlock)) {
                                            ret = true;
                                            break;
                                        }
                                    }
                                    if (ret)
                                        break;
                                }
                            }
                            else if (isSwitchBranches) {
                                if (blockInfo.IsFullSwitchBranches()) {
                                    bool init = false;
                                    bool first = true;
                                    string val0 = string.Empty;
                                    bool isSame = true;
                                    foreach (var cbi in blockInfo.ChildBlocks) {
                                        bool r = cbi.TryGetVarConstInBlockScope(-1, -1, name, queryBlock, out var tval);
                                        if (!init) {
                                            init = true;
                                            first = r;
                                            val0 = tval;
                                        }
                                        else if (r) {
                                            if (first) {
                                                if (val0 != tval) {
                                                    isSame = false;
                                                    break;
                                                }
                                            }
                                            else {
                                                isSame = false;
                                                break;
                                            }
                                        }
                                    }
                                    if (isSame) {
                                        if (first) {
                                            val = val0;
                                            ret = true;
                                            break;
                                        }
                                    }
                                    else {
                                        val = string.Empty;
                                        ret = true;
                                        break;
                                    }
                                }
                                else {
                                    val = string.Empty;
                                    if (blockInfo.ExistsSetVarInBlockScope(0, 0, name, false, queryBlock)) {
                                        ret = true;
                                        break;
                                    }
                                }
                            }
                            else {
                                bool find = blockInfo.TryGetVarConstInBlockScope(-1, -1, name, queryBlock, out var val0);
                                if (find) {
                                    val = val0;
                                    ret = true;
                                    break;
                                }
                            }
                            var basicBlock = BasicBlocks[ix];
                            if (basicBlock.TryGetVarSetConst(-1, name, out val)) {
                                ret = true;
                                break;
                            }
                        }
                    }
                }
                return ret;
            }
            internal bool TryGetVarConstInParent(string name, BlockInfo queryBlock, out string val)
            {
                if (!IsFuncBlockInfoConstructed()) {
                    //It is not possible to determine whether a variable has a constant value before the data structure for
                    //data flow analysis is constructed.
                    val = string.Empty;
                    return false;
                }
                bool ret = false;
                val = string.Empty;
                var parent = Parent;
                if (null != parent) {
                    int ix = parent.FindChildIndex(this, out var six);
                    //It is not necessary to determine whether a branch of a sub-block has multiple branches, and the assignments
                    //in other branches will not affect the current branch. Therefore, only the leading nodes of the current
                    //sub-block need to be processed.
                    if (parent.TryGetVarConstInBlockScope(ix, -1, name, queryBlock, out val)) {
                        ret = true;
                    }
                    else {
                        bool findDecl = parent.ExistsDeclVarInCurBlock(name);
                        if (!findDecl) {
                            ret = parent.TryGetVarConstInParent(name, queryBlock, out val);
                        }
                    }
                }
                return ret;
            }
            internal bool TryGetVarConstInBasicBlock(int basicBlockIndex, int basicBlockStatementIndex, string name, out string val)
            {
                if (!IsFuncBlockInfoConstructed()) {
                    //It is not possible to determine whether a variable has a constant value before the data structure for
                    //data flow analysis is constructed.
                    val = string.Empty;
                    return false;
                }
                Debug.Assert(basicBlockIndex >= 0 && basicBlockIndex < BasicBlocks.Count);
                var basicBlock = BasicBlocks[basicBlockIndex];

                bool ret = false; 
                if (IsLoopOrInLoop()) {
                    val = string.Empty;
                    if (ExistsSetVarInBlockScope(0, 0, name, false, this)) {
                        ret = true;
                    }
                }
                else if (basicBlock.TryGetVarSetOrCacheConst(basicBlockStatementIndex, name, out val)) {
                    ret = true;
                }
                else if (TryGetVarConstInBlockScope(basicBlockIndex, basicBlockStatementIndex, name, this, out val)) {
                    ret = true;
                }
                else {
                    bool findDecl = ExistsDeclVarInCurBlock(name);
                    if (findDecl) {
                        ret = false;
                    }
                    else {
                        ret = TryGetVarConstInParent(name, this, out val);
                    }
                }
                if (ret) {
                    basicBlock.CacheVarConst(basicBlockStatementIndex, name, val);
                }
                return ret;
            }
            internal bool TryGetCurVarConst(string name, out string val)
            {
                return TryGetVarConstInBasicBlock(CurBasicBlockIndex, CurBasicBlock().CurStatementIndex, name, out val);
            }

            internal bool FindVarInfoInBlockScope(int endBasicBlockIndex, int endBasicBlockStatementIndex, string name, BlockInfo queryBlock, out BlockInfo? blockInfo, out BasicBlockInfo? basicBlockInfo, out BasicBlockStatementInfo? basicBlockStmInfo, out BlockVarInfo? vinfo)
            {
                blockInfo = null;
                basicBlockInfo = null;
                basicBlockStmInfo = null;
                vinfo = null;
                if (this != queryBlock && ExistsDeclVarInCurBlock(name)) {
                    return false;
                }
                if (endBasicBlockIndex == -1)
                    endBasicBlockIndex = BasicBlocks.Count - 1;
                Debug.Assert(endBasicBlockIndex >= 0 && endBasicBlockIndex < BasicBlocks.Count);

                bool ret = false;
                var lastBasicBlock = BasicBlocks[endBasicBlockIndex];
                if (lastBasicBlock.FindVarInfo(endBasicBlockStatementIndex, name, out basicBlockStmInfo, out vinfo)) {
                    blockInfo = this;
                    basicBlockInfo = lastBasicBlock;
                    ret = true;
                }
                else {
                    //The search for sub-statement blocks and basic blocks is performed alternately.
                    for (int ix = endBasicBlockIndex - 1; ix >= 0; --ix) {
                        var cblockInfo = ChildBlocks[ix];
                        Debug.Assert(null != cblockInfo);
                        if (cblockInfo.FindVarInfoInBlockScope(-1, -1, name, queryBlock, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo)) {
                            ret = true;
                            break;
                        }
                        foreach (var sbi in cblockInfo.SubsequentBlocks) {
                            if (sbi.FindVarInfoInBlockScope(-1, -1, name, queryBlock, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo)) {
                                ret = true;
                                break;
                            }
                        }
                        var basicBlock = BasicBlocks[ix];
                        if (basicBlock.FindVarInfo(-1, name, out basicBlockStmInfo, out vinfo)) {
                            blockInfo = this;
                            basicBlockInfo = basicBlock;
                            ret = true;
                            break;
                        }
                    }
                }
                return ret;
            }
            internal bool FindVarInfoInParent(string name, BlockInfo queryBlock, out BlockInfo? blockInfo, out BasicBlockInfo? basicBlockInfo, out BasicBlockStatementInfo? basicBlockStmInfo, out BlockVarInfo? vinfo)
            {
                bool ret = false;
                var parent = Parent;
                if (null != parent) {
                    int ix = parent.FindChildIndex(this, out var six);
                    //It is not necessary to determine whether a branch of a sub-block has multiple branches, and the assignments
                    //in other branches will not affect the current branch. Therefore, only the leading nodes of the current
                    //sub-block need to be processed.
                    if (parent.FindVarInfoInBlockScope(ix, -1, name, queryBlock, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo)) {
                        ret = true;
                    }
                    else {
                        bool findDecl = parent.ExistsDeclVarInCurBlock(name);
                        if (!findDecl) {
                            ret = parent.FindVarInfoInParent(name, queryBlock, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo);
                        }
                    }
                }
                else {
                    blockInfo = null;
                    basicBlockInfo = null;
                    basicBlockStmInfo = null;
                    vinfo = null;
                }
                return ret;
            }
            internal bool FindVarInfoInBasicBlock(int basicBlockIndex, int basicBlockStatementIndex, string name, out BlockInfo? blockInfo, out BasicBlockInfo? basicBlockInfo, out BasicBlockStatementInfo? basicBlockStmInfo, out BlockVarInfo? vinfo)
            {
                bool ret;
                if (FindVarInfoInBlockScope(basicBlockIndex, basicBlockStatementIndex, name, this, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo)) {
                    ret = true;
                }
                else {
                    bool findDecl = ExistsDeclVarInCurBlock(name);
                    if (findDecl) {
                        ret = false;
                    }
                    else {
                        ret = FindVarInfoInParent(name, this, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo);
                    }
                }
                return ret;
            }
            internal bool FindVarInfo(string name, out BlockInfo? blockInfo, out BasicBlockInfo? basicBlockInfo, out BasicBlockStatementInfo? basicBlockStmInfo, out BlockVarInfo? vinfo)
            {
                return FindVarInfoInBasicBlock(CurBasicBlockIndex, CurBasicBlock().CurStatementIndex, name, out blockInfo, out basicBlockInfo, out basicBlockStmInfo, out vinfo);
            }

            internal Dictionary<string, BlockVarInfo> GetUsingOuterVarsInBlockScope(bool excludeDeclInCurBlock)
            {
                var allDeclVars = new HashSet<string>();
                var dict = new Dictionary<string, BlockVarInfo>();
                QueryUsingOuterVarsInBlockScope(excludeDeclInCurBlock, this, allDeclVars, dict);
                return dict;
            }
            internal Dictionary<string, BlockVarInfo> GetSetOuterVarsInBlockScope(bool excludeDeclInCurBlock)
            {
                var allDeclVars = new HashSet<string>();
                var dict = new Dictionary<string, BlockVarInfo>();
                QuerySetOuterVarsInBlockScope(excludeDeclInCurBlock, this, allDeclVars, dict);
                return dict;
            }
            internal Dictionary<string, BlockVarInfo> GetSetOuterObjsInBlockScope(bool excludeDeclInCurBlock)
            {
                var allDeclVars = new HashSet<string>();
                var dict = new Dictionary<string, BlockVarInfo>();
                QuerySetOuterObjsInBlockScope(excludeDeclInCurBlock, this, allDeclVars, dict);
                return dict;
            }
            internal void QueryUsingOuterVarsInBlockScope(bool excludeDeclInQueryBlock, BlockInfo queryBlock, HashSet<string> excludeVars, Dictionary<string, BlockVarInfo> allUsingVars)
            {
                var newExcludeVars = excludeVars;
                if (excludeDeclInQueryBlock || this != queryBlock) {
                    newExcludeVars = new HashSet<string>(excludeVars);
                    foreach (var bbi in BasicBlocks) {
                        foreach (var bbsi in bbi.Statements) {
                            foreach (var pair in bbsi.DeclVars) {
                                if (!newExcludeVars.Contains(pair.Key))
                                    newExcludeVars.Add(pair.Key);
                            }
                        }
                    }
                }
                foreach (var bbi in BasicBlocks) {
                    foreach (var bbsi in bbi.Statements) {
                        foreach (var pair in bbsi.UsingVars) {
                            if (!newExcludeVars.Contains(pair.Key))
                                allUsingVars[pair.Key] = pair.Value;
                        }
                    }
                }
                if (this != queryBlock) {
                    foreach (var sbi in SubsequentBlocks) {
                        sbi.QueryUsingOuterVarsInBlockScope(excludeDeclInQueryBlock, queryBlock, newExcludeVars, allUsingVars);
                    }
                }
                foreach (var cbi in ChildBlocks) {
                    cbi.QueryUsingOuterVarsInBlockScope(excludeDeclInQueryBlock, queryBlock, newExcludeVars, allUsingVars);
                }
            }
            internal void QuerySetOuterVarsInBlockScope(bool excludeDeclInQueryBlock, BlockInfo queryBlock, HashSet<string> excludeVars, Dictionary<string, BlockVarInfo> allSetVars)
            {
                var newExcludeVars = excludeVars;
                if (excludeDeclInQueryBlock || this != queryBlock) {
                    newExcludeVars = new HashSet<string>(excludeVars);
                    foreach (var bbi in BasicBlocks) {
                        foreach (var bbsi in bbi.Statements) {
                            foreach (var pair in bbsi.DeclVars) {
                                if (!newExcludeVars.Contains(pair.Key))
                                    newExcludeVars.Add(pair.Key);
                            }
                        }
                    }
                }
                foreach (var bbi in BasicBlocks) {
                    foreach (var bbsi in bbi.Statements) {
                        foreach (var pair in bbsi.SetVars) {
                            if (!newExcludeVars.Contains(pair.Key))
                                allSetVars[pair.Key] = pair.Value;
                        }
                    }
                }
                if (this != queryBlock) {
                    foreach (var sbi in SubsequentBlocks) {
                        sbi.QuerySetOuterVarsInBlockScope(excludeDeclInQueryBlock, queryBlock, newExcludeVars, allSetVars);
                    }
                }
                foreach (var cbi in ChildBlocks) {
                    cbi.QuerySetOuterVarsInBlockScope(excludeDeclInQueryBlock, queryBlock, newExcludeVars, allSetVars);
                }
            }
            internal void QuerySetOuterObjsInBlockScope(bool excludeDeclInQueryBlock, BlockInfo queryBlock, HashSet<string> excludeObjs, Dictionary<string, BlockVarInfo> allSetObjs)
            {
                var newExcludeVars = excludeObjs;
                if (excludeDeclInQueryBlock || this != queryBlock) {
                    newExcludeVars = new HashSet<string>(excludeObjs);
                    foreach (var bbi in BasicBlocks) {
                        foreach (var bbsi in bbi.Statements) {
                            foreach (var pair in bbsi.DeclVars) {
                                if (!newExcludeVars.Contains(pair.Key))
                                    newExcludeVars.Add(pair.Key);
                            }
                        }
                    }
                }
                foreach (var bbi in BasicBlocks) {
                    foreach (var bbsi in bbi.Statements) {
                        foreach (var pair in bbsi.SetObjs) {
                            if (!newExcludeVars.Contains(pair.Key))
                                allSetObjs[pair.Key] = pair.Value;
                        }
                    }
                }
                if (this != queryBlock) {
                    foreach (var sbi in SubsequentBlocks) {
                        sbi.QuerySetOuterObjsInBlockScope(excludeDeclInQueryBlock, queryBlock, newExcludeVars, allSetObjs);
                    }
                }
                foreach (var cbi in ChildBlocks) {
                    cbi.QuerySetOuterObjsInBlockScope(excludeDeclInQueryBlock, queryBlock, newExcludeVars, allSetObjs);
                }
            }

            internal bool ExistsUsingVarInBlockScope(int startBasicBlockIndex, int startBasicBlockStatementIndex, string name, bool excludeDeclInQueryBlock, BlockInfo queryBlock)
            {
                bool ret = false;
                bool stop = false;
                if (excludeDeclInQueryBlock || this != queryBlock) {
                    stop = ExistsDeclVarInCurBlock(name);
                }
                if (!stop) {
                    Debug.Assert(startBasicBlockIndex >= 0 && startBasicBlockIndex < BasicBlocks.Count);

                    if (IsLoop()) {
                        startBasicBlockIndex = 0;
                        startBasicBlockStatementIndex = 0;
                    }
                    for (int bbIx = startBasicBlockIndex; bbIx < BasicBlocks.Count; ++bbIx) {
                        var bbi = BasicBlocks[bbIx];
                        for (int bbsIx = startBasicBlockStatementIndex; bbsIx < bbi.Statements.Count; ++bbsIx) {
                            var bbsi = bbi.Statements[bbsIx];
                            if (bbsi.UsingVars.TryGetValue(name, out var bvinfo)) {
                                ret = true;
                                break;
                            }
                        }
                        if (ret)
                            break;
                    }
                    if (!ret) {
                        if (this != queryBlock) {
                            foreach (var sbi in SubsequentBlocks) {
                                if (sbi.ExistsUsingVarInBlockScope(0, 0, name, excludeDeclInQueryBlock, queryBlock)) {
                                    ret = true;
                                    break;
                                }
                            }
                        }
                        foreach (var cbi in ChildBlocks) {
                            if (cbi.ExistsUsingVarInBlockScope(0, 0, name, excludeDeclInQueryBlock, queryBlock)) {
                                ret = true;
                                break;
                            }
                        }
                    }
                }
                return ret;
            }
            internal bool ExistsSetVarInBlockScope(int startBasicBlockIndex, int startBasicBlockStatementIndex, string name, bool excludeDeclInQueryBlock, BlockInfo queryBlock)
            {
                bool ret = false;
                bool stop = false;
                if (excludeDeclInQueryBlock || this != queryBlock) {
                    stop = ExistsDeclVarInCurBlock(name);
                }
                if (!stop) {
                    Debug.Assert(startBasicBlockIndex >= 0 && startBasicBlockIndex < BasicBlocks.Count);

                    if (IsLoop()) {
                        startBasicBlockIndex = 0;
                        startBasicBlockStatementIndex = 0;
                    }
                    for (int bbIx = startBasicBlockIndex; bbIx < BasicBlocks.Count; ++bbIx) {
                        var bbi = BasicBlocks[bbIx];
                        for (int bbsIx = startBasicBlockStatementIndex; bbsIx < bbi.Statements.Count; ++bbsIx) {
                            var bbsi = bbi.Statements[bbsIx];
                            if (bbsi.SetVars.TryGetValue(name, out var bvinfo)) {
                                ret = true;
                                break;
                            }
                        }
                        if (ret)
                            break;
                    }
                    if (!ret) {
                        if (this != queryBlock) {
                            foreach (var sbi in SubsequentBlocks) {
                                if (sbi.ExistsSetVarInBlockScope(0, 0, name, excludeDeclInQueryBlock, queryBlock)) {
                                    ret = true;
                                    break;
                                }
                            }
                        }
                        foreach (var cbi in ChildBlocks) {
                            if (cbi.ExistsSetVarInBlockScope(0, 0, name, excludeDeclInQueryBlock, queryBlock)) {
                                ret = true;
                                break;
                            }
                        }
                    }
                }
                return ret;
            }
            internal bool ExistsSetObjInBlockScope(int startBasicBlockIndex, int startBasicBlockStatementIndex, string name, bool excludeDeclInQueryBlock, BlockInfo queryBlock)
            {
                bool ret = false;
                bool stop = false;
                if (excludeDeclInQueryBlock || this != queryBlock) {
                    stop = ExistsDeclVarInCurBlock(name);
                }
                if (!stop) {
                    Debug.Assert(startBasicBlockIndex >= 0 && startBasicBlockIndex < BasicBlocks.Count);

                    if (IsLoop()) {
                        startBasicBlockIndex = 0;
                        startBasicBlockStatementIndex = 0;
                    }
                    for (int bbIx = startBasicBlockIndex; bbIx < BasicBlocks.Count; ++bbIx) {
                        var bbi = BasicBlocks[bbIx];
                        for (int bbsIx = startBasicBlockStatementIndex; bbsIx < bbi.Statements.Count; ++bbsIx) {
                            var bbsi = bbi.Statements[bbsIx];
                            if (bbsi.SetObjs.TryGetValue(name, out var bvinfo)) {
                                ret = true;
                                break;
                            }
                        }
                        if (ret)
                            break;
                    }
                    if (!ret) {
                        if (this != queryBlock) {
                            foreach (var sbi in SubsequentBlocks) {
                                if (sbi.ExistsSetObjInBlockScope(0, 0, name, excludeDeclInQueryBlock, queryBlock)) {
                                    ret = true;
                                    break;
                                }
                            }
                        }
                        foreach (var cbi in ChildBlocks) {
                            if (cbi.ExistsSetObjInBlockScope(0, 0, name, excludeDeclInQueryBlock, queryBlock)) {
                                ret = true;
                                break;
                            }
                        }
                    }
                }
                return ret;
            }

            internal bool ExistsUsingVarInParent(string name, BlockInfo queryBlock)
            {
                bool ret = false;
                var parent = Parent;
                if (null != parent) {
                    int ix = parent.FindChildIndex(this, out var six);
                    //Other branches of the sub-block do not need to be considered (except in loops, which
                    //have been handled in the following function call).
                    if (parent.ExistsUsingVarInBlockScope(ix + 1, 0, name, false, queryBlock)) {
                        ret = true;
                    }
                    else {
                        bool findDecl = parent.ExistsDeclVarInCurBlock(name);
                        if (!findDecl) {
                            ret = parent.ExistsUsingVarInParent(name, queryBlock);
                        }
                    }
                }
                return ret;
            }
            internal bool ExistsSetVarInParent(string name, BlockInfo queryBlock)
            {
                bool ret = false;
                var parent = Parent;
                if (null != parent) {
                    int ix = parent.FindChildIndex(this, out var six);
                    //Other branches of the sub-block do not need to be considered (except in loops, which
                    //have been handled in the following function call).
                    if (parent.ExistsSetVarInBlockScope(ix + 1, 0, name, false, queryBlock)) {
                        ret = true;
                    }
                    else {
                        bool findDecl = parent.ExistsDeclVarInCurBlock(name);
                        if (!findDecl) {
                            ret = parent.ExistsSetVarInParent(name, queryBlock);
                        }
                    }
                }
                return ret;
            }
            internal bool ExistsSetObjInParent(string name, BlockInfo queryBlock)
            {
                bool ret = false;
                var parent = Parent;
                if (null != parent) {
                    int ix = parent.FindChildIndex(this, out var six);
                    //Other branches of the sub-block do not need to be considered (except in loops, which
                    //have been handled in the following function call).
                    if (parent.ExistsSetObjInBlockScope(ix + 1, 0, name, false, queryBlock)) {
                        ret = true;
                    }
                    else {
                        bool findDecl = parent.ExistsDeclVarInCurBlock(name);
                        if (!findDecl) {
                            ret = parent.ExistsSetObjInParent(name, queryBlock);
                        }
                    }
                }
                return ret;
            }

            internal bool ExistsUsingVarStartBasicBlock(int basicBlockIndex, int basicBlockStatementIndex, string name, bool excludeDeclInCurBlock)
            {
                bool ret;
                if (ExistsUsingVarInBlockScope(basicBlockIndex, basicBlockStatementIndex, name, excludeDeclInCurBlock, this)) {
                    ret = true;
                }
                else {
                    bool findDecl = ExistsDeclVarInCurBlock(name);
                    if (findDecl) {
                        ret = false;
                    }
                    else {
                        ret = ExistsUsingVarInParent(name, this);
                    }
                }
                return ret;
            }
            internal bool ExistsSetVarStartBasicBlock(int basicBlockIndex, int basicBlockStatementIndex, string name, bool excludeDeclInCurBlock)
            {
                bool ret;
                if (ExistsSetVarInBlockScope(basicBlockIndex, basicBlockStatementIndex, name, excludeDeclInCurBlock, this)) {
                    ret = true;
                }
                else {
                    bool findDecl = ExistsDeclVarInCurBlock(name);
                    if (findDecl) {
                        ret = false;
                    }
                    else {
                        ret = ExistsSetVarInParent(name, this);
                    }
                }
                return ret;
            }
            internal bool ExistsSetObjStartBasicBlock(int basicBlockIndex, int basicBlockStatementIndex, string name, bool excludeDeclInCurBlock)
            {
                bool ret;
                if (ExistsSetObjInBlockScope(basicBlockIndex, basicBlockStatementIndex, name, excludeDeclInCurBlock, this)) {
                    ret = true;
                }
                else {
                    bool findDecl = ExistsDeclVarInCurBlock(name);
                    if (findDecl) {
                        ret = false;
                    }
                    else {
                        ret = ExistsSetObjInParent(name, this);
                    }
                }
                return ret;
            }

            internal bool ExistsUsingVarInCurBlockScope(string name)
            {
                return ExistsUsingVarStartBasicBlock(CurBasicBlockIndex, CurBasicBlock().CurStatementIndex, name, false);
            }
            internal bool ExistsSetVarInCurBlockScope(string name)
            {
                return ExistsSetVarStartBasicBlock(CurBasicBlockIndex, CurBasicBlock().CurStatementIndex, name, false);
            }
            internal bool ExistsSetObjInCurBlockScope(string name)
            {
                return ExistsSetObjStartBasicBlock(CurBasicBlockIndex, CurBasicBlock().CurStatementIndex, name, false);
            }

            internal bool ExistsDeclVarInCurBlock(string name)
            {
                return FindDeclVarInCurBlock(name, out var bbi, out var bbsi, out var vi);
            }
            internal bool FindDeclVarInCurBlock(string name, out BasicBlockInfo? basicBlockInfo, out BasicBlockStatementInfo? basicBlockStmInfo, out DeclVarInfo? vinfo)
            {
                bool ret = false;
                basicBlockInfo = null;
                basicBlockStmInfo = null;
                vinfo = null;
                foreach (var bbi in BasicBlocks) {
                    if (bbi.FindDeclVarInfo(name, out basicBlockStmInfo, out vinfo))
                        break;
                }
                return ret;
            }

            internal bool IsFuncBlockInfoConstructed()
            {
                Debug.Assert(null != OwnerFunc);
                return OwnerFunc.BlockInfoConstructed;
            }
            internal bool IsLoop()
            {
                Debug.Assert(null != Syntax);
                bool ret = false;
                string syntaxId = Syntax.GetId();
                if (syntaxId == "for" || syntaxId == "while" || syntaxId == "do") {
                    ret = true;
                }
                return ret;
            }
            internal bool IsLoopOrInLoop()
            {
                bool ret = IsLoop();
                if (!ret && null != Parent) {
                    ret = Parent.IsLoopOrInLoop();
                }
                return ret;
            }
            internal bool IsIfBranches()
            {
                Debug.Assert(null != Syntax);
                bool ret = false;
                string syntaxId = Syntax.GetId();
                if (syntaxId == "if") {
                    ret = true;
                }
                return ret;
            }
            internal bool IsSwitchBranches()
            {
                Debug.Assert(null != Syntax);
                bool ret = false;
                string syntaxId = Syntax.GetId();
                if (syntaxId == "switch") {
                    ret = true;
                }
                return ret;
            }
            internal bool IsFullIfBranches()
            {
                Debug.Assert(null != Syntax);
                bool ret = false;
                string syntaxId = Syntax.GetId();
                if (syntaxId == "if") {
                    if (SubsequentBlocks.Count > 0) {
                        var stm = Syntax as Dsl.StatementData;
                        if (null != stm) {
                            foreach (var sbi in SubsequentBlocks) {
                                var func = stm.GetFunction(sbi.FuncSyntaxIndex);
                                if (null != func && func.GetId() == "else") {
                                    ret = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                return ret;
            }
            internal bool IsFullSwitchBranches()
            {
                Debug.Assert(null != Syntax);
                bool ret = false;
                string syntaxId = Syntax.GetId();
                if (syntaxId == "switch") {
                    foreach (var cbi in ChildBlocks) {
                        if (null != cbi && null != cbi.Syntax && cbi.Syntax.GetId() == "default") {
                            ret = true;
                            break;
                        }
                    }
                }
                return ret;
            }

            internal void ClearTemporaryInfo()
            {
                VarTypesOnPrologue.Clear();
                VarCopyTemporaries.Clear();
                VarTemporaries.Clear();
                AllVarTemporaries.Clear();
            }
            internal void AddCopyTemporary(string varName, string tempName)
            {
                if (!AllVarTemporaries.ContainsKey(varName)) {
                    var vinfo = GetVarInfo(varName, VarUsage.Find);
                    if (null != vinfo) {
                        VarTypesOnPrologue[varName] = vinfo.Type;
                    }
                    VarCopyTemporaries.Add(varName, tempName);
                    AllVarTemporaries.Add(varName, tempName);
                }
            }
            internal void AddTemporary(string varName, string tempName)
            {
                if (!AllVarTemporaries.ContainsKey(varName)) {
                    var vinfo = GetVarInfo(varName, VarUsage.Find);
                    if (null != vinfo) {
                        VarTypesOnPrologue[varName] = vinfo.Type;
                    }
                    VarTemporaries.Add(varName, tempName);
                    AllVarTemporaries.Add(varName, tempName);
                }
            }
            internal Dictionary<string, string> GetUnionTemporary()
            {
                var dict = new Dictionary<string, string>(VarCopyTemporaries);
                foreach (var pair in VarTemporaries) {
                    dict.Add(pair.Key, pair.Value);
                }
                return dict;
            }
            internal bool TryGetTemporary(string varName, out string? tempName)
            {
                bool ret = false;
                if (VarCopyTemporaries.TryGetValue(varName, out tempName)) {
                    ret = true;
                }
                else if (VarTemporaries.TryGetValue(varName, out tempName)) {
                    ret = true;
                }
                else {
                    ret = TryGetTemporaryInParent(varName, out tempName);
                }
                return ret;
            }
            internal bool TryGetTemporaryInParent(string varName, out string? tempName)
            {
                bool ret = false;
                bool findDecl = ExistsDeclVarInCurBlock(varName);
                if (!findDecl && null != Parent) {
                    ret = Parent.TryGetTemporary(varName, out tempName);
                }
                else {
                    tempName = string.Empty;
                }
                return ret;
            }
            internal void Print(int indent)
            {
                Print(indent, 0);
            }
            internal void Print(int indent, int index)
            {
                Console.WriteLine("{0}===block[id:{1} index:{2}]===", GetIndentString(indent), BlockId, index);
                if (null != Syntax) {
                    Console.WriteLine("{0}syntax:{1} func index:{2}", GetIndentString(indent), Syntax.GetId(), FuncSyntaxIndex);
                }
                if (null != OwnerFunc) {
                    Console.WriteLine("{0}owner func:{1}", GetIndentString(indent), OwnerFunc.Name);
                }
                Debug.Assert(BasicBlocks.Count == ChildBlocks.Count + 1);
                if (ChildBlocks.Count > 0) {
                    Console.WriteLine("{0}==BasicBlock[count:{1}+1] & ChildBlock[count:{1}]==", GetIndentString(indent), ChildBlocks.Count);
                    for (int ix = 0; ix < BasicBlocks.Count && ix < ChildBlocks.Count; ++ix) {
                        var bbi = BasicBlocks[ix];
                        bbi.Print(indent + 1, ix);
                        var cbi = ChildBlocks[ix];
                        cbi.Print(indent + 1, ix);
                    }
                }
                var lastBasicBlock = BasicBlocks[BasicBlocks.Count - 1];
                lastBasicBlock.Print(indent + 1, BasicBlocks.Count - 1);
                if (SubsequentBlocks.Count > 0) {
                    Console.WriteLine("{0}==SubsequentBlock[count:{1}]==", GetIndentString(indent), SubsequentBlocks.Count);
                    for (int ix = 0; ix < SubsequentBlocks.Count; ++ix) {
                        var sbi = SubsequentBlocks[ix];
                        sbi.Print(indent, ix);
                    }
                }
            }
        }
        internal sealed class StructInfo
        {
            internal string Name = string.Empty;
            internal List<VarInfo> Fields = new List<VarInfo>();
            internal Dictionary<string, int> FieldName2Indexes = new Dictionary<string, int>();
        }
        internal sealed class CBufferInfo
        {
            internal string Name = string.Empty;
            internal string Register = string.Empty;
            internal List<VarInfo> Variables = new List<VarInfo>();
        }
        internal enum SyntaxUsage
        {
            Anything = 0,
            Operator,
            MemberName,
            FuncName,
            TypeName,
            MaxNum
        }
        internal readonly ref struct ParseContextInfo
        {
            internal SyntaxUsage Usage { get; init; } = SyntaxUsage.Anything;
            internal bool IsInAssignLHS { get; init; } = false;
            internal bool IsObjInAssignLHS { get; init; } = false;
            internal bool IsInCondExp { get; init; } = false;
            internal bool IsInStructInitOrAssign { get; init; } = false;
            internal bool IsTopLevelStatement { get; init; } = false;

            internal string LhsType { get; init; } = string.Empty;

            public ParseContextInfo()
            { }
            public ParseContextInfo(ParseContextInfo inherit)
            {
                Usage = inherit.Usage;
                IsInAssignLHS = inherit.IsInAssignLHS;
                IsObjInAssignLHS = inherit.IsObjInAssignLHS;
                IsInCondExp = inherit.IsInCondExp;

                LhsType = inherit.LhsType;
            }
        }
        internal sealed class CaseInfo
        {
            internal Dsl.ISyntaxComponent? CaseBlock = null;
            internal List<StringBuilder> Exps = new List<StringBuilder>();
            internal List<Dsl.ISyntaxComponent> Statements = new List<Dsl.ISyntaxComponent>();
        }

        private static Dictionary<string, StructInfo> s_StructInfos = new Dictionary<string, StructInfo>();
        private static Dictionary<string, CBufferInfo> s_CBufferInfos = new Dictionary<string, CBufferInfo>();
        private static Dictionary<string, string> s_GlobalTypeDefs = new Dictionary<string, string>();
        private static Dictionary<string, Dictionary<int, VarInfo>> s_GlobalVarInfos = new Dictionary<string, Dictionary<int, VarInfo>>();
        private static List<VarInfo> s_DeclGlobals = new List<VarInfo>();
        private static List<string> s_InitGlobals = new List<string>();
        private static Dictionary<string, FuncInfo> s_FuncInfos = new Dictionary<string, FuncInfo>();
        private static HashSet<string> s_BranchRemovedFuncs = new HashSet<string>();
        private static bool s_IsVectorizing = false;
        private static bool s_IsFullVectorized = true;

        private static bool s_IsTorch = true;
        private static bool s_AutoVectorizeBranch = true;
        private static bool s_EnableVectorization = true;
        private static bool s_EnableConstPropagation = true;

        private static string s_MainEntryFuncSignature = string.Empty;
        private static SortedDictionary<string, Dsl.StatementData> s_EntryFuncs = new SortedDictionary<string, Dsl.StatementData>();
        private static Dictionary<string, HashSet<string>> s_FuncCallFuncs = new Dictionary<string, HashSet<string>>();
        private static List<string> s_AllFuncSigs = new List<string>();
        private static Dictionary<string, Dsl.StatementData> s_AllFuncDsls = new Dictionary<string, Dsl.StatementData>();
        private static Dictionary<string, StringBuilder> s_AllFuncCodes = new Dictionary<string, StringBuilder>();

        private static HashSet<string> s_AllUsingFuncOrApis = new HashSet<string>();
        private static Dictionary<string, StringBuilder> s_AutoGenCodes = new Dictionary<string, StringBuilder>();

        private static HashSet<string> s_ScalarUsingFuncOrApis = new HashSet<string>();
        private static Dictionary<string, StringBuilder> s_ScalarAutoGenCodes = new Dictionary<string, StringBuilder>();

        private static HashSet<string> s_GlobalCalledScalarFuncs = new HashSet<string>();
        private static HashSet<string> s_GlobalUsingFuncOrApis = new HashSet<string>();
        private static Dictionary<string, StringBuilder> s_GlobalAutoGenCodes = new Dictionary<string, StringBuilder>();

        private static Dictionary<string, HashSet<string>> s_FuncOverloads = new Dictionary<string, HashSet<string>>();
        private static Stack<BlockInfo> s_LexicalScopeStack = new Stack<BlockInfo>();
        private static int s_LastBlockId = 0;
        private static int s_UniqueNumber = 0;

        private static StringBuilderPool s_StringBuilderPool = new StringBuilderPool();
        private static bool s_IsDebugMode = false;
        private static bool s_UseHlsl2018 = true;

        private static Dsl.ValueData s_ConstDslValueOne = new Dsl.ValueData("1", Dsl.ValueData.NUM_TOKEN);
        private static string s_FloatFormat = "###########################0.00#####";
        private static string s_DoubleFormat = "###########################0.000000##########";

        private static List<string> s_TorchImports = new List<string> {
            "hlsl_lib_torch.py",
            "hlsl_lib_torch_swizzle.py",
        };
        private static List<string> s_NumpyImports = new List<string> {
            "hlsl_lib_numpy.py",
            "hlsl_lib_numpy_swizzle.py",
        };
        private static List<string> s_TorchImportsFor2021 = new List<string> {
            "hlsl_lib_torch.py",
            "hlsl_lib_torch_swizzle.py",
        };
        private static List<string> s_NumpyImportsFor2021 = new List<string> {
            "hlsl_lib_numpy.py",
            "hlsl_lib_numpy_swizzle.py",
        };
        private static string s_TorchInc = "hlsl_inc_torch.py";
        private static string s_NumpyInc = "hlsl_inc_numpy.py";
        private static string s_TorchIncFor2021 = "hlsl_inc_torch.py";
        private static string s_NumpyIncFor2021 = "hlsl_inc_numpy.py";

        private static Dictionary<string, string> s_HlslType2TorchTypes = new Dictionary<string, string> {
            { "half", "torch.float16" },
            { "float", "torch.float32" },
            { "double", "torch.float64" },
            { "bool", "torch.bool" },
            { "int", "torch.int32" },
            { "uint", "torch.int32" },
            { "dword", "torch.int32" },
        };
        private static Dictionary<string, string> s_HlslType2NumpyTypes = new Dictionary<string, string> {
            { "half", "np.float16" },
            { "float", "np.float32" },
            { "double", "np.float64" },
            { "bool", "np.bool" },
            { "int", "np.int32" },
            { "uint", "np.uint32" },
            { "dword", "np.uint32" },
        };

        private const string c_TupleTypePrefix = "tuple_";
        private const string c_VectorialTupleTypeTag = "_t_";
        private const string c_VectorialTypePrefix = "t_";
        private const string c_VectorialNameSuffix = "_arr";
        private const string c_VectorialAdapterNameSuffix = "_arr_adapter";
        private static Dictionary<string, string> s_OperatorNames = new Dictionary<string, string> {
            { "+", "h_add" },
            { "-", "h_sub" },
            { "*", "h_mul" },
            { "/", "h_div" },
            { "%", "h_mod" },
            { "&", "h_bitand" },
            { "|", "h_bitor" },
            { "^", "h_bitxor" },
            { "~", "h_bitnot" },
            { "<<", "h_lshift" },
            { ">>", "h_rshift" },
            { "&&", "h_and" },
            { "||", "h_or" },
            { "!", "h_not" },
            { "==", "h_equal" },
            { "!=", "h_not_equal" },
            { "<", "h_less_than" },
            { ">", "h_greater_than" },
            { "<=", "h_less_equal_than" },
            { ">=", "h_greater_equal_than" },
        };

        private static HashSet<string> s_KeepBaseTypeFuncs = new HashSet<string> {
            "h_inc",
            "h_dec",
            "h_add",
            "h_sub",
            "h_mul",
            "h_div",
            "h_mod",
            "h_bitand",
            "h_bitor",
            "h_bitxor",
            "h_bitnot",
            "h_lshift",
            "h_rshift",
        };
        private static HashSet<string> s_KeepFullTypeFuncs = new HashSet<string> {
            "h_matmul",
        };

        private static Dictionary<string, string> s_BuiltInFuncs = new Dictionary<string, string> {
            { "float", "@@" },
            { "float2", "@@" },
            { "float3", "@@" },
            { "float4", "@@" },
            { "double", "@@" },
            { "double2", "@@" },
            { "double3", "@@" },
            { "double4", "@@" },
            { "uint", "@@" },
            { "uint2", "@@" },
            { "uint3", "@@" },
            { "uint4", "@@" },
            { "dword", "@@" },
            { "dword2", "@@" },
            { "dword3", "@@" },
            { "dword4", "@@" },
            { "int", "@@" },
            { "int2", "@@" },
            { "int3", "@@" },
            { "int4", "@@" },
            { "bool", "@@" },
            { "bool2", "@@" },
            { "bool3", "@@" },
            { "bool4", "@@" },
            { "half", "@@" },
            { "half2", "@@" },
            { "half3", "@@" },
            { "half4", "@@" },
            { "float2x2", "@@" },
            { "float3x3", "@@" },
            { "float4x4", "@@" },
            { "double2x2", "@@" },
            { "double3x3", "@@" },
            { "double4x4", "@@" },
            { "uint2x2", "@@" },
            { "uint3x3", "@@" },
            { "uint4x4", "@@" },
            { "dword2x2", "@@" },
            { "dword3x3", "@@" },
            { "dword4x4", "@@" },
            { "int2x2", "@@" },
            { "int3x3", "@@" },
            { "int4x4", "@@" },
            { "bool2x2", "@@" },
            { "bool3x3", "@@" },
            { "bool4x4", "@@" },
            { "half2x2", "@@" },
            { "half3x3", "@@" },
            { "half4x4", "@@" },
            { "abort", "void" },
            { "abs", "@0" },
            { "acos", "@0" },
            { "all", "bool" },
            { "AllMemoryBarrier", "void" },
            { "AllMemoryBarrierWithGroupSync", "void" },
            { "any", "bool" },
            { "asdouble", "double$0" },
            { "asfloat", "float$0" },
            { "asin", "@0" },
            { "asint", "int$0" },
            { "asuint", "uint$0" },
            { "atan", "@0" },
            { "atan2", "@0" },
            { "ceil", "@0" },
            { "CheckAccessFullyMapped", "bool" },
            { "clamp", "@m" },
            { "clip", "void" },
            { "cos", "@0" },
            { "cosh", "@0" },
            { "countbits", "uint" },
            { "cross", "float3" },
            { "D3DCOLORtoUBYTE4", "int4" },
            { "ddx", "@0" },
            { "ddx_coarse", "float" },
            { "ddx_fine", "float" },
            { "ddy", "@0" },
            { "ddy_coarse", "float" },
            { "ddy_fine", "float" },
            { "degrees", "@0" },
            { "determinant", "float" },
            { "DeviceMemoryBarrier", "void" },
            { "DeviceMemoryBarrierWithGroupSync", "void" },
            { "distance", "float" },
            { "dot", "@0-$0" },
            { "dst", "@0" },
            { "errorf", "void" },
            { "EvaluateAttributeCentroid", "@0" },
            { "EvaluateAttributeAtSample", "@0" },
            { "EvaluateAttributeSnapped", "@0" },
            { "exp", "@0" },
            { "exp2", "@0" },
            { "f16tof32", "float" },
            { "f32tof16", "uint" },
            { "faceforward", "@0" },
            { "firstbithigh", "int"},
            { "firstbitlow", "int"},
            { "floor", "@0" },
            { "fma", "@0" },
            { "fmod", "@0" },
            { "frac", "@0" },
            { "frexp", "@0" },
            { "fwidth", "@0" },
            { "GetRenderTargetSampleCount", "uint" },
            { "GetRenderTargetSamplePosition", "float2" },
            { "GroupMemoryBarrier", "void" },
            { "GroupMemoryBarrierWithGroupSync", "void" },
            { "InterlockedAdd", "void" },
            { "InterlockedAnd", "void" },
            { "InterlockedCompareExchange", "void" },
            { "InterlockedCompareStore", "void" },
            { "InterlockedExchange", "void" },
            { "InterlockedMax", "void" },
            { "InterlockedMin", "void" },
            { "InterlockedOr", "void" },
            { "InterlockedXor", "void" },
            { "isfinite", "bool$0" },
            { "isinf", "bool$0" },
            { "isnan", "bool$0" },
            { "ldexp", "@0" },
            { "length", "float" },
            { "lerp", "@0" },
            { "lit", "float4" },
            { "log", "@0" },
            { "log10", "@0" },
            { "log2", "@0" },
            { "mad", "@0" },
            { "max", "@0" },
            { "min", "@0" },
            { "modf", "@0" },
            { "msad4", "uint4" },
            { "normalize", "@0" },
            { "pow", "@0" },
            { "printf", "void" },
            { "Process2DQuadTessFactorsAvg", "void" },
            { "Process2DQuadTessFactorsMax", "void" },
            { "Process2DQuadTessFactorsMin", "void" },
            { "ProcessIsolineTessFactors", "void" },
            { "ProcessQuadTessFactorsAvg", "void" },
            { "ProcessQuadTessFactorsMax", "void" },
            { "ProcessQuadTessFactorsMin", "void" },
            { "ProcessTriTessFactorsAvg", "void" },
            { "ProcessTriTessFactorsMax", "void" },
            { "ProcessTriTessFactorsMin", "void" },
            { "radians", "@0" },
            { "rcp", "@0" },
            { "reflect", "@0" },
            { "refract", "@0" },
            { "reversebits", "uint" },
            { "round", "@0" },
            { "rsqrt", "@0" },
            { "saturate", "@0" },
            { "sign", "int$0" },
            { "sin", "@0" },
            { "sincos", "@0" },
            { "sinh", "@0" },
            { "smoothstep", "@m" },
            { "sqrt", "@0" },
            { "step", "@m" },
            { "tan", "@0" },
            { "tanh", "@0" },
            { "tex1D", "float4" },
            { "tex1Dbias", "float4" },
            { "tex1Dgrad", "float4" },
            { "tex1Dlod", "float4" },
            { "tex1Dproj", "float4" },
            { "tex2D", "float4" },
            { "tex2Dbias", "float4" },
            { "tex2Dgrad", "float4" },
            { "tex2Dlod", "float4" },
            { "tex2Dproj", "float4" },
            { "tex3D", "float4" },
            { "tex3Dbias", "float4" },
            { "tex3Dgrad", "float4" },
            { "tex3Dlod", "float4" },
            { "tex3Dproj", "float4" },
            { "texCUBE", "float4" },
            { "texCUBEbias", "float4" },
            { "texCUBEgrad", "float4" },
            { "texCUBElod", "float4" },
            { "texCUBEproj", "float4" },
            { "transpose", "@0" },
            { "trunc", "@0" },
        };
        private static Dictionary<string, Dictionary<string, string>> s_BuiltInMemFuncs = new Dictionary<string, Dictionary<string, string>> {
            { "Texture2D", new Dictionary<string, string> {
                    { "Load", "@0-$04" },
                    { "Sample", "@1-$14" },
                    { "SampleBias", "@1-$14" },
                    { "SampleLevel", "@1-$14" },
                    { "SampleGrad", "@1-$14" },
                    { "Gather", "@1-$14" },
                }},
            { "Texture2DArray", new Dictionary<string, string> {
                    { "Load", "@0-$04" },
                    { "Sample", "@1-$14" },
                    { "SampleBias", "@1-$14" },
                    { "SampleLevel", "@1-$14" },
                    { "SampleGrad", "@1-$14" },
                    { "Gather", "@1-$14" },
                }},
            { "Texture3D", new Dictionary<string, string> {
                    { "Load", "@0-$04" },
                    { "Sample", "@1-$14" },
                    { "SampleBias", "@1-$14" },
                    { "SampleLevel", "@1-$14" },
                    { "SampleGrad", "@1-$14" },
                    { "Gather", "@1-$14" },
                }},
            { "TextureCube", new Dictionary<string, string> {
                    { "Load", "@0-$04" },
                    { "Sample", "@1-$14" },
                    { "SampleBias", "@1-$14" },
                    { "SampleLevel", "@1-$14" },
                    { "SampleGrad", "@1-$14" },
                    { "Gather", "@1-$14" },
                }},
            { "TextureCubeArray", new Dictionary<string, string> {
                    { "Load", "@0-$04" },
                    { "Sample", "@1-$14" },
                    { "SampleBias", "@1-$14" },
                    { "SampleLevel", "@1-$14" },
                    { "SampleGrad", "@1-$14" },
                    { "Gather", "@1-$14" },
                }},
            { "ByteAddressBuffer", new Dictionary<string, string> {
                    { "Load", "@0-$0" },
                    { "Load2", "@0-$02" },
                    { "Load3", "@0-$03" },
                    { "Load4", "@0-$04" },
                }},
            { "RWByteAddressBuffer", new Dictionary<string, string> {
                    { "Load", "@0-$0" },
                    { "Load2", "@0-$02" },
                    { "Load3", "@0-$03" },
                    { "Load4", "@0-$04" },
                    { "Store", "void" },
                    { "Store2", "void" },
                    { "Store3", "void" },
                    { "Store4", "void" },
                }},
        };
        private static Dictionary<string, Dictionary<string, string>> s_BuiltInMembers = new Dictionary<string, Dictionary<string, string>> {
        };

        private static Dictionary<string, string> s_BaseTypeAbbrs = new Dictionary<string, string> {
            { "bool", "b" },
            { "int", "i" },
            { "uint", "u" },
            { "dword", "u" },
            { "half", "h" },
            { "float", "f" },
            { "double", "d" },
            { "min16float", "s16f" },
            { "min10float", "s10f" },
            { "min16int", "s16i" },
            { "min12int", "s12i" },
            { "min16uint", "s16u" },
            { "texture", "tx" },
            { "Texture1D", "t1d" },
            { "Texture1DArray", "t1da" },
            { "Texture2D", "t2d" },
            { "Texture2DArray", "t2da" },
            { "Texture3D", "t3d" },
            { "TextureCube", "tc" },
            { "sampler", "sx" },
            { "sampler1D", "s1d" },
            { "sampler2D", "s2d" },
            { "sampler3D", "s3d" },
            { "samplerCUBE", "sc" },
            { "sampler_state", "ss" },
            { "SamplerState", "ss" },
        };
    }
}
