//gen code for numpy, run BatchCommand.exe gen_hlsl_lib_numpy_swizzle.dsl > hlsl_lib_numpy_swizzle.py
//BatchCommand.exe compiled from https://github.com/dreamanlan/BatchCommand

script(main)
{
    $x = ["", "x"];
    $xy = ["", "x", "y"];
    $xyz = ["", "x", "y", "z"];
    $xyzw = ["", "x", "y", "z", "w"];
    echo("#----------------------------------------");
    echo("# these code generated from gen_hlsl_lib_numpy_swizzle.dsl");
    echo("#---begin---");
    echo();
    call("writeListDef", $xyzw);
    echo();
    call("writeGet1", $x);
    call("writeGet234", 2, $xy);
    call("writeGet234", 3, $xyz);
    call("writeGet234", 4, $xyzw);
    call("writeSet1", $x);
    call("writeSet234", 2, $xy);
    call("writeSet234", 3, $xyz);
    call("writeSet234", 4, $xyzw);
    call("writeArrGet1", $x);
    call("writeArrGet234", 2, $xy);
    call("writeArrGet234", 3, $xyz);
    call("writeArrGet234", 4, $xyzw);
    call("writeArrSet1", $x);
    call("writeArrSet234", 2, $xy);
    call("writeArrSet234", 3, $xyz);
    call("writeArrSet234", 4, $xyzw);
    call("writeBroadcastSet1", $x);
    call("writeBroadcastSet234", 2, $xy);
    call("writeBroadcastSet234", 3, $xyz);
    call("writeBroadcastSet234", 4, $xyzw);
    echo("#---end---");
    echo("#----------------------------------------");
    return(0);
};
script(tag2index)args($m)
{
    if($m=="x"){
        return 0;
    }
    elseif($m=="y"){
        return 1;
    }
    elseif($m=="z"){
        return 2;
    }
    elseif($m=="w"){
        return 3;
    };
    return -1;
};
script(buildvlist)args($v, $num)
{
    $sb = newstringbuilder();
    $prestr = "";
    loop($num){
        appendformat($sb, "{0}{1}", $prestr, $v);
        $prestr=", ";
    };
    return stringbuildertostring($sb);
};
script(buildmindexdef)args($m, $m1, $m2, $m3, $m4)
{
    $sb = newstringbuilder();
    $prestr = "";
    $ms = [$m1, $m2, $m3, $m4];
    appendformat($sb, "g_{0}_index", $m);
    appendformat($sb, "{0}", " = torch.asarray([");
    looplist($ms){
        $ix = tag2index($$);
        if($ix>=0){
            appendformat($sb, "{0}{1}", $prestr, $ix);
            $prestr=", ";
        };
    };
    appendformat($sb, "{0}", "], device=device)");
    return stringbuildertostring($sb);
};
script(buildmlist)args($v, $m1, $m2, $m3, $m4)
{
    $sb = newstringbuilder();
    $prestr = "";
    $ms = [$m1, $m2, $m3, $m4];
    looplist($ms){
        $ix = tag2index($$);
        if($ix>=0){
            appendformat($sb, "{0}{1}[{2}]", $prestr, $v, $ix);
            $prestr=", ";
        };
    };
    return stringbuildertostring($sb);
};
script(buildamlist)args($v, $m1, $m2, $m3, $m4)
{
    $sb = newstringbuilder();
    $prestr = "";
    $ms = [$m1, $m2, $m3, $m4];
    looplist($ms){
        $ix = tag2index($$);
        if($ix>=0){
            appendformat($sb, "{0}{1}[..., {2}]", $prestr, $v, $ix);
            $prestr=", ";
        };
    };
    return stringbuildertostring($sb);
};
script(writeListDef)args($xyzw)
{
    $handled = {};
    looplist($xyzw){
        $1 = $$;
    looplist($xyzw){
        $2 = $$;
    looplist($xyzw){
        $3 = $$;
    looplist($xyzw){
        $4 = $$;

        $tag = $1+$2+$3+$4;
        $len = $tag.Length;
        $find = hashtableget($handled, $tag, false);
        hashtableset($handled, $tag, true);
        if(!$find && $len>0){
            if($len==1){
                echo("{0}.squeeze(0)", buildmindexdef($tag, $1, $2, $3, $4));
            }
            else{
                echo(buildmindexdef($tag, $1, $2, $3, $4));
            };
        };
    };
    };
    };
    };
};
script(writeGet1)args($x)
{
    $handled = {};
    looplist($x){
        $1 = $$;
    looplist($x){
        $2 = $$;
    looplist($x){
        $3 = $$;
    looplist($x){
        $4 = $$;

        $tag = $1+$2+$3+$4;
        $len = $tag.Length;
        $find = hashtableget($handled, $tag, false);
        hashtableset($handled, $tag, true);
        if(!$find && $len>0){
            if($len==1){
                writeblock
                {:     
def swizzle_n_{% $tag %}(v):
    return v
                :};
            }
            else{
                writeblock
                {:
def swizzle_n_{% $tag %}(v):
    return torch.asarray([{% buildvlist("v", $len) %}], device=device)
                :};
            };
        };
    };
    };
    };
    };
};
script(writeSet1)
{
    writeblock
    {:     
def swizzle_set_n_x(v, val):
    raise
    :};
};
script(writeGet234)args($n, $xyzw)
{
    $handled = {};
    looplist($xyzw){
        $1 = $$;
    looplist($xyzw){
        $2 = $$;
    looplist($xyzw){
        $3 = $$;
    looplist($xyzw){
        $4 = $$;

        $tag = $1+$2+$3+$4;
        $len = $tag.Length;
        $find = hashtableget($handled, $tag, false);
        hashtableset($handled, $tag, true);
        if(!$find && $len>0){
            if($len==1){
                writeblock
                {:
def swizzle_n{% $n %}_{% $tag %}(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_{% $tag %}_index).squeeze(0)
                :};
            }
            else{
                writeblock
                {:
def swizzle_n{% $n %}_{% $tag %}(v):
    v = get_vm_gpu_tensor(v)
    return v.index_select(0, g_{% $tag %}_index)
                :};
            };
            /*
            if($len==1){
                writeblock
                {:
def swizzle_n{% $n %}_{% $tag %}(v):
    return {% buildmlist("v", $1, $2, $3, $4) %}
                :};
            }
            else{
                writeblock
                {:
def swizzle_n{% $n %}_{% $tag %}(v):
    return torch.asarray([{% buildmlist("v", $1, $2, $3, $4) %}], device=device)
                :};
            };
            */
        };
    };
    };
    };
    };
};
script(writeSet234)args($n, $xyzw)
{
    $handled = {};
    looplist($xyzw){
        $1 = $$;
    looplist($xyzw){
        $2 = $$;
    looplist($xyzw){
        $3 = $$;
    looplist($xyzw){
        $4 = $$;
        
        $counts = {};
        hashtableset($counts, $1, hashtableget($counts, $1, 0) + 1);
        hashtableset($counts, $2, hashtableget($counts, $2, 0) + 1);
        hashtableset($counts, $3, hashtableget($counts, $3, 0) + 1);
        hashtableset($counts, $4, hashtableget($counts, $4, 0) + 1);
        $ctx = hashtableget($counts, "x", 0);
        $cty = hashtableget($counts, "y", 0);
        $ctz = hashtableget($counts, "z", 0);
        $ctw = hashtableget($counts, "w", 0);
        if($ctx<=1 && $cty<=1 && $ctz<=1 && $ctw<=1){
            $tag = $1+$2+$3+$4;
            $len = $tag.Length;
            $find = hashtableget($handled, $tag, false);
            hashtableset($handled, $tag, true);
            if(!$find && $len>0){
                writeblock
                {:
def swizzle_set_n{% $n %}_{% $tag %}(v, val):
                :};
                $ms = [$1, $2, $3, $4];
                $six = 0;
                looplist($ms){
                    $ix = tag2index($$);
                    if($ix>=0){
                        if($len==1){
                            echo("    v[{0}] = val", $ix);
                        }
                        else{
                            echo("    v[{0}] = val[{1}]", $ix, $six);
                        };
                        $six = $six + 1;
                    };
                };
                echo("    return val");
            };
        };
    };
    };
    };
    };
};
script(writeArrGet1)args($x)
{
    $handled = {};
    looplist($x){
        $1 = $$;
    looplist($x){
        $2 = $$;
    looplist($x){
        $3 = $$;
    looplist($x){
        $4 = $$;

        $tag = $1+$2+$3+$4;
        $len = $tag.Length;
        $find = hashtableget($handled, $tag, false);
        hashtableset($handled, $tag, true);
        if(!$find && $len>0){
            if($len==1){
                writeblock
                {:     
def swizzle_t_n_{% $tag %}(v):
    return v
                :};
            }
            else{
                writeblock
                {:
def swizzle_t_n_{% $tag %}(v):
    return v.unsqueeze(1).index_select(1, g_{% $tag %}_index)
                :};
                /*
                writeblock
                {:
def swizzle_t_n_{% $tag %}(v):
    return torch.column_stack(({% buildvlist("v", $len) %}))
                :};
                */
            };
        };
    };
    };
    };
    };
};
script(writeArrSet1)
{
    writeblock
    {:     
def swizzle_set_t_n_x(v, val):
    v.copy_(val)
    return val
    :};
};
script(writeArrGet234)args($n, $xyzw)
{
    $handled = {};
    looplist($xyzw){
        $1 = $$;
    looplist($xyzw){
        $2 = $$;
    looplist($xyzw){
        $3 = $$;
    looplist($xyzw){
        $4 = $$;

        $tag = $1+$2+$3+$4;
        $len = $tag.Length;
        $find = hashtableget($handled, $tag, false);
        hashtableset($handled, $tag, true);
        if(!$find && $len>0){
            if($len==1){
                writeblock
                {:
def swizzle_t_n{% $n %}_{% $tag %}(v):
    return torch.clone({% buildamlist("v", $1, $2, $3, $4) %})
                :};
            }
            else{
                writeblock
                {:
def swizzle_t_n{% $n %}_{% $tag %}(v):
    return v.index_select(1, g_{% $tag %}_index)
                :};
                /*
                writeblock
                {:
def swizzle_t_n{% $n %}_{% $tag %}(v):
    return torch.column_stack(({% buildamlist("v", $1, $2, $3, $4) %}))
                :};
                */
            };
        };
    };
    };
    };
    };
};
script(writeArrSet234)args($n, $xyzw)
{
    $handled = {};
    looplist($xyzw){
        $1 = $$;
    looplist($xyzw){
        $2 = $$;
    looplist($xyzw){
        $3 = $$;
    looplist($xyzw){
        $4 = $$;
        
        $counts = {};
        hashtableset($counts, $1, hashtableget($counts, $1, 0) + 1);
        hashtableset($counts, $2, hashtableget($counts, $2, 0) + 1);
        hashtableset($counts, $3, hashtableget($counts, $3, 0) + 1);
        hashtableset($counts, $4, hashtableget($counts, $4, 0) + 1);
        $ctx = hashtableget($counts, "x", 0);
        $cty = hashtableget($counts, "y", 0);
        $ctz = hashtableget($counts, "z", 0);
        $ctw = hashtableget($counts, "w", 0);
        if($ctx<=1 && $cty<=1 && $ctz<=1 && $ctw<=1){
            $tag = $1+$2+$3+$4;
            $len = $tag.Length;
            $find = hashtableget($handled, $tag, false);
            hashtableset($handled, $tag, true);
            if(!$find && $len>0){
                writeblock
                {:
def swizzle_set_t_n{% $n %}_{% $tag %}(v, val):
                :};
                $ms = [$1, $2, $3, $4];
                $six = 0;
                looplist($ms){
                    $ix = tag2index($$);
                    if($ix>=0){
                        if($len==1){
                            echo("    v[..., {0}] = val", $ix);
                        }
                        else{
                            echo("    v[..., {0}] = val[..., {1}]", $ix, $six);
                        };
                        $six = $six + 1;
                    };
                };
                echo("    return val");
            };
        };
    };
    };
    };
    };
};
script(writeBroadcastSet1)
{
    writeblock
    {:     
def swizzle_set_and_broadcast_n_x(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n_x(v, val), v
    :};
};
script(writeBroadcastSet234)args($n, $xyzw)
{
    $handled = {};
    looplist($xyzw){
        $1 = $$;
    looplist($xyzw){
        $2 = $$;
    looplist($xyzw){
        $3 = $$;
    looplist($xyzw){
        $4 = $$;
        
        $counts = {};
        hashtableset($counts, $1, hashtableget($counts, $1, 0) + 1);
        hashtableset($counts, $2, hashtableget($counts, $2, 0) + 1);
        hashtableset($counts, $3, hashtableget($counts, $3, 0) + 1);
        hashtableset($counts, $4, hashtableget($counts, $4, 0) + 1);
        $ctx = hashtableget($counts, "x", 0);
        $cty = hashtableget($counts, "y", 0);
        $ctz = hashtableget($counts, "z", 0);
        $ctw = hashtableget($counts, "w", 0);
        if($ctx<=1 && $cty<=1 && $ctz<=1 && $ctw<=1){
            $tag = $1+$2+$3+$4;
            $len = $tag.Length;
            $find = hashtableget($handled, $tag, false);
            hashtableset($handled, $tag, true);
            if(!$find && $len>0){
                writeblock
                {:
def swizzle_set_and_broadcast_n{% $n %}_{% $tag %}(v, val):
    v = torch.tile(v, (len(val), 1))
    return swizzle_set_t_n{% $n %}_{% $tag %}(v, val), v
                :};
            };
        };
    };
    };
    };
    };
};