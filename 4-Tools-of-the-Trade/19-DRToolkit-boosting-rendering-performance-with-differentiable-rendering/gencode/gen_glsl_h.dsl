//gen code for glsl.h, run BatchCommand.exe gen_glsl_h.dsl > glsl_autogen.h, then include glsl_autogen in glsl.h
//BatchCommand.exe compiled from https://github.com/dreamanlan/BatchCommand

script(main)
{
    $glslVec2Types = ["vec2", "dvec2", "ivec2", "uvec2", "bvec2"];
    $glslVec3Types = ["vec3", "dvec3", "ivec3", "uvec3", "bvec3"];
    $glslVec4Types = ["vec4", "dvec4", "ivec4", "uvec4", "bvec4"];
    $glslVecTypePrefixes = ["vec", "dvec", "ivec", "uvec", "bvec"];
    echo("//----------------------------------------");
    echo("// these code generated from gen_glsl_h.dsl");
    echo("//---begin---");
    call("writeCast", $glslVec2Types);
    call("writeCast", $glslVec3Types);
    call("writeCast", $glslVec4Types);
    echo();
    call("writeComp", $glslVecTypePrefixes);
    echo("//---end---");
    echo("//----------------------------------------");
    return(0);
};
script(writeCast)args($typeNames)
{
    looplist($typeNames){
        $typeName1 = $$;
        looplist($typeNames){
            $typeName2 = $$;
            if($typeName1 != $typeName2){
                writeblock
                {:
{% $typeName1 %} glsl_{% $typeName1 %}({% $typeName2 %} arg)
{
    return {% $typeName1 %}(arg);
}            
                :};
            };
        };
    };
};
script(writeComp)args($typeNames)
{
    looplist($typeNames){
        $typeName = $$;

        writeblock
        {:
ivec2 greaterThan({% $typeName %}2 x, {% $typeName %}2 y)
{
    return ivec2(1, 1) - ivec2(step(x, y));
}
ivec3 greaterThan({% $typeName %}3 x, {% $typeName %}3 y)
{
    return ivec3(1, 1, 1) - ivec3(step(x, y));
}
ivec4 greaterThan({% $typeName %}4 x, {% $typeName %}4 y)
{
    return ivec4(1, 1, 1, 1) - ivec4(step(x, y));
}
ivec2 lessThan({% $typeName %}2 x, {% $typeName %}2 y)
{
    return ivec2(1.0, 1.0) - ivec2(step(y, x));
}
ivec3 lessThan({% $typeName %}3 x, {% $typeName %}3 y)
{
    return ivec3(1.0, 1.0, 1.0) - ivec3(step(y, x));
}
ivec4 lessThan({% $typeName %}4 x, {% $typeName %}4 y)
{
    return ivec4(1.0, 1.0, 1.0, 1.0) - ivec4(step(y, x));
}
ivec2 greaterThanEqual({% $typeName %}2 x, {% $typeName %}2 y)
{
    return ivec2(step(y, x));
}
ivec3 greaterThanEqual({% $typeName %}3 x, {% $typeName %}3 y)
{
    return ivec3(step(y, x));
}
ivec4 greaterThanEqual({% $typeName %}4 x, {% $typeName %}4 y)
{
    return ivec4(step(y, x));
}
ivec2 lessThanEqual({% $typeName %}2 x, {% $typeName %}2 y)
{
    return ivec2(step(x, y));
}
ivec3 lessThanEqual({% $typeName %}3 x, {% $typeName %}3 y)
{
    return ivec3(step(x, y));
}
ivec4 lessThanEqual({% $typeName %}4 x, {% $typeName %}4 y)
{
    return ivec4(step(x, y));
}
ivec2 equal({% $typeName %}2 x, {% $typeName %}2 y)
{
    return ivec2(step(x, y) * step(y, x));
}
ivec3 equal({% $typeName %}3 x, {% $typeName %}3 y)
{
    return ivec3(step(y, x) * step(y, x));
}
ivec4 equal({% $typeName %}4 x, {% $typeName %}4 y)
{
    return ivec4(step(y, x) * step(y, x));
}
ivec2 notEqual({% $typeName %}2 x, {% $typeName %}2 y)
{
    return (ivec2(1, 1) - ivec2(step(x, y))) * (ivec2(1, 1) - ivec2(step(y, x)));
}
ivec3 notEqual({% $typeName %}3 x, {% $typeName %}3 y)
{
    return (ivec3(1, 1, 1) - ivec3(step(x, y))) * (ivec3(1, 1, 1) - ivec3(step(y, x)));
}
ivec4 notEqual({% $typeName %}4 x, {% $typeName %}4 y)
{
    return (ivec4(1, 1, 1, 1) - ivec4(step(x, y))) * (ivec4(1, 1, 1, 1) - ivec4(step(y, x)));
}
        :};        
    };
};