newoption {
    trigger = "cuda",
    description = "enable cuda environment"
}

workspace "GPUZen3Onesweep"
    location "build"
    configurations { "Debug", "Release" }

architecture "x86_64"

project "Onesweep"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    cppdialect "C++17"

    -- Src
    files { 
        "onesweep.cpp", 
        "json.hpp"
     }

    files { "radixsort/*" }
    includedirs { "radixsort" }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    
    files { "libs/orochi/Orochi/OrochiUtils.h" }
    files { "libs/orochi/Orochi/OrochiUtils.cpp" }

    if _OPTIONS["cuda"] then
        defines {"OROCHI_ENABLE_CUEW"}
        includedirs {"$(CUDA_PATH)/include"}
    end

    links { "version" }

    -- UTF8
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("onesweep_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("onesweep")
        optimize "Full"
    filter{}
