tex2d0("shaderlib/wood.jpg");
tex2d1("shaderlib/rgbanoise256.png");
texcube2("shaderlib/texcube0.jpg");
tex2d3("shaderlib/font.png");
entry(mainImage);
resolution_on_full_vec(32, 24, 1);
resolution_on_gpu_full_vec(64, 48, 1);
addbuffer(bufferA){
    entry(fillChannel0);
};
addbuffer(bufferB){
    entry(fillChannel1);
    texcube0("shaderlib/texcube0.jpg");
    tex2d1("shaderlib/wood.jpg");
};
tex2d1("bufferB");