tex2d0("shaderlib/wood.jpg");
tex2d1("shaderlib/rgbanoise256.png");
texcube2("shaderlib/texcube0.jpg");
tex2d3("shaderlib/font.png");
entry(mainImage);
resolution_on_full_vec(160, 120, 1);
resolution_on_gpu_full_vec(640, 480, 1);
addbuffer(bufferA){
    entry(fillChannel0);
};
addbuffer(bufferB){
    entry(fillChannel1);
    tex2d0(bufferB);
    tex2d1(bufferA);
};
tex2d0(bufferA);
tex2d1(bufferB);
