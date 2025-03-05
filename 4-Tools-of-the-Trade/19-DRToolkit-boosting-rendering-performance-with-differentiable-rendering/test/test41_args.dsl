tex2d0("shaderlib/rocktiles.jpg");
tex2d1("shaderlib/stars.jpg");
tex2d2("shaderlib/stars.jpg");
tex2d3("shaderlib/font.png");
entry(mainImage);
resolution_on_full_vec(160, 120, 1);
resolution_on_gpu_full_vec(160, 120, 1);

addbuffer(bufferA){
    entry(fillBufferA);
    tex2d0(bufferA);
    texcube1("shaderlib/texcube0.jpg");
    tex2d2("shaderlib/organic4.jpg");
    tex2d3("shaderlib/lichen.jpg");
};
/*
addbuffer(bufferB){
    entry(fillBufferB);
    tex2d0(bufferA);
};

addbuffer(bufferC){
    entry(fillBufferC);
    tex2d0(bufferB);
};

addbuffer(bufferD){
    entry(fillBufferD);
    tex2d0(bufferA);
    tex2d1(bufferD);
};
*/
tex2d0(bufferA);
//tex2d1(bufferD);
//tex2d2(bufferB);
//tex2d3(bufferC);
