tex2d0("shaderlib/rgbanoise256.png");
tex2d1("shaderlib/abstract1.jpg");
tex2d2("shaderlib/lichen.jpg");
tex2d3("shaderlib/font.png");
entry(mainImage);
resolution_on_full_vec(210, 118, 1);
resolution_on_gpu_full_vec(320, 240, 1);

/*
addbuffer(bufferA){
    entry(fillBufferA);
    tex2d0("shaderlib/stars.jpg");
    tex2d1("shaderlib/pebbles.png");
    tex2d2("shaderlib/organic1.jpg");
    texcube3("shaderlib/texcube5.png");
};
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
tex2d0(bufferA);
tex2d1(bufferD);
tex2d2(bufferB);
tex2d3(bufferC);
*/
