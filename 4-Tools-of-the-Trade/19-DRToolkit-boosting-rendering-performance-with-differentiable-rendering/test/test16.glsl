struct Data
{
   vec3 col;
   float area[4];
} var[5];

Data v;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    var = Data[5](
        Data(vec3(0.5, 0.0, 0.0), float[4](0.0,0.0,0.4,0.5)),
        Data(vec3(0.1, 0.5, 0.0), float[4](0.0,0.5,0.5,0.8)),
        Data(vec3(0.2, 0.3, 0.5), float[4](0.0,0.8,0.7,1.0)),
        Data(vec3(0.3, 0.1, 0.7), float[4](0.5,0.0,0.9,0.5)),
        Data(vec3(0.4, 0.8, 1.0), float[4](0.3,0.3,0.8,0.9))
    );
    v = Data(vec3(0.0), float[4](0.1,0.0,0.0,0.0));
    
    float v2[2] = float[2](1.0,1.0);
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.1 + 0.2*cos(iTime+uv.xyx+vec3(0,2,4));

    for(int i=0;i<5;++i){
        if(uv.x >= var[i].area[0] && uv.x <= var[i].area[2] && 
            uv.y >= var[i].area[1] && uv.y <= var[i].area[3]){
            col += var[i].col;
        }
    }
    // Output to screen
    fragColor = vec4(col,1.0);
}
