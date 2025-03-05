#define DITHER
#define AUTO_MODE
#define DOWN_SCALE 2.0

#define MAX_STEPS 196
#define MIN_DIST 0.002
#define NORMAL_SMOOTHNESS 0.1
#define PI 3.14159265359

#define PALETTE_SIZE 16
#define SUB_PALETTE_SIZE 8

#define RGB(r,g,b) (vec3(r,g,b) / 255.0)

vec3 palette[PALETTE_SIZE];
vec3 subPalette[SUB_PALETTE_SIZE];

//Initalizes the color palette.
void InitPalette()
{
    //16-Color C64 color palette.
	palette = vec3[](
        RGB(  0,  0,  0),
        RGB(255,255,255),
        RGB(152, 75, 67),
        RGB(121,193,200),	
        RGB(155, 81,165),
        RGB(104,174, 92),
        RGB( 62, 49,162),
        RGB(201,214,132),	
        RGB(155,103, 57),
        RGB(106, 84,  0),
        RGB(195,123,117),
        RGB( 85, 85, 85),	
        RGB(138,138,138),
        RGB(163,229,153),
        RGB(138,123,206),
        RGB(173,173,173)
	);
    
    //8-Color metalic-like sub palette.
	subPalette = vec3[](
        palette[ 6],
        palette[11],
        palette[ 4],
        palette[14],
        palette[ 5],
        palette[ 3],
        palette[13],
        palette[ 1]
	);
	
}

//Blends the nearest two palette colors with dithering.
vec3 GetDitheredPalette(float x,vec2 pixel)
{
	float idx = clamp(x,0.0,1.0)*float(SUB_PALETTE_SIZE-1);
	
	vec3 c1 = vec3(0);
	vec3 c2 = vec3(0);
	
	c1 = subPalette[int(idx)];
    c2 = subPalette[int(idx) + 1];
    
    #ifdef DITHER
    	float dith = texture(iChannel0, pixel / iChannelResolution[0].xy).r;
    	float mixAmt = float(fract(idx) > dith);
    #else
    	float mixAmt = fract(idx);
    #endif
    
	return mix(c1,c2,mixAmt);
}

//Returns a 2D rotation matrix for the given angle.
mat2 Rotate(float angle)
{
	return mat2(cos(angle), sin(angle), -sin(angle), cos(angle));   
}

//Distance field functions & operations by iq. (https://iquilezles.org/articles/distfunctions)
float opU( float d1, float d2 )
{
    return min(d1,d2);
}

float opS( float d1, float d2 )
{
    return max(-d1,d2);
}

float opI( float d1, float d2 )
{
    return max(d1,d2);
}

vec3 opRep( vec3 p, vec3 c )
{
    vec3 q = mod(p,c)-0.5*c;
    return q;
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}

float sdCylinder( vec3 p, vec3 c )
{
  return length(p.xz-c.xy)-c.z;
}

//Scene definition/distance function.
float Scene(vec3 pos)
{
    float map = -sdSphere(pos, 24.0);
    
    vec3 rep = opRep(pos - 2.0, vec3(4.0));
    
    map = opU(map, opI(sdBox(pos, vec3(5.5)), sdSphere(rep, 1.0)));
    
    vec3 gSize = vec3(0, 0, 0.25);
    
    float grid = opU(opU(sdCylinder(rep.xyz, gSize), sdCylinder(rep.xzy, gSize)), sdCylinder(rep.zxy, gSize));
     
    grid = opI(sdBox(pos,vec3(4.5)),grid);
    
    map = opU(map, grid);
    
    return map;
}

//Returns the normal of the surface at the given position.
vec3 Normal(vec3 pos)
{
	vec3 offset = vec3(NORMAL_SMOOTHNESS, 0, 0);
    
    vec3 normal = vec3
    (
        Scene(pos - offset.xyz) - Scene(pos + offset.xyz),
        Scene(pos - offset.zxy) - Scene(pos + offset.zxy),
        Scene(pos - offset.yzx) - Scene(pos + offset.yzx)
    );
    
    return normalize(normal);
}

//Marches a ray defined by the origin and direction and returns the hit position.
vec3 RayMarch(vec3 origin,vec3 direction)
{
    float hitDist = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Scene(origin + direction * hitDist);
        
        hitDist += sceneDist;
        
        if(sceneDist < MIN_DIST)
        {
            break;
        }
    }
    
    return origin + direction * hitDist;
}

//Scene shading.
vec3 Shade(vec3 position, vec3 normal, vec3 rayOrigin,vec3 rayDirection,vec2 pixel)
{
    vec3 color = vec3(0);
    
    float ang = iTime * 2.0;
    
    vec3 lightPos = vec3(cos(ang), cos(ang*2.0), sin(ang)) * 2.0;  
    
    //Normal shading
	float shade = 0.4 * max(0.0, dot(normal, normalize(-lightPos)));
    
    //Specular highlight
    shade += 0.6 * max(0.0, dot(-reflect(normalize(position - lightPos), normal), rayDirection));
    
    //Linear falloff
    shade *= (16.0-distance(position, lightPos))/16.0,
    
    //Apply palette
    color = GetDitheredPalette(shade, pixel);

    //color = mix(color, vec3(0.1), step(22.0, length(position)));
    
    return color;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    InitPalette();
    
    vec2 aspect = iResolution.xy / iResolution.y;
    
    fragCoord = floor(fragCoord / DOWN_SCALE) * DOWN_SCALE;
    
	vec2 uv = fragCoord.xy / iResolution.y;
    
    vec2 mouse = iMouse.xy / iResolution.xy - 0.5;
    
    vec2 camAngle = vec2(0);
    
    #ifdef AUTO_MODE
		camAngle.x = PI * (-1.0 / 8.0) * sin(iTime * 0.5);
    	camAngle.y = -iTime;
    #else
        camAngle.x = PI * mouse.y + PI / 2.0;
        camAngle.x += PI / 3.0;

        camAngle.y = 2.0 * PI * -mouse.x;
        camAngle.y += PI;
    #endif
    
    vec3 rayOrigin = vec3(0 , 0, -16.0);
    vec3 rayDirection = normalize(vec3(uv - aspect / 2.0, 1.0));
    
    mat2 rotateX = Rotate(camAngle.x);
    mat2 rotateY = Rotate(camAngle.y);
    
    //Transform ray origin and direction
    rayOrigin.yz *= rotateX;
    rayOrigin.xz *= rotateY;
    rayDirection.yz *= rotateX;
    rayDirection.xz *= rotateY;
    
    vec3 scenePosition = RayMarch(rayOrigin, rayDirection);
    
    vec3 outColor = Shade(scenePosition,Normal(scenePosition), rayOrigin, rayDirection, fragCoord / DOWN_SCALE);
    
    //Palette preview
    if(uv.x < 0.05) 
    {
        outColor = GetDitheredPalette(uv.y, fragCoord / DOWN_SCALE);
    }
    
	fragColor = vec4(outColor, 1.0);
}