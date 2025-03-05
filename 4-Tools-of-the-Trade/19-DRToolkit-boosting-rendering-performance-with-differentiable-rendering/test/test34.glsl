// Stateless smoke by nimitz 2022 (twitter: @stormoid)
// https://www.shadertoy.com/view/WtdfR8
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
// Contact the author for other licensing options

mat3 rot_x(float a){float sa = sin(a); float ca = cos(a); return mat3(1.,.0,.0,    .0,ca,sa,   .0,-sa,ca);}
mat3 rot_y(float a){float sa = sin(a); float ca = cos(a); return mat3(ca,.0,sa,    .0,1.,.0,   -sa,.0,ca);}
mat3 rot_z(float a){float sa = sin(a); float ca = cos(a); return mat3(ca,sa,.0,    -sa,ca,.0,  .0,.0,1.);}

mat2 mm2(in float a){float c = cos(a), s = sin(a);return mat2(c,s,-s,c);}
const mat4 m4 = mat4(-0.164, -0.223, -0.455, 0.846, 
                     -0.714, 0.576, 0.344, 0.198, 
                     -0.526, -0.782, 0.301, -0.146,
                     -0.431, 0.084, -0.764, -0.473)*1.93;

float map(vec3 p)
{
    float d = 0.;
    float lp = length(p.xz);
    p.xz *= mm2(p.y*.05 - iTime*0.015);
    p.y *= .58;
    vec4 q = vec4(p, iTime*0.4 - p.y*.55);
    q.y -= iTime*0.16;
    float cl = dot(p.xz,p.xz);
    vec3 bp = p;
    q *= .85;
    float z = 1.15;
    float trk = 1.;
    
    for(int i = 0; i < 6; i++)
    {
        d += .75-abs(dot(cos(q*.85), sin(q.yzwx)) - .9)*z;   
        z *= 0.65;
        q *= m4;
        q += (sin(q.zxwy*trk) + (cos(q*1.5 - 2.5)*0.3))*0.3;
        trk *= 1.4;
    }
    return d*1.2 - cl*0.2 + .0;
}

vec4 render( in vec3 ro, in vec3 rd )
{
    const vec3 lpos = vec3(0.5, 1, 1.);
	vec4 rez = vec4(0);
	float t = 6.5;
	for(int i=0; i<20; i++)
	{
		if(rez.a > 0.97 || t > 18.)break;

		vec3 pos = ro + t*rd;
        float dn = map(pos);
		float den = clamp(dn, .0, 1.);
        
        if (dn < 0.0) 
        {
            t += .2;
            continue;
        }
        vec4 col = vec4(1.3*vec3(0.105,0.105,0.11)*smoothstep(-12.,5., pos.y),0.08)*den;
        float dif =  clamp((dn - map(pos*vec3(1.2) + .3))/8., 0.01, 1. );
        float dif2 =  clamp((dn - map(pos*vec3(1.1) + .7))/6., 0.01, 1. );
        col.xyz *= vec3(0.01,0.01,0.01) + vec3(0.14,0.12,0.1)*dif + vec3(0.15,0.12,0.1)*dif2;
        
        rez = rez + col*(1. - rez.a);
        t += clamp(0.12 - den*.1, 0.05, 0.15);
        
	}
	return clamp(rez, 0.0, 1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{	
	vec2 q = fragCoord.xy / iResolution.xy;
    vec2 p = q - 0.5;
	p.x*=iResolution.x/iResolution.y;
	vec2 mo = iMouse.xy / iResolution.xy-.5;
    mo = (mo==vec2(-.5)) ? vec2(0.12,0.15) : mo;
	mo.x *= iResolution.x/iResolution.y;
    mo*=4.14;
	mo.y = clamp(mo.y*0.6-.5,-4. ,.15 );
	
    vec3 ro = vec3(0.,-0.0,12.);
    vec3 rd = normalize(vec3(p,-1.5));
    
    mat3 cam = rot_x(-mo.y)*rot_y(-mo.x);
	rd *= cam;
    ro *= cam;
    
    vec4 scn = render(ro, rd);
    vec3 col = vec3(0.1, 0.1, 0.11)*smoothstep(-1.,1.,rd.y)*7.;
    
    col = col*(1.0-scn.w) + scn.xyz;

    col = pow(col, vec3(0.45));
    col *= pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.1)*0.9+0.1; //Vign
	fragColor = vec4( col, 1.0 );
}
