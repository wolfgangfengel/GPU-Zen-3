// The MIT License
// Copyright © 2019 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Computing the exact distance to a generic (non symmetric) ellipsoid
// requires solving a sixth degree equation, which can be difficult.
// Approximating the distance is easier though. This shaders shows one
// such approximation that produces better results than the naive
// distance bound. More info here:
//
// https://iquilezles.org/articles/ellipsoids
//
// Left, naive ellipsoid distance approximation (single square root)
// Right, improved approximation (two square roots).
//
// Note how the improved approximation produces a more accurate intersection
// for the same number of raymarching steps (specially noticeable in the first
// frame of the animation). Note also how the penumbra shadow estimation works
// best with since since it has a more eucliden distance as input.
//
// The technique is based on dividing the bad approximation's distance estimation
// by the length of its gradient to get a first order approximation to the true
// distance (see https://iquilezles.org/articles/distance)


// List of other 3D SDFs: https://www.shadertoy.com/playlist/43cXRl
//
// and https://iquilezles.org/articles/distfunctions


#define AA 2   // make this 3 is you have a fast computer

//------------------------------------------------------------------

// generic ellipsoid - simple but bad approximated distance
float sdEllipsoid_Bad( in vec3 p, in vec3 r ) 
{
    return (length(p/r)-1.0)*min(min(r.x,r.y),r.z);
}


// generic ellipsoid - improved approximated distance
float sdEllipsoid( in vec3 p, in vec3 r ) 
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

//------------------------------------------------------------------

vec2 map( in vec3 p, int id )
{
    // ellipsoid
    float d1 = (id==0) ? sdEllipsoid_Bad( p, vec3(0.18,0.3,0.02) ) :
                         sdEllipsoid(     p, vec3(0.18,0.3,0.02) );

    // plane
    float d2 = p.y+0.3;
    
    return (d1<d2) ? vec2(d1,1.0) : vec2(d2,2.0);
}

vec2 castRay( in vec3 ro, in vec3 rd, int id )
{
    float m = 0.0;
    float t = 0.0;
    const float tmax = 100.0;
    for( int i=0; i<100; i++ )
    {
	    vec2 h = map( ro+rd*t, id );
        if( h.x<0.001 ) break;
        if( t>=tmax ) break;
        m = h.y;
        t += h.x;        
    }

    return (t<tmax) ? vec2(t,m) : vec2(0.0);
}


float calcSoftshadow( in vec3 ro, in vec3 rd, in int id)
{
	float res = 1.0;
    float t = 0.01;
    for( int i=0; i<256; i++ )
    {
		float h = map( ro + rd*t, id ).x;
        res = min( res, smoothstep(0.0,1.0,8.0*h/t ));
        t += clamp( h, 0.005, 0.02 );
        if( res<0.001 || t>5.0 ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

vec3 calcNormal( in vec3 pos, in int id )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy, id ).x + 
					  e.yyx*map( pos + e.yyx, id ).x + 
					  e.yxy*map( pos + e.yxy, id ).x + 
					  e.xxx*map( pos + e.xxx, id ).x );
}

float calcAO( in vec3 pos, in vec3 nor, in int id )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos, id).x;
        occ += (hr-dd)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 2.0*occ, 0.0, 1.0 );    
}
 
// https://iquilezles.org/articles/checkerfiltering
float checkersGradBox( in vec2 p )
{
    // filter kernel
    //vec2 w = fwidth(p) + 0.001;
    vec2 w = vec2(0.001, 0.001);
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;                  
}

vec3 render( in vec3 ro, in vec3 rd, int id )
{ 
    vec3 col = vec3(0.0);
    
    vec2  res = castRay(ro,rd, id);

    if( res.y>0.5 )
    {
        float t   = res.x;
        vec3  pos = ro + t*rd;
        vec3  nor;
        float occ;

        // material        
        if( res.y>1.5 )
        {
        	nor = vec3(0.0,1.0,0.0);
            col = 0.05*vec3(1.0);
            col *= 0.7+0.3*checkersGradBox( pos.xz*2.0 );
            occ = 1.0;

        }
        else
        {
            nor = calcNormal( pos, id );
            occ = 0.5+0.5*nor.y;
            col = vec3(0.2);
        }

        // lighting
        occ *= calcAO( pos, nor, id );

        vec3  lig = normalize( vec3(-0.5, 1.9, 0.8) );
        vec3  hal = normalize( lig-rd );
        float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);

        float sha = calcSoftshadow( pos, lig, id );
        sha = sha*sha;

        float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),32.0)*
                    dif * sha *
                    (0.04 + 0.96*pow( clamp(1.0+dot(hal,rd),0.0,1.0), 5.0 ));

        //vec3 lin = vec3(0.0);
        //lin += 2.00*dif*vec3(3.30,2.50,2.00)*sha;
        //lin += 0.50*amb*vec3(0.30,0.60,1.50)*occ;
        //lin += 0.30*bac*vec3(0.40,0.30,0.25)*occ;
        //col = col*lin;
        //col += 2.00*spe*vec3(3.30,2.50,2.00);
        
        col *= 5.0;
        col *= vec3(0.2,0.3,0.4)*amb*occ + 1.6*vec3(1.0,0.9,0.75)*dif*sha;
        col += vec3(2.8,2.2,1.8)*spe*3.0;            


        
        //col = mix( col, vec3(0.1), 1.0-exp(-0.03*t) );
    }
	return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // camera	
    vec3 ro = vec3( 1.0*cos(0.2*iTime), 0.12, 1.0*sin(0.2*iTime) );
    vec3 ta = vec3( 0.0, 0.0, 0.0 );
    // camera-to-world transformation
    vec3 cw = normalize(ta-ro);
    vec3 cu = normalize( cross(cw,vec3(0.0, 1.0,0.0)) );
    vec3 cv =          ( cross(cu,cw) );

    // scene selection
    int id = (fragCoord.x>iResolution.x/2.0) ? 1 : 0;

    // render
    vec3 tot = vec3(0.0);
	#if AA>1
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ )
    {
        // pixel coordinates
        vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
        vec2 fc = o + vec2( mod(fragCoord.x,iResolution.x/2.0), fragCoord.y);
		#else    
        vec2 fc = vec2( mod(fragCoord.x,iResolution.x/2.0), fragCoord.y);
		#endif
        vec2 p = (-vec2(iResolution.x/2.0,iResolution.y) + 2.0*fc)/iResolution.y;

        // ray direction
        vec3 rd = normalize( p.x*cu + p.y*cv + 2.0*cw );

        // render	
        vec3 col = render( ro, rd, id );

		// gamma
        col = pow( col, vec3(0.4545) );

        tot += col;
	#if AA>1
    }
    tot /= float(AA*AA);
	#endif

    // separator    
	tot *= smoothstep( 1.0, 2.5, abs(fragCoord.x-iResolution.x/2.0) );
    
    fragColor = vec4( tot, 1.0 );
}