// Created by inigo quilez - iq/2014
//   https://www.youtube.com/c/InigoQuilez
//   https://iquilezles.org/
// I share this piece (art and code) here in Shadertoy and through its Public API, only for educational purposes. 
// You cannot use, sell, share or host this piece or modifications of it as part of your own commercial or non-commercial product, website or project.
// You can share a link to it or an unmodified screenshot of it provided you attribute "by Inigo Quilez, @iquilezles and iquilezles.org". 
// If you are a teacher, lecturer, educator or similar and these conditions are too restrictive for your needs, please contact me and we'll work it out.

//-----------------------------------------------------------------------

float hash( float n ) { return fract(sin(n)*43758.5453); }

float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
               mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);

}

float fbm( vec2 p, float sp)
{
    const mat2 m = mat2( 0.8, 0.6, -0.6, 0.8 );

	float f = 0.0;
    f += 0.5000*noise( p ); p = m*p*2.02; p.y -= sp*iTime;
    f += 0.2500*noise( p ); p = m*p*2.03; p.x -= sp*iTime;
    f += 0.1250*noise( p ); p = m*p*2.01; p.x += sp*iTime;
    f += 0.0625*noise( p );
    return f/0.9375;
}

//-----------------------------------------------------------------------

float box( in vec2 p, in float x, in float y, in float dirx, in float diry, in float radx, in float rady )
{
	vec2  q = p - vec2(x,y);
	float u = dot( q, vec2(dirx,diry) );
	float v = dot( q, vec2(diry,dirx)*vec2(-1.0,1.0) );
	vec2  d = abs(vec2(u,v)) - vec2(radx,rady);
	return max(d.x,d.y);
}

float det( vec2 a, vec2 b ) { return a.x*b.y-b.x*a.y; }
vec3 sdBezier( vec2 b0, vec2 b1, vec2 b2, in vec2 p ) 
{
  b0 -= p;
  b1 -= p;
  b2 -= p;
	
  float a =     det(b0,b2);
  float b = 2.0*det(b1,b0);
  float d = 2.0*det(b2,b1);
  float f = b*d - a*a;
  vec2  d21 = b2-b1;
  vec2  d10 = b1-b0;
  vec2  d20 = b2-b0;
  vec2  gf = 2.0*(b*d21+d*d10+a*d20); gf = vec2(gf.y,-gf.x);
  vec2  pp = -f*gf/dot(gf,gf);
  vec2  d0p = b0-pp;
  float ap = det(d0p,d20);
  float bp = 2.0*det(d10,d0p);
  float t = clamp( (ap+bp)/(2.0*a+b+d), 0.0 ,1.0 );
  return vec3( mix(mix(b0,b1,t), mix(b1,b2,t),t), t );
}

float sdTriangle( in vec2 p0, in vec2 p1, in vec2 p2, in vec2 p )
{
	vec2 e0 = p1 - p0;
	vec2 e1 = p2 - p1;
	vec2 e2 = p0 - p2;

	vec2 v0 = p - p0;
	vec2 v1 = p - p1;
	vec2 v2 = p - p2;

	vec2 pq0 = v0 - e0*clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 );
	vec2 pq1 = v1 - e1*clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 );
	vec2 pq2 = v2 - e2*clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 );
    
    vec2 d = min( min( vec2( dot( pq0, pq0 ), v0.x*e0.y-v0.y*e0.x ),
                       vec2( dot( pq1, pq1 ), v1.x*e1.y-v1.y*e1.x )),
                       vec2( dot( pq2, pq2 ), v2.x*e2.y-v2.y*e2.x ));

	return -sqrt(d.x)*sign(d.y);
}

vec2 sdSegment( vec2 a, vec2 b, vec2 p )
{
	vec2 pa = p - a;
	vec2 ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	
	return vec2( length( pa - ba*h ), h );
}
//------------------------------------------------------

float fillRectangle( in vec2 p, in float x, in float y, in float dirx, in float diry, in float radx, in float rady )
{
	float d = box(p,x,y,dirx,diry,radx,rady);
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(-w, w, d);
}

float fillTriangle( in vec2 p, float x1, float y1, float x2, float y2, float x3, float y3 )
{ 
    float d = sdTriangle( vec2(x1,y1), vec2(x2,y2), vec2(x3,y3), p );
	
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(0.0, w, d);
}

float fillQuad( in vec2 p, float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4 )
{ 
    float d1 = sdTriangle( vec2(x1,y1), vec2(x2,y2), vec2(x3,y3), p );
    float d2 = sdTriangle( vec2(x1,y1), vec2(x3,y3), vec2(x4,y4), p );
    float d = min( d1, d2 );
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(0.0, w, d);
}

float fillBezier( in vec2 p, float x1, float y1, float x2, float y2, float x3, float y3, float th1, float th2 )
{ 
	vec3 be = sdBezier( vec2(x1,y1), vec2(x2,y2), vec2(x3,y3), p );
    float d = length(be.xy) - mix(th1,th2,be.z);
	
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(-w, w, d);
}

float fillLine( in vec2 p, float x1, float y1, float x2, float y2, float th1, float th2 )
{ 
	vec2 li = sdSegment( vec2(x1,y1), vec2(x2,y2), p );
    float d = li.x - mix(th1,th2,li.y);
	
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(-w, w, d);
}
//------------------------------------------------------
	
float logo( vec2 q )
{
	vec2 p = q - vec2(0.1,0.0);
	float f = 0.0;

	f = mix( f, 1.0, fillRectangle( p, -0.70, 0.00,  1.00, 0.000,  0.08, 0.40) );
	f = mix( f, 1.0, fillRectangle( p, -0.30, 0.00,  1.00, 0.000,  0.08, 0.50) );
	f = mix( f, 1.0, fillTriangle(  p, -0.84, 0.45, -0.56, 0.450, -0.70, 0.30) );	
	f = mix( f, 1.0, fillTriangle(  p, -0.56,-0.45, -0.84,-0.450, -0.70,-0.30) );	
	f = mix( f, 1.0, fillTriangle(  p, -0.44, 0.55, -0.16, 0.550, -0.30, 0.35) );	
	f = mix( f, 1.0, fillTriangle(  p, -0.16,-0.55, -0.44,-0.550, -0.30,-0.35) );	
	f = mix( f, 1.0, fillBezier(    p, -0.85,-0.01, -1.00, 0.080, -0.83, 0.10, 0.015, 0.002) );	
	f = mix( f, 1.0, fillQuad(      p, -0.05, 0.70,  0.20, 0.705,  0.24,-0.10, 0.050, 0.200) );	
	f = mix( f, 1.0, fillTriangle(  p, -0.07, 0.25,  0.15, 0.150,  0.24,-0.50) );	
	f = mix( f, 1.0, fillTriangle(  p,  0.00,-0.10,  0.18,-0.150,  0.17,-0.80) );	
	f = mix( f, 1.0, fillRectangle( p,  0.30, 0.60,  0.90, 0.440,  0.12, 0.03) );
	f = mix( f, 1.0, fillQuad(      p,  0.40, 0.68,  0.60, 0.400,  0.58, 0.100, 0.40, 0.150) );
	f = mix( f, 1.0, fillTriangle(    p,  0.58, 0.10,  0.35,-0.050,  0.40, 0.15) );	
	f = mix( f, 1.0, fillTriangle(    p,  0.37, 0.03,  0.35,-0.050,  0.15, 0.17) );	
	
	float d = (abs(sin(4.5*(p.x-1.15))*0.15+0.05-p.y) - 0.01 - 0.04*clamp(1.0-abs(p.x+0.45)/0.5,0.0,1.0));
	d = max( abs(p.x+0.4)-0.45, d );
    float w = fwidth(d)*8.0;
	f = mix( f, 1.0, 1.0 - smoothstep(0.0, w, d) );
	
	return f;
}

// bumped logo
float slogo( in vec2 p )
{
	float s = logo( p );
	s -= s*0.02*pow(fbm( 20.0*p.yx, 0.0 ),3.0);
	
	return s;
}

//-----------------------------------------------------------------------

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float time = mod( iTime, 40.0 );
	
	vec2 q = fragCoord.xy / iResolution.xy;
	vec2 p = (-iResolution.xy + 2.0*fragCoord.xy) / iResolution.y;

	p *= 1.5-1.45*smoothstep(0.0,36.0,time);
	
	float al = smoothstep( 3.0,12.0, time);
	
    // logo
	float f = 1.0 - al*0.4*logo( p );
	
	// background color
	vec3 col = vec3(f*0.2,f*0.6,f*0.7);

    // add texture
	col += f*3.0*smoothstep( 0.53, 0.9, fbm( 0.8*p.yx + 9.0, 0.0 ) );
	col *= 0.7 + 0.3*smoothstep( 0.2, 0.8, fbm( 4.0*p  + fbm(32.0*p, 1.0), 2.0 ) );
    col *= 0.4;
	
    // calc normal	
	float a = slogo( p );
	float b = slogo( p + vec2(2.0,0.0)/ iResolution.y );
	float c = slogo( p + vec2(0.0,2.0)/ iResolution.y );
	vec3 nor = normalize( vec3(a-b,1.0/iResolution.y,a-c) );

	// lighting
	col += al*a*    vec3(0.7,0.55,0.4)*(1.0-f)*(0.75+0.25*dot(nor,normalize(vec3(-1.0,0.2,1.0))));
	col += al*a*2.0*vec3(2.0,0.80,0.3)*0.3*pow(clamp(dot(nor,normalize(vec3(-1.0,0.5,1.0))),0.0,1.0),2.0);
	col += al*a*3.0*vec3(1.0,1.00,1.0)*0.6*pow(clamp(dot(nor,normalize(vec3(-1.0,0.5,1.0))),0.0,1.0),8.0);

	
	// vigneting
	col *= 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y);
	
    // fade in	
	col *=     smoothstep(  0.0,  6.0, time );
	col *= 1.0-smoothstep( 30.0, 36.0, time );

	fragColor = vec4(col,1.0);
}
