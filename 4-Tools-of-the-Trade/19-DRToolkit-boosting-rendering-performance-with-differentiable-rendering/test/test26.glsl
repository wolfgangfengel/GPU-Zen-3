// Hazel Quantock 2017
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// https://creativecommons.org/licenses/by-nc-sa/3.0/

float g_blastTime;
vec3 g_cloudCentre;
void InitBlastParams()
{
    g_blastTime = fract(iTime/20.);
    g_cloudCentre = vec3(0,g_blastTime*5.,0);
}

vec3 Flow( vec3 pos )
{
    // make a toroidal roll, like a mushroom cloud
    vec3 p = pos - g_cloudCentre;
    vec3 v;
    v.xz = -normalize(p.xz)*p.y;
    v.y = length(p.xz)-.8;
    //v *= smoothstep(.0,.5,length(pos.xz)); bad: this squashes/stretches it
    v *= .1;
    
    // reduce velocity with distance from cloud top edge
    float g = length(vec2(p.y,length(p.xz)-.8))-1.; // this doesn't match the one in SDF, but it looks better with the mismatch
    v *= exp2(-pow(g*3.,2.));
    
    return v;
}

float SDF( vec3 pos )
{
    // multi fractal
    const float period = 1.6;
    float tt = fract(iTime/period);
    float t[2] = float[2]( tt*period, (tt-1.)*period );
    vec3 uvw = (pos-g_cloudCentre)/30.;
    float f[2] = float[2]( .0, .0 );

    // applying flow to the whole SDF causes a "pulsing" - because we're displacing in a straight line so convex curves shrink
    for ( int i=0; i < 2; i++ )
    {
	    vec3 offset = Flow(pos)*t[i];
        vec3 u = uvw+offset*.2; offset *= .2; // makes the loop more obvious but looks generally good
        f[i] += texture(iChannel0,offset+u*2.).x/2.;
        f[i] += texture(iChannel0,offset+u*4.).x/4.;
        f[i] += texture(iChannel0,offset+u*8.).x/8.;
        f[i] += texture(iChannel0,offset+u*16.).x/16.;
        f[i] += texture(iChannel0,offset+u*32.).x/32.;
    }
    
    float ff = mix( f[0], f[1], tt );// actually better with a longer fade imo smoothstep(.4,.6,tt) );

    ff *= .5; // strength of clouds vs bounding shapes

    vec3 p = pos - g_cloudCentre;
    float bulge = 1.-exp2(-20.*g_blastTime);
    float g = length(vec2(p.y,length(p.xz)-1.*bulge))-1.;
    ff *= bulge; // smooth sphere to start
    
    // vertical column
    float h = length(pos.xz)-.7+.2*(g_cloudCentre.y-pos.y-1.2); // cylinder - slightly tapered to cone
    h = max(h, pos.y-g_cloudCentre.y); // cut off at top (inside cloud)
    h = max(h,(g_cloudCentre.y*1.25-4.-pos.y)*.3); // softer cut off at the bottom
    
    g = min(g,h);
    ff += g*.6;
    
    return ff;
}

void mainImage( out vec4 o, in vec2 uv )
{
    InitBlastParams();
    
    vec3 ray = vec3((uv-iResolution.xy*.5)/iResolution.y,.9);
    ray = normalize(ray);
    vec3 pos = vec3(0,2,-5);
    
    vec2 a = iMouse.xy/iResolution.xy - .5;
    if ( iMouse.x == .0 && iMouse.y == .0 )
        a = vec2(0,.15);
    a *= vec2(3,2);
    
    vec3 csx = vec3(cos(a.x),sin(a.x),-sin(a.x));
    vec3 csy = vec3(cos(a.y),sin(a.y),-sin(a.y));
    
    pos.yz = pos.yz*csy.x + pos.zy*csy.yz;
    pos.xz = pos.xz*csx.x + pos.zx*csx.yz;
    ray.yz = ray.yz*csy.x + ray.zy*csy.yz;
    ray.xz = ray.xz*csx.x + ray.zx*csx.yz;
    
    pos.y = max(.01,pos.y);
    
    float softness = .1+pow(g_blastTime,2.)*.5;
    float density = 1.2/softness;
    
    const float epsilon = .001; // could scale this to pixel size - works well in big scenes
    float visibility = 1.;
    float light0 = 0.;
    float light1 = 0.;
    vec3 sunDir = normalize(vec3(1));
    for ( int i=0; i < 20; i++ ) // can get away with really low loop counds because of the soft edges!
    {
        float h = SDF(pos);
        float vis = smoothstep(epsilon,softness,h); // really should do an integral thing using previous h
        if ( pos.y < .0 ) vis = 1.;
        h = max(h,epsilon); // ensure we always march forward
        if ( vis < 1. )
        {
            float newvis = visibility * pow(vis,h*density);
	    	light0 += (visibility - newvis)
                		*smoothstep( -.5, 1., (SDF(pos+sunDir*softness) - h)/softness );
            vec3 lightDelta = g_cloudCentre-pos;
	    	light1 += (visibility - newvis)
                		*pow(smoothstep( -1., 1., (SDF(pos+normalize(lightDelta)*softness) - h)/softness ),2.)
                		/(dot(lightDelta,lightDelta)+1.); // inverse square falloff
            visibility = newvis;
        }
        
        if ( vis <= 0.
            || pos.y < .0 ) // cut off to ground plane (assumes camera is above)
            break;
        pos += h*ray;
    }

    o = vec4(.1,.2,.3,1); // ambient
    o += light0*vec4(.9,.8,.7,0);
	o *= pow(g_blastTime,.5)*.5; // albedo (before the glow, so I can blance the two lights separately
    
    o += light1*vec4(8,2,.25,0)/(25.*pow(g_blastTime,2.));
    
    // sky
    o = mix( o, vec4(.2,.4,.8,1)+.003/g_blastTime, visibility );
    
    o = pow(o,vec4(1./2.2));
}