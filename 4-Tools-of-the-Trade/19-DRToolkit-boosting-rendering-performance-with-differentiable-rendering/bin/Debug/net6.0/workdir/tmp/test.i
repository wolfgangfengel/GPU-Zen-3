#line 1 "workdir/test.glsl"



float gameTime;



mat2 rot(in float a) { return mat2(cos(a),sin(a),-sin(a),cos(a)); }

float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); }



float petalDcp(in vec2 uv, in float w)
{
 uv.x = abs(uv.x) + .25 + .25*w;
 return length(uv) - .5;
}

float petal(in vec3 p, in float m)
{
 float tt = mod(gameTime, 6.283185307*.5);

 float ouv = m - .015;
 float w = m;
 float a = m;
 const float b = .5;
 p.y -= .45;
 p.z -= b*1.;
 p.zy *= rot(ouv*2.);
 float pDcp = petalDcp(p.xy, w);
 p.x = abs(p.x);
 p.xz *= rot(-.25);
 float c1 = length(p.yz) - b;
 return max(max(pDcp, abs(c1) - .01), p.z);
}

vec2 repRot(in vec2 p, in float aIt)
{
 return p*rot(-(6.283185307/aIt)*floor((atan(p.x, p.y)/6.283185307 + .5)*aIt) - 6.283185307*.5 - 6.283185307/(aIt*2.));
}

float flower(in vec3 p, in float aIt, in float m)
{
 p.xy = repRot(p.xy, aIt);
 return petal(p, m);
}

float df(in vec3 _pp, inout int m) {

    _pp.y = -_pp.y;
    _pp.xz *= rot(1.016), _pp.xy *= rot(-0.640);

    float dd = 10e9, ee = 10e9;
    vec3 p = _pp;

    const float fsz = .25;
    const vec2 n = vec2(cos((6.283185307*.5*.125)),sin((6.283185307*.5*.125)));

    bool b = false;
    for(float g = 0.; g < 3.; g++)
    {
        p = (b = !b) ? p.xzy : p.zxy;
        float r = length(p.xy);
        vec3 pp = vec3(log(r) - gameTime*(.1+((g+1.)*.051)), atan(p.x, p.y) , p.z/r);
        float e = dot(pp.xy, n), f = dot(pp.xy, vec2(n.y,-n.x));
        {float k = 1.2021; e = mod(e, k) - k*.5;}
        float l = .65; f += 1.3; float i = mod(floor(f/l) + g, 3.); f = mod(f, l) - l*.5;
        float d = (length(vec2(e, pp.z)) - 0.015/r)*r;
        bool j = i == 0.;
        dd = opSmoothUnion(dd, d, .1);
        float ff = flower(vec3(e, f, pp.z + .06)/fsz, smoothstep(-1., 1., r*r)*(j ? 5. : 2.), smoothstep(1., -0., r*r))*fsz*r;
        ee = min(ee, ff);
        if(ee == ff) m = j ? 1 : 0;
    }

    float ff = min(dd, ee);
    if(ff == dd) m = 0;
    return ff*.8;
}





vec3 normal(in vec3 p, inout int m) { float d = df(p, m); vec2 u = vec2(0.,.0002); return normalize(vec3(df(p + u.yxx, m),df(p + u.xyx, m),df(p + u.xxy, m)) - d); }

struct rmRes { vec3 p; int i; bool h; };
rmRes rm(in vec3 c, in vec3 r, inout int m)
{
    rmRes s = rmRes(c + r*0., 0, false);
    float d;
    int i = 0;
    for(; i < 16; i++) {
        d = df(s.p, m);
        if(d < .0002) { s.h = true; break; }
        if(distance(c,s.p) > 30.) break;
        s.p += d*r;
    }
    s.i = i;
    return s;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = vec2(1071.0, 503.0) / iResolution.xy;
    uv = vec2(0.5950, 0.4965);
    vec2 coord = uv * iResolution.xy;
    coord = fragCoord;

    int m = 0;
    vec2 st = (coord - iResolution.xy*.5)/iResolution.x;
    gameTime = iTime;

    vec3 c = vec3(0.,0.,-10.), r = normalize(vec3(st,1.));

    rmRes res = rm(c,r,m);

    vec3 sky = (vec3(0.955,0.912,0.931) - dot(st,st)*.2);
    vec3 color = sky;

    if(res.h)
    {
        vec3 n = normal(res.p, m);
        const vec3 ld = normalize(vec3(0.,1.,-.1));
        float d = max(0., dot(n, ld));
        float s = pow(max(0., dot(r, reflect(ld, n))), 1.);
        color = mix(vec3(0.500,0.763,0.915), vec3(1.), d);
        color *= m == 1 ? vec3(0.905,0.170,0.292) :vec3(0.885,0.882,0.945);

        color = mix(color, sky, smoothstep(20., 25., distance(res.p, c)));
        color = mix(color, sky, smoothstep(0.5, 3., dot(st,st)*10.));
    }

    fragColor = vec4(color, 1.0);
}
