//#define hlsl_attr(x)
#define DTR 0.01745329
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))

vec3 glv;
float tt;

float bx(vec3 p,vec3 s)
{
    vec3 q=abs(p)-s;
    return min(max(q.x,max(q.y,q.z)),0.)+length(max(q,0.));
}
float cy(vec3 p, vec2 s)
{
    p.y+=s.x/2.;
    p.y-=clamp(p.y,0.,s.x);
    return length(p)-s.y;
}

float shatter(vec3 p, float d, float n, float a, float s)
{
    hlsl_attr([unroll(1)])
	for(float i=0.;i<n;i++) {
		p.xy*=rot(a);
        p.xz*=rot(a*0.5);
        p.yz*=rot(a+a);
		float c=mod(i,3.)==0.?p.x:mod(i,3.)==1.?p.y:p.z;
		c=abs(c)-s;
        d=max(d,-c);
	}
	return d; 
}

vec3 lattice(vec3 p, int iter)
{
    hlsl_attr([unroll(3)])
    for(int i = 0; i < iter; i++) {
        p.xy *= rot(45.*DTR);
        p.xz *= rot(45.*DTR);
        p=abs(p)-1.;
        
        p.xy *= rot(-45.*DTR);
        p.xz *= rot(-45.*DTR);
    }
    return p;
}

float mp(vec3 p, inout vec3 oc, inout float oa, inout float io, inout vec3 ss, inout vec3 vb, inout int ec)
{
    //now with mouse control
    if(iMouse.z>0.) {
        p.yz*=rot(2.0*(iMouse.y/iResolution.y-0.5));
        p.zx*=rot(-7.0*(iMouse.x/iResolution.x-0.5));
    }
    vec3 pp = p;
    
    p.xz*=rot(tt*0.2);
    p.xy*=rot(tt*0.2);

    p = lattice(p, 3);

    float sd = cy(p,vec2(1.)) - 0.05;

    sd = shatter(p,sd, 1.,sin(tt*0.1),0.2);

    sd = min(sd, bx(p,vec3(0.1,2.1,8.)) - 0.3);

    sd = mix(sd, cy(p, vec2(4,1)), cos(tt*0.5)*0.5+0.5);

    sd = abs(sd)-0.001;
    if(sd<0.001) {
        oc=mix(vec3(1.,0.1,0.6), vec3(0.,0.6,1.), pow(length(pp)*0.18,1.5));
        io=1.1;
        oa=0.05 + 1.-length(pp)*0.2;
        ss=vec3(0.);
        vb=vec3(0.,2.5,2.5);
        ec=2;	
    }
    return sd;
}

void tr(vec3 ro, vec3 rd, inout vec3 oc, inout float oa, inout float cd, inout float td, inout float io, inout vec3 ss, inout vec3 vb, inout int ec)
{
    vb.x=0.;
    cd=0.;
    for(float i=0.;i<64.;i++) {
        float sd = mp(ro+rd*cd, oc, oa, io, ss, vb, ec);
        cd+=sd;
        td+=sd;
        if(sd<0.0001 || cd>128.)
            break;
    }
}
vec3 nm(vec3 cp, inout vec3 oc, inout float oa, inout float io, inout vec3 ss, inout vec3 vb, inout int ec)
{
    mat3 k=mat3(cp,cp,cp)-mat3(.001);
    return normalize(mp(cp, oc, oa, io, ss, vb, ec)-vec3(mp(k[0], oc, oa, io, ss, vb, ec),mp(k[1], oc, oa, io, ss, vb, ec),mp(k[2], oc, oa, io, ss, vb, ec)));
}

vec3 px(vec3 rd, vec3 cp, vec3 cr, vec3 cn, float cd, inout vec3 oc, inout float oa, inout float io, inout vec3 ss, inout vec3 vb, inout int ec)
{
    vec3 cc=vec3(0.7,0.4,0.6)+length(pow(abs(rd+vec3(0,0.5,0)),vec3(3)))*0.3+glv;
    if(cd>128.) {
        oa=1.;
        return cc;
    }
    vec3 l=vec3(0.4,0.7,0.8);
    float df=clamp(length(cn*l),0.,1.);
    vec3 fr=pow(1.-df,3.)*mix(cc,vec3(0.4),0.5);
    float sp=(1.-length(cross(cr,cn*l)))*0.2;
    float ao=min(mp(cp+cn*0.3, oc, oa, io, ss, vb, ec)-0.3,0.3)*0.5;
    cc=mix((oc*(df+fr+ss)+fr+sp+ao+glv),oc,vb.x);
    return cc;
}

vec4 render(vec2 frag, vec2 res, float time)
{
    vec2 uv;
    vec3 cp,cr,ro,rd;
    vec4 fc = vec4(0.1);
    vec3 oc, ss, vb;
    float oa, io, cd, td;
    int es=0,ec;
    tt=mod(time, 260.);
    uv=vec2(frag.x/res.x,frag.y/res.y);
    uv-=0.5;
    uv/=vec2(res.y/res.x,1);
    ro=vec3(0,0,-15);
    rd=normalize(vec3(uv,1));

    for(int i=0;i<64;i++)
    {
        tr(ro, rd, oc, oa, cd, td, io, ss, vb, ec);
        cp=ro+rd*cd;
        vec3 cn = nm(cp, oc, oa, io, ss, vb, ec);
        ro=cp-cn*0.01;
        cr=refract(rd,cn,i%2==0?1./io:io);
        if(length(cr)==0.&&es<=0) {
            cr=reflect(rd,cn);
            es=ec;
        }
        if(max(es,0) % 3 == 0 && cd < 128.)
            rd=cr;
        es--;
        if(vb.x > 0. && i%2 == 1)
            oa=pow(clamp(cd/vb.y,0.,1.),vb.z);
        vec3 cc = px(rd, cp, cr, cn, cd, oc, oa, io, ss, vb, ec);
        fc=fc+vec4(cc*oa,oa)*(1.-fc.a);
        if((fc.a>=1. || cd>128.))
            break;
    }
    vec4 col = fc/clamp(fc.a, 0.01, 1.0);
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 coord = fragCoord.xy;
    //coord = vec2(31.0,30.0);
    fragColor = render(coord,iResolution.xy,iTime);
}
