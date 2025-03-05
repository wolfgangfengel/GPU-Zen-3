float random2d(vec2 Seed)
{
  float randomno = fract(sin(dot(Seed, vec2(12.9898, 78.233))) * 43758.5453);
  return randomno - 0.5;
}
bool intersectBox(vec3 ro, vec3 rd, out float tnear, out float tfar)
{
  vec3 boxmin = vec3(0.f, 0.f, 50.f);
  vec3 boxmax = vec3(160.f, 120.f, 100.f);

  vec3 invR = 1.0f / rd;
  vec3 tbot = invR * (boxmin - ro);
  vec3 ttop = invR * (boxmax - ro);

  vec3 tmin = vec3(100000.0f);
  vec3 tmax = vec3(0.0f);

  tmin = min(tmin, min(ttop, tbot));
  tmax = max(tmax, max(ttop, tbot));

  tnear = max(max(tmin.x, tmin.y), tmin.z);
  tfar = min(min(tmax.x, tmax.y), tmax.z);

  return tfar > tnear;
}
float intersectPlane(vec3 ro, vec3 rd, vec3 pn, float pd)
{
  float denominator = dot(rd, pn);
  float ro_pn_d = dot(ro, pn) / denominator;
  float t = -(ro_pn_d + pd / denominator);
  return t;
}
float sphIntersect( vec3 ro, vec3 rd, vec4 sph )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    float d = -1.0;
    if( h >= 0.0 ) {
        h = sqrt( h );
        d = -b - h;
    }
    return d;
}
vec3 calcLighting(in vec3 ro, in vec3 rd, in float rz, in vec3 normal, in vec3 lightPos, in vec3 lightColor, in vec3 objColor)
{
    vec3 fPos = ro + rd*rz;
    // ambient
    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(ro - fPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * objColor;
    return result;
}

float tri(in float x){return abs(fract(x)-.5);}
vec3 tri3(in vec3 p){return vec3( tri(p.z+tri(p.y*1.)), tri(p.z+tri(p.x*1.)), tri(p.y+tri(p.x*1.)));}
mat2 m2 = mat2(0.970,  0.242, -0.242,  0.970);

float triNoise3d(in vec3 p, in float spd)
{
  float z=1.4;
  float rz = 0.;
  vec3 bp = p;
	for (int i=0; i<=3; i++ )
	{
    vec3 dg = tri3(bp*2.);
    p += (dg+iTime*spd);

    bp *= 1.8;
	z *= 1.5;
	p *= 1.2;
    //p.xz*= m2;
    
    rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
    bp += 0.14;
	}
	return rz;
}
vec3 fogmap(in vec3 p, in float scaleByDist, in float mixDist, in int i, in vec3 col, in float minDist, in float rz, in float alpha, in float velocity)
{
    float fr = triNoise3d(p * 2.2 / (scaleByDist + 20.0), velocity) * alpha;
    float grd = 0.3 * float(i);
    vec3 col2 = vec3(1.0, 1.0, 1.0) * (1.0 + grd);
    float r = smoothstep(0.1, minDist + 0.11, rz);
    fr *= r * r;
    col2 = col2 * 0.5 + col2 * smoothstep(0.0, 15.0, rz);
    col = mix(col, col2, clamp(fr * smoothstep(mixDist - 0.4, (mixDist * 1.75 + 2.0), rz), 0.0, 1.0));
    return col;
}
float RayIntersectSphere(vec3 ro, vec3 rd, vec3 center, float radius, bool reverse)
{
    float r = 0.1;
    vec3 oc = center - ro;
    float prj = dot(rd, oc);
    float oc2 = dot(oc, oc);
    float prj2 = prj * prj;
    float radius2 = radius * radius;

    float dist2 = oc2 - prj2;
    if (dist2 <= radius2) {
        float discriminant2 = radius2 - dist2;
        if (discriminant2 < 0.00001) {
            r = prj;
        }
        else {
            float discriminant = sqrt(discriminant2);
            r = reverse ? prj - discriminant : prj + discriminant;
        }
    }
    return r;
}

float get_camera_z()
{
    return 10.0;//-30.0 + 120.0*sin(iTime*1.0);
}
vec3 get_camera_pos()
{
    return vec3(80.0+20.0*sin(iTime*0.6), 60.0, get_camera_z());
}

const vec2 vp = vec2(320.0, 240.0);
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 p = fragCoord.xy/iResolution.xy-0.5;
  vec2 q = fragCoord.xy/iResolution.xy;
	p.x *= iResolution.x/iResolution.y;
  vec2 mo = iMouse.xy / iResolution.xy;
  mo.y = 1.0 - mo.y;
	mo.x *= iResolution.x / iResolution.y;
    
  //vec3 eyedir = normalize(vec3(cos(mo.x),mo.y*2.-0.2,sin(mo.x)));
  //vec3 rightdir = normalize(vec3(cos(mo.x+1.5708),0.,sin(mo.x+1.5708)));
  //vec3 updir = normalize(cross(rightdir,eyedir));
	//vec3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
	
  vec3 boxmin = vec3(0.f, 0.f, 0.f);
  vec3 boxmax = vec3(160.f, 120.f, 120.f);
  vec3 center = (boxmin + boxmax) / 2.0;
  vec3 size = boxmax - boxmin;
    
	vec3 ro = vec3(80.0, 60.0, 50.0);
  ro = get_camera_pos();
  vec3 eyedir = normalize(vec3(0.0, -0.6, 1.0));
  vec3 rightdir = normalize(cross(eyedir, vec3(0.0, 1.0, 0.0)));
  vec3 updir = normalize(cross(rightdir, eyedir));
  rightdir = normalize(cross(eyedir, updir));
	vec3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
	
  vec3 color = vec3(1.0, 1.0, 1.0);
  vec3 lightPos = vec3(80.0, 120.0, 50.0);
  vec3 lightCol = vec3(1.0, 1.0, 1.0);
  vec3 plane = vec3(0.0, 1.0, -1.0);
  float rz = intersectPlane(ro, rd, plane, 57.0);
  color = calcLighting(ro, rd, rz, plane, lightPos, lightCol, vec3(0.0, 0.5, 1.0));
  for(int ix=0; ix < 3;++ix){
      vec3 sc = vec3(40.0 + 30.0 * float(ix), 20.0 * float(ix), 60.0 + 10.0 * float(ix));
      float srz = sphIntersect(ro, rd, vec4(sc, 5.0));
      if(srz>0.0 && srz<rz){
          rz = srz;
          vec3 sn = normalize((ro+rd*rz) - sc);
          color = calcLighting(ro, rd, rz, sn, lightPos, lightCol * 2.0, vec3(0.3, 1.0, 0.3));
      }
  }
  
  float minD, maxD;
  bool r = intersectBox(ro, rd, minD, maxD);
  
  float radius1 = min(size.x, min(size.y, size.z)) / 2.0;
  float radius2 = max(size.x, max(size.y, size.z)) / 2.0;
  float radius = (radius1 + radius2) / 2.0;
  
  vec3 fpt1 = ro + minD * rd;
  vec3 fpt2 = ro + maxD * rd;
  
  vec3 oc = ro - center;
  float oc2 = dot(oc, oc);
  float r2 = 37.5 * 37.5;
  
  /*
  bool isZ = (abs(rd.x) < abs(rd.z));
  float signx = sign(rd.x);
  float signz = sign(rd.z);
  float distStep = 5.0;
  vec3 n = isZ ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  float d1 = isZ ? -(floor(abs(ro.z) / distStep) + signz * 1.0) * distStep : -(floor(abs(ro.x) / distStep) + signx * 1.0) * distStep;
  float d2 = isZ ? -(floor(abs(ro.z) / distStep) + signz * 2.0) * distStep : -(floor(abs(ro.x) / distStep) + signx * 2.0) * distStep;
  float d3 = isZ ? -(floor(abs(ro.z) / distStep) + signz * 3.0) * distStep : -(floor(abs(ro.x) / distStep) + signx * 3.0) * distStep;

  float denominator = dot(rd, n);
  float ro_n_d = dot(ro, n) / denominator;
  float t1 = -(ro_n_d + d1 / denominator);
  float t2 = -(ro_n_d + d2 / denominator);
  float t3 = -(ro_n_d + d3 / denominator);
  */
  
  
  float coe[3] = float[3](7.5, 15.5, 27.0);
  
  float t1 = oc2 < r2 ? RayIntersectSphere(ro, rd, center, coe[0], false) : coe[0];
  float t2 = oc2 < r2 ? RayIntersectSphere(ro, rd, center, coe[1], false) : coe[1];
  float t3 = oc2 < r2 ? RayIntersectSphere(ro, rd, center, coe[2], false) : coe[2];
  
  vec3 pt1 = ro + t1 * rd;
  vec3 pt2 = ro + t2 * rd;
  vec3 pt3 = ro + t3 * rd;
  
  float scale = 0.1;
  float scaleByDist = 1.0;
  float mixDist = 1.0;
  float minDist = 3.0;
  float alpha = 0.5;
  float velocity = 0.03;
  
  vec3 pts[3] = vec3[3](pt1, pt2, pt3);
  
  if (fpt2.y - center.y >= size.y / 2.0) {
  }
  else {
      for(int i=0; i<3; ++i) {
          color = fogmap(pts[i] * scale, coe[i] * scaleByDist, coe[i] * mixDist, i, color, minDist, rz, alpha, velocity);
          if(coe[i] * mixDist>rz)
              break;
      }
  }
  color /= 2.0;
  fragColor = r ? vec4(color, 1.0) : vec4(0.5,0.5,0.5,1.0);
}