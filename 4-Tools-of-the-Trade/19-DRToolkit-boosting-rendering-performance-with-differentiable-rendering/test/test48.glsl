const float PI = float(3.14159265359);
const vec3 BLOOD_COLOR = vec3(0.3, 0.03, 0.04);

// Globals
vec2 GEyeRot = vec2(0.0);
vec2 GEyelidRot = vec2(1.0);
float GEyeBrowDown = 0.0;

float saturate(float X)
{
    return clamp(X, 0.0, 1.0);
}

vec3 saturate(vec3 X)
{
    return clamp(X, 0.0, 1.0);
}

float Sphere(vec3 Pos, float Radius)
{
	return length(Pos) - Radius;
}

float Max(vec3 V)
{
    return max(max(V.x, V.y), V.z);
}

float Box(vec3 Pos, vec3 Ext) 
{
	vec3 Dist = abs(Pos) - Ext;
	return length(max(Dist, vec3(0.0))) + Max(min(Dist, vec3(0.0)));
}

float Rectangle(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float UnionSoft(float a, float b, float r) 
{
	float e = max(r - abs(a - b), 0.0);
	return min(a, b) - e * e * 0.25 / r;
}

float SubstractRound(float a, float b, float r) 
{
	vec2 u = max(vec2(r + a,r - b), vec2(0));
	return min(-r, max (a, -b)) + length(u);
}

void Rotate(inout vec2 p, float a) 
{
    p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

float EyelidCutSDF(vec3 Pos)
{    
    float EyelidHi = mix(length(Pos.xy - vec2(-0.1, -1.2)) - 1.3, +Pos.y + 0.20, GEyelidRot.x);
    float EyelidLo = mix(length(Pos.xy - vec2(+0.0, +1.0)) - 1.3, -Pos.y - 0.15, GEyelidRot.y);
    float EyelidCut = SubstractRound(EyelidHi, -EyelidLo, 0.02);
    
    EyelidCut = SubstractRound(EyelidCut, 
                               -(UnionSoft(
                                   length(Pos.xy - vec2(-0.1, +1.0)) - 1.3, 
                                   -Pos.y - 0.14, 0.01)),
                               0.02);
    
    return EyelidCut;
}

float EyeCornerSDF(vec3 Pos)
{
    Pos -= vec3(0.53, -0.15, -0.79);
    Rotate(Pos.xy, -0.3);
    float Dist = length(Pos.xz) - 0.07;
    Dist = UnionSoft(Dist, Sphere(Pos - vec3(0.05, 0.03, 0.03), 0.1), 0.02);
    return Dist;
}

float EyeballSDF(vec3 Pos)
{
    vec3 EyePos = Pos;
	Rotate(EyePos.xz, GEyeRot.x);
    Rotate(EyePos.yz, GEyeRot.y);
    
    float Eye = Sphere(EyePos, 1.0);
	float EyeCorner = EyeCornerSDF(Pos);
	    
    float EyeBulge = SubstractRound(Sphere(EyePos - vec3(0.0, 0.0, -0.53), 0.5), EyeCorner, 0.1);
    
    float Dist = UnionSoft(UnionSoft(Eye, EyeBulge, 0.01), EyeCorner, 0.02);
    return Dist;
}

float IrisScene(vec3 Pos)
{
	Rotate(Pos.xz, GEyeRot.x);
    Rotate(Pos.yz, GEyeRot.y);    
    
    float Eye = Sphere(Pos, 1.0);
    
    float EyeBulge = Sphere(Pos - vec3(0.0, 0.0, -1.0), 0.2);

    float Dist = SubstractRound(Eye, EyeBulge, 0.1);
    
	return Dist;
}

float Scene(vec3 p)
{
    vec3 t = p;
    Rotate(t.yz, 0.5);
    Rotate(t.xy, 0.55);     
    float Face = length(t.xz - vec2(-0.5, 1.15)) - 1.8;
    
    t = p;
    Face = UnionSoft(Face, length(t.xz - vec2(0.2, 0.02)) - 1.0, 0.1);
    
    
    // Top
    t = p;
    float Top = length(t.yz - vec2(0.8,-0.3)) - 0.9;
    
    Top = UnionSoft(Top, length(t.yz - vec2(mix(0.8, 0.65, GEyeBrowDown),-0.3)) - 0.9, 0.1);
    
    t = p - vec3(1.5, 0.65, -0.7);
    Rotate(t.xy, -0.25);
    Rotate(t.xz, -0.1);
    Top = UnionSoft(Top, Box(t, vec3(1.5, 0.5, 0.45)), 0.2);
        
    t = p;
    Rotate(t.xy, -0.15 * PI);
    Rotate(t.xz, 0.20 * PI);  
    Top = SubstractRound(Top, t.x + 1.6, 1.0);
    
    Top = UnionSoft(Top, length(p.xz - vec2(-0.3, 1.5)) - 2.3, 0.1);
            
	Face = UnionSoft(Face, Top, 0.2);        
    
    // Nose    
    t = p;
    Rotate(t.yz, 0.2 * PI);
    Face = UnionSoft(Face, length(t.xz - vec2(1.5,-0.5)) - 0.6, 0.1);
    
    t = p;
    Rotate(t.xy, 0.25 * PI);
    Face = UnionSoft(Face, length(t.xz - vec2(1.25,-0.9)) - 0.6, 0.2);
    
    t = p;
    Rotate(t.xy, -0.15 * PI);
    Face = UnionSoft(Face, length(t.xz - vec2(1.4,-0.9)) - 0.5, 0.1);    
    
    // Right line
    t = p;
    Rotate(t.xz, 0.30 * PI);
    Face = SubstractRound(Face, length(t.xy - vec2(-1.7,-0.1)) - 0.5, 0.4);
    
    // Cheek
    Face = UnionSoft(Face, Sphere(p - vec3(0.2 * GEyeRot.x, -1.2 + 0.05 * GEyelidRot.y + 0.2 * GEyeRot.y, -0.35), 1.0), 0.1);
            
  	float EyelidCut = EyelidCutSDF(p);	
    
    float Eyelid = Sphere(p, 1.05);
    
    t = p;
    Rotate(t.xy, 0.1);
    Rotate(t.xz, -0.6);
    Eyelid = UnionSoft(Eyelid, max(-t.z - 0.25, -p.x + 0.3), 0.2);


    // Animated eye bulge
    vec3 t2 = p;
    Rotate(t2.xz, GEyeRot.x);
    Rotate(t2.yz, GEyeRot.y);
    
    float EyeBulge = Sphere(t2 - vec3(0.0, 0.0, -0.7), 0.45);
    Eyelid = UnionSoft(Eyelid, EyeBulge, 0.1);
    
    
	Face = SubstractRound(Face, Sphere(p, 0.9), 0.3);
    
    Face = UnionSoft(Eyelid, Face, 0.05);
    
    Face = SubstractRound(Face, EyelidCut, 0.05);

	return min(Face, EyeballSDF(p));
}

float CastRay(in vec3 ro, in vec3 rd)
{
    const float maxd = 5.0;
    
	float h = 1.0;
    float t = 0.0;
   
    for (int i = 0; i < 50; ++i)
    {
        if (h < 0.001 || t > maxd)
        {
            break;
        }
        
	    h = Scene(ro + rd * t);
        t += h;
    }

    if (t > maxd)
    {
        t = -1.0;
    }
	
    return t;
}

float CastIrisRay(in vec3 ro, in vec3 rd)
{
    const float maxd = 0.5;
    
	float h = 1.0;
    float t = 0.0;
   
    for (int i = 0; i < 8; ++i)
    {
        if (h < 0.001 || t > maxd)
        {
            break;
        }
        
	    h = IrisScene(ro + rd * t);
        t += h;
    }

    if (t > maxd)
    {
        t = -1.0;
    }
	
    return t;
}

// https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
vec3 EnvBRDFApprox(vec3 SpecularColor, float Roughness, float NdotV)
{
    const vec4 C0 = vec4(-1, -0.0275, -0.572, 0.022);
    const vec4 C1 = vec4(1, 0.0425, 1.04, -0.04);
    vec4 R = Roughness * C0 + C1;
    float A004 = min(R.x * R.x, exp2(-9.28 * NdotV)) * R.x + R.y;
    vec2 AB = vec2(-1.04, 1.04) * A004 + R.zw;
    return SpecularColor * AB.x + AB.y;
}

// https://www.shadertoy.com/view/4djSRW
float Hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// https://www.shadertoy.com/view/4ddfWr
float SmoothNoise(in vec2 o) 
{
	vec2 p = floor(o);
	vec2 f = fract(o);

	float a = Hash12(p);
	float b = Hash12(p+vec2(1,0));
	float c = Hash12(p+vec2(0,1));
	float d = Hash12(p+vec2(1,1));
	
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	
	vec2 t = 3.0 * f2 - 2.0 * f3;
	
	float u = t.x;
	float v = t.y;

	float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
    
    return res;
}

float IrisTexture(vec2 UV)
{
    float Spots = 1.5 * texture(bufferA_iChannel1, 3.0 * UV).x;
    
    UV *= 10.0;
    
    float Ret = SmoothNoise(10.0 * UV / pow(length(UV), 0.7)) * 0.7;
        
    UV *= 30.0;

    Ret += SmoothNoise(10.0 * UV / pow(length(UV), 0.7)) * 0.7;
    
    return Ret * Spots;
}

void Repeat(inout float p, float w)
{
    p = mod(p, w) - 0.5 * w;
}

float SkinHeight(vec2 Pos, float StoneMask)
{    
    float Skin = 1.0 - mix(texture(bufferA_iChannel1, 1.0 * Pos.xy).x, texture(bufferA_iChannel1, 2.0 * Pos.xy).x, 0.3);
    float Stone = mix(texture(bufferA_iChannel1, 0.5 * Pos.xy).x, texture(bufferA_iChannel1, 1.3 * Pos.xy).x, 0.3);
    
    return mix(Skin, Stone, StoneMask);
}

float Circle(vec2 Pos, float Radius)
{
    return (length(Pos / Radius) - 1.0) * Radius;
}

float VeinTexture(vec2 EyeUV)
{
    float Veins = 10000.0;
    
    vec2 Pos = EyeUV.xy - 0.5 + 0.05;
    Pos.x = abs(Pos.x);
    
    vec2 TempPos = Pos.xy;
    Rotate(TempPos.xy, SmoothNoise(EyeUV.xy*30.0)*0.3 + SmoothNoise(EyeUV.xy*10.0)*0.1);
    Veins = min(Veins, abs(TempPos.y));
    
    TempPos = Pos.xy + vec2(0.0, -0.05);
    Rotate(TempPos.xy, SmoothNoise(EyeUV.xy*40.0)*0.3 + SmoothNoise(EyeUV.xy*15.0)*0.1);
    Veins = min(Veins, abs(TempPos.y));
    
    TempPos = Pos.xy + vec2(0.0, -0.03);
    Rotate(TempPos.xy, SmoothNoise(EyeUV.xy*50.0)*0.3 + SmoothNoise(EyeUV.xy*20.0)*0.1);
    Veins = min(Veins, abs(TempPos.y));
    
    TempPos = Pos.xy + vec2(0.0, -0.01);
    Rotate(TempPos.xy, SmoothNoise(EyeUV.xy*40.0)*0.3 + SmoothNoise(EyeUV.xy*20.0)*0.1);
    Veins = min(Veins, abs(TempPos.y));    
    
    TempPos = Pos.xy + vec2(0.0, -0.07);
    Rotate(TempPos.xy, SmoothNoise(EyeUV.xy*30.0)*0.3 + SmoothNoise(EyeUV.xy*10.0)*0.1);
    Veins = min(Veins, abs(TempPos.y));     
    
    return exp(-Veins * 30.0);
}

vec3 EyeballTexture(vec2 EyeUV)
{
    vec3 Color = vec3(1.0, 0.95, 0.9);
    
    float BloodNoise = pow(0.3 * texture(bufferA_iChannel1, 3.0 * EyeUV).x + 0.7 * texture(bufferA_iChannel1, 1.0 * EyeUV).x, 0.5);
    BloodNoise = (0.5 * BloodNoise - 0.5);
    float Blood = smoothstep(0.15, 0.45, 0.5 * BloodNoise + 2.0 * length(abs(EyeUV-vec2(0.49, 0.5))));
    
    float VeinMask = VeinTexture(EyeUV) * 0.6 * smoothstep(0.0, 0.15, length(EyeUV-vec2(0.49, 0.5))-0.04);
    
    Color = mix(Color, BLOOD_COLOR, Blood);
    Color = mix(Color, BLOOD_COLOR, VeinMask);
    
    return Color;
}

void RotateEyeTo(float Time, float StartTime, float Speed, vec2 TargetRot)
{    
    float LocalTime = Time - StartTime;    
    float RotDist = length(GEyeRot - TargetRot);
    float RotTime = RotDist / Speed;
    
    // Smooth start and stop
    float T = smoothstep(0.0, 1.0, saturate(LocalTime / RotTime));
    
    // Overshoot and come back
    T += T > 0.0 ? mix(0.1, 0.0, saturate((LocalTime - RotTime) / (0.1))) : 0.0;
    
    GEyeRot = mix(GEyeRot, TargetRot, T);
}

void fillBufferA(out vec4 fragColor, in vec2 fragCoord)
{
    // Scene
    float SceneTime = mod(iTime, 48.0);

    float FadeOutStart = 36.0;
    float FadeInStart = 44.0;
    float FadeInSpeed = 0.25;
    float FadeOutSpeed = 0.25;
	float SceneFade = SceneTime < FadeInStart
    				? smoothstep(0.0, 1.0, saturate(FadeOutSpeed * (SceneTime - FadeOutStart))) 
    				: smoothstep(1.0, 0.0, saturate(FadeOutSpeed * (SceneTime - FadeInStart)));
    float StoneTime = SceneTime < FadeInStart ? max(SceneTime - 15.0, 0.0) : 0.0;
    float StoneFadeTime = 0.2 * StoneTime;
    float AnimTime = SceneTime < FadeInStart ? min(SceneTime, 25.0) : 0.0;
    
	float ZoomIn = smoothstep(8.0, 9.5, SceneTime) * smoothstep(17.0, 16.0, SceneTime);
    
    // Macro eye movement
    float EyeRotTime = mod(AnimTime, 8.0);
	GEyeRot = vec2(0.0, 0.0);
    
    RotateEyeTo(EyeRotTime, 00.4, 0.50, vec2(-0.03, -0.00));
    RotateEyeTo(EyeRotTime, 00.8, 0.50, vec2(+0.04, +0.00));  
    RotateEyeTo(EyeRotTime, 01.2, 0.50, vec2(+0.00, +0.00));
    RotateEyeTo(EyeRotTime, 01.6, 0.50, vec2(-0.02, +0.00));
    
    RotateEyeTo(EyeRotTime, 02.0, 0.90, vec2(-0.15, -0.10));
    RotateEyeTo(EyeRotTime, 02.6, 0.90, vec2(-0.10, -0.10));
    
    RotateEyeTo(EyeRotTime, 03.0, 1.30, vec2(+0.19, -0.08));
    RotateEyeTo(EyeRotTime, 03.5, 1.00, vec2(+0.12, -0.09));
    RotateEyeTo(EyeRotTime, 04.0, 2.00, vec2(-0.20, +0.00));
    RotateEyeTo(EyeRotTime, 05.0, 2.00, vec2(+0.20, -0.05));
    RotateEyeTo(EyeRotTime, 06.0, 0.80, vec2(+0.10, +0.05));
    RotateEyeTo(EyeRotTime, 07.0, 0.25, vec2(+0.07, +0.03));
    RotateEyeTo(EyeRotTime, 07.9, 0.80, vec2(+0.00, +0.00));
    

    // Eyelid movement
    GEyelidRot.x = min(0.0, -2.0 * GEyeRot.y - 0.2);
    GEyelidRot.y = +GEyeRot.y + 0.1;
    
	float BlinkTime = mod(AnimTime - 2.5, 5.0);
    float Blink = smoothstep(0.3, 0.0, BlinkTime);
    GEyelidRot.xy = mix(GEyelidRot.xy, vec2(0.8, 0.8), vec2(Blink));
    
    
    GEyeBrowDown = Blink * 1.5 - 6.0 * GEyeRot.y;
    
    
	vec2 ScreenUV = fragCoord.xy / iResolution.xy;
    vec2 ScreenPos = -1.0 + 2.0 * ScreenUV;
	ScreenPos.x *= iResolution.x / iResolution.y;
    ScreenPos.x *= -1.0;

    float FOV = 3.0;
    float AngleZ = -0.6;
    vec3 RayOrigin = vec3(-0.8, -0.1, -3.5);
    
    AngleZ = mix(AngleZ, -0.1, pow(ZoomIn, 2.0));
    RayOrigin = mix(RayOrigin, vec3(-0.02, -0.05, -2.3), ZoomIn);
    
    RayOrigin.x += sin(1.5 * 3.0 * AnimTime * 0.3) * 0.01 * (1.0 - 0.5 * ZoomIn);
    RayOrigin.y += cos(1.5 * 2.0 * AnimTime * 0.3) * 0.02 * (1.0 - 0.5 * ZoomIn);
	vec3 RayDir = normalize(vec3(ScreenPos.xy, FOV));
    
    Rotate(RayOrigin.xz, AngleZ);
    Rotate(RayDir.xz, AngleZ);    
   

    float DOFMask = 1.0;
    vec3 BackgroundColorA = 0.7 * vec3(0.03, 0.04, 0.04);
    vec3 BackgroundColorB = 0.7 * vec3(0.22, 0.33, 0.39);
    vec3 Background = mix(
        BackgroundColorA, 
        BackgroundColorB, 
        vec3(smoothstep(0.5, 1.0, ScreenUV.x) * smoothstep(0.0, 0.8, ScreenUV.y)));
    vec3 Color = Background;
    
	float t = CastRay(RayOrigin, RayDir);
    if (t > 0.0)
    {                
        vec3 Pos = RayOrigin + t * RayDir;
        vec3 LightDir = normalize(vec3(-0.5, 0.5, -0.5));
        
        float Eps = 0.001;      
        float GeomHeight0 = Scene(Pos);
        float GeomHeightX = Scene(Pos - vec3(Eps, 0.0, 0.0));
		float GeomHeightY = Scene(Pos - vec3(0.0, Eps, 0.0));
        float GeomHeightZ = Scene(Pos - vec3(0.0, 0.0, Eps));
        
        vec3 Normal;
        Normal.x = GeomHeight0 - GeomHeightX;
        Normal.y = GeomHeight0 - GeomHeightY;
        Normal.z = GeomHeight0 - GeomHeightZ;
        Normal = normalize(Normal);        
        
        vec3 ViewDir = -RayDir;
        float NdotV = saturate(dot(Normal, ViewDir));        
        
        vec3 EyePos = Pos;
        Rotate(EyePos.xz, GEyeRot.x);
        Rotate(EyePos.yz, GEyeRot.y);
        
        vec3 SpherePos = normalize(EyePos);
        vec2 EyeUV = vec2(1.0 + (atan(SpherePos.z, SpherePos.x)) / (PI), 0.5 - asin(SpherePos.y) / PI);
        float IrisMask = smoothstep(0.03, 0.0, length(EyeUV-0.5) - 0.06);
        float LimbalRingMask = smoothstep(0.01, 0.0, length(EyeUV-0.5) - 0.075) * smoothstep(-0.02, 0.007, length(EyeUV-0.5) - 0.075);
        
        vec2 IrisUV = EyeUV;
        if (IrisMask > 0.0)
        {
            vec3 IrisRayOrigin = Pos;
                        
            vec3 RefrRayV = refract(RayDir, Normal, 0.5);
            vec3 IrisRayDir = normalize(mix(RayDir, RefrRayV, smoothstep(0.0, 1.0, saturate(length(RefrRayV)))));            
            
            float IrisT = CastIrisRay(IrisRayOrigin, IrisRayDir);
            vec3 IrisPos = IrisRayOrigin + IrisT * IrisRayDir;
            
			vec3 SpherePos = IrisPos;
        	Rotate(SpherePos.xz, GEyeRot.x);
            Rotate(SpherePos.yz, GEyeRot.y);
            SpherePos = normalize(SpherePos);
            IrisUV = vec2(1.0 + (atan(SpherePos.z, SpherePos.x)) / PI, 0.5 - asin(SpherePos.y) / PI);
        }
        
        vec3 IrisGreen = saturate(vec3(0.11, 0.13, 0.06) * 5.0);
        vec3 IrisYellow = saturate(vec3(0.2, 0.13, 0.06) * 5.0);
        
        float IrisYellowMask = smoothstep(0.045, 0.02, length(IrisUV-0.5));
        
        IrisYellowMask = max(IrisYellowMask, smoothstep(0.3, 0.9, texture(bufferA_iChannel1, 5.0 * IrisUV).x));
        
        vec3 IrisBaseColor = vec3(IrisTexture(IrisUV-0.5)) * mix(IrisGreen, IrisYellow, IrisYellowMask);

        float IrisBlackSpotMask = smoothstep(0.055, 0.045, length(IrisUV-0.5)) * smoothstep(0.4, 1.0, texture(bufferA_iChannel1, 20.0 * IrisUV).x);
        IrisBaseColor *= 1.0 - IrisBlackSpotMask;
       
		float Pupil = smoothstep(0.02, 0.0, length(IrisUV-0.5) - 0.018);
        
        vec3 BaseColor = EyeballTexture(EyeUV);
        BaseColor = mix(BaseColor, IrisBaseColor, IrisMask);
        BaseColor = mix(BaseColor, vec3(0.0), max(LimbalRingMask, Pupil));

        // Skin
        float SkinMask = smoothstep(0.00, 0.01, EyeballSDF(Pos));
        
        vec3 SkinBaseColor0 = vec3(0.64, 0.39, 0.30);
        vec3 SkinBaseColor1 = 0.8 * vec3(0.48, 0.18, 0.10);
        
        float SkinSpots = 0.0;
        SkinSpots = max(SkinSpots, texture(bufferA_iChannel0, 1.7 * (Pos.xy + 0.11)).x);
        SkinSpots = max(SkinSpots, texture(bufferA_iChannel0, 1.1 * (Pos.xy + 0.17)).x);
        SkinSpots = max(SkinSpots, texture(bufferA_iChannel0, 0.3 * (Pos.xy + 0.79)).x);
        
        vec3 SkinBaseColor = mix(SkinBaseColor0, SkinBaseColor1, vec3(SkinSpots));
        
		float EyelidCutDist = EyelidCutSDF(Pos);
        float EyelidInteriorMask = smoothstep(0.1, 0.0, EyelidCutDist);
        SkinBaseColor = mix(SkinBaseColor, mix(BLOOD_COLOR, SkinBaseColor0, 0.5), EyelidInteriorMask);        
        
        vec3 StoneBaseColor = mix(
            vec3(0.05), 
            vec3(0.67, 0.57, 0.40), 
            pow(texture(bufferA_iChannel2, 0.5 * Pos.xy + vec2(0.61, 0.49)).xxx, vec3(2.2)));
        
        float Edge0 = StoneFadeTime + length(Pos) - 2.0;
        float Edge1 = Edge0 - 0.02;
        float StoneMaskTex = mix(texture(bufferA_iChannel1, 0.5 * Pos.xy + 0.5).x, texture(bufferA_iChannel2, 0.5 * Pos.xy + 0.5).x, 0.5);
        float StoneMask = smoothstep(Edge0 - 0.5, Edge1 - 0.5, StoneMaskTex);
        float NormalStoneMask = smoothstep(Edge0, Edge1, StoneMaskTex);
        float EyeStoneMask = smoothstep(2.2, 3.5, StoneFadeTime);

        float EyeAOMask = smoothstep(-0.1, 0.00, EyelidCutDist);
        BaseColor *= mix(vec3(1.0), SkinBaseColor1, EyeAOMask);

        SkinBaseColor = mix(SkinBaseColor, StoneBaseColor, StoneMask);
        BaseColor = mix(BaseColor, SkinBaseColor, SkinMask);
        
        float EyeCornerMask = smoothstep(0.05, 0.0, EyeCornerSDF(Pos)) * (1.0 - SkinMask);
        BaseColor = mix(BaseColor, 0.6 * BLOOD_COLOR, EyeCornerMask);

        BaseColor = mix(BaseColor, StoneBaseColor, EyeStoneMask);
        
       	vec3 WetnessNormal = Normal;
        WetnessNormal.xyz += (SmoothNoise(EyeUV.xy * 60.0) - 0.5) * 0.05;
        WetnessNormal = normalize(WetnessNormal);
        
        float SkinHeight0 = SkinHeight(5.0 * Pos.xy, NormalStoneMask);
        float SkinHeightX = SkinHeight(5.0 * (Pos.xy + vec2(Eps, 0.0)), NormalStoneMask);
        float SkinHeightY = SkinHeight(5.0 * (Pos.xy + vec2(0.0, Eps)), NormalStoneMask);
        
        float BumpStrength = mix(0.005, 0.01, NormalStoneMask);
        
        vec3 SkinNormal;
        SkinNormal.x = GeomHeight0 - GeomHeightX + (SkinHeight0 - SkinHeightX) * BumpStrength;
        SkinNormal.y = GeomHeight0 - GeomHeightY + (SkinHeight0 - SkinHeightY) * BumpStrength;
        SkinNormal.z = GeomHeight0 - GeomHeightZ;
        SkinNormal = normalize(SkinNormal);
        
        vec3 BaseNormal = Normal;
        BaseNormal = mix(BaseNormal, SkinNormal, NormalStoneMask);
        WetnessNormal = mix(WetnessNormal, SkinNormal, NormalStoneMask);        
        
        vec3 LayerNormal = mix(Normal, WetnessNormal, 1.0 - IrisMask);
        LayerNormal = normalize(mix(LayerNormal, SkinNormal, SkinMask * (1.0 - EyelidInteriorMask)));
        vec3 Reflection = reflect(RayDir, LayerNormal);
        
        float Shadow = smoothstep(0.0, 1.0, length(Pos - vec3(0.8, 0.1, -0.1)) - 0.6);
        
        Color = BaseColor * (Shadow * saturate(dot(BaseNormal, LightDir)) + vec3(0.1,0.05,0.0));
        //Color = vec3(1.0) * saturate(dot(BaseNormal, LightDir));
        
        float EyeReflectionOcclusion = (1.0 - EyeCornerMask)
            * smoothstep(+0.04, +0.01, EyelidCutDist) 
            * smoothstep(-0.04, -0.01, EyelidCutDist);
        vec3 SpecularColor = vec3(mix(0.04, 0.02, SkinMask * (1.0 - StoneMask)));
        vec3 EnvSpecularColor = EnvBRDFApprox(SpecularColor, 0.0, NdotV) * (1.0 - EyeReflectionOcclusion);
        
        vec3 Env = pow(2.0 * texture(bufferA_iChannel3, Reflection).xxx, vec3(2.2));
        
        Env *= mix(1.0, smoothstep(0.0, 0.2, BaseColor.y), NormalStoneMask);
        Env *= mix(1.0, 0.5, SkinMask * (1.0 - StoneMask));
        
        Color += Shadow * EnvSpecularColor * Env;
        
        
        float EdgeAA = pow(smoothstep(0.5, 1.0, 1.0 + dot(RayDir, Normal)), 1.0);
        EdgeAA *= smoothstep(-0.5, -1.5, Pos.x);
        
        vec3 TempPos = Pos;
        Rotate(TempPos.xy, -0.15 * PI);
        Rotate(TempPos.xz, 0.1 * PI);  
        EdgeAA = max(EdgeAA, smoothstep(0.3, 0.05, TempPos.x + 2.1));

        Color = mix(Color, Background, EdgeAA);
        
        DOFMask = max(smoothstep(2.6, 1.6, t), smoothstep(3.0, 5.1, t));
    }

    // Fade out
	Color = mix(Color, BackgroundColorA, SceneFade);
                    
    fragColor = vec4(Color, DOFMask);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
 	vec2 UV = fragCoord.xy / iResolution.xy;
    
    // Chromatic Abberation
    vec2 CAOffset = (UV - 0.5) * 0.005;

    vec3 Color;
    Color.x = texture(iChannel0, UV - CAOffset).x;
    Color.y = texture(iChannel0, UV).y;
    Color.z = texture(iChannel0, UV + CAOffset).z; 
    
    // Vignette
    float Vignette = UV.x * UV.y * (1.0 - UV.x) * (1.0 - UV.y);
    Vignette = clamp(pow(16.0 * Vignette, 0.3), 0.0, 1.0);
    Color *= 0.5 * Vignette + 0.5;
    
    Color = pow(Color, vec3(1.0 / 2.2));
    
    fragColor = vec4(Color, 1.0);
}
