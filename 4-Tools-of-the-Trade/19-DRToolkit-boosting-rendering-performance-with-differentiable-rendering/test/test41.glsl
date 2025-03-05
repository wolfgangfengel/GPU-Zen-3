// Multiple Transparency - @P_Malin

// Transparency experiment - manually using a stack to handle ray recursion.


// Scene Render

#define kMaxTraceDist 1000.0
#define kFarDist 1100.0

#define MAT_FG_BEGIN 	10

#define PI 3.141592654

///////////////////////////
// Data Storage
///////////////////////////

vec4 LoadVec4( sampler2D sampler, in vec2 vAddr )
{
    return texelFetch( sampler, ivec2(vAddr), 0 );
}

vec3 LoadVec3( sampler2D sampler, in vec2 vAddr )
{
    return LoadVec4( sampler, vAddr ).xyz;
}

bool AtAddress( vec2 p, vec2 c ) { return all( equal( floor(p), floor(c) ) ); }

void StoreVec4( in vec2 vAddr, in vec4 vValue, inout vec4 fragColor, in vec2 fragCoord )
{
    fragColor = AtAddress( fragCoord, vAddr ) ? vValue : fragColor;
}

void StoreVec3( in vec2 vAddr, in vec3 vValue, inout vec4 fragColor, in vec2 fragCoord )
{
    StoreVec4( vAddr, vec4( vValue, 0.0 ), fragColor, fragCoord);
}

///////////////////////////
// Camera
///////////////////////////

struct CameraState
{
    vec3 vPos;
    vec3 vTarget;
    float fFov;
};
    
void Cam_LoadState( out CameraState cam, sampler2D sampler, vec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + vec2(0,0) );
    cam.vPos = vPos.xyz;
    vec4 targetFov = LoadVec4( sampler, addr + vec2(1,0) );
    cam.vTarget = targetFov.xyz;
    cam.fFov = targetFov.w;
}

void Cam_StoreState( vec2 addr, const in CameraState cam, inout vec4 fragColor, in vec2 fragCoord )
{
    StoreVec4( addr + vec2(0,0), vec4( cam.vPos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + vec2(1,0), vec4( cam.vTarget, cam.fFov ), fragColor, fragCoord );    
}

mat3 Cam_GetWorldToCameraRotMatrix( const CameraState cameraState )
{
    vec3 vForward = normalize( cameraState.vTarget - cameraState.vPos );
	vec3 vRight = normalize( cross(vec3(0, 1, 0), vForward) );
	vec3 vUp = normalize( cross(vForward, vRight) );
    
    return mat3( vRight, vUp, vForward );
}

vec2 Cam_GetViewCoordFromUV( const in vec2 vUV )
{
	vec2 vWindow = vUV * 2.0 - 1.0;
	vWindow.x *= iResolution.x / iResolution.y;

	return vWindow;	
}

void Cam_GetCameraRay( const vec2 vUV, const CameraState cam, out vec3 vRayOrigin, out vec3 vRayDir )
{
    vec2 vView = Cam_GetViewCoordFromUV( vUV );
    vRayOrigin = cam.vPos;
    float fPerspDist = 1.0 / tan( radians( cam.fFov ) );
    vRayDir = normalize( Cam_GetWorldToCameraRotMatrix( cam ) * vec3( vView, fPerspDist ) );
}

vec2 Cam_GetUVFromWindowCoord( const in vec2 vWindow )
{
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x *= iResolution.y / iResolution.x;

    return vScaledWindow * 0.5 + 0.5;
}

vec2 Cam_WorldToWindowCoord(const in vec3 vWorldPos, const in CameraState cameraState )
{
    vec3 vOffset = vWorldPos - cameraState.vPos;
    vec3 vCameraLocal;

    vCameraLocal = vOffset * Cam_GetWorldToCameraRotMatrix( cameraState );
	
    vec2 vWindowPos = vCameraLocal.xy / (vCameraLocal.z * tan( radians( cameraState.fFov ) ));
    
    return vWindowPos;
}

///////////////////////////
// Scene
///////////////////////////

struct SceneResult
{
	float fDist;
	int iObjectId;
    vec3 vUVW;
};
    
void Scene_Union( inout SceneResult a, const in SceneResult b )
{
    if ( b.fDist < a.fDist )
    {
        a = b;
    }
}

    
void Scene_Trim( inout SceneResult a, const in SceneResult b )
{
    if ( a.fDist < -b.fDist )
    {
        a.fDist = -b.fDist;
    }
}

SceneResult Scene_GetDistance( const vec3 vPos, const int iInsideObject );    

vec3 Scene_GetNormal(const in vec3 vPos, const int iInsideObject)
{
    const float fDelta = 0.001;
    vec2 e = vec2( -1, 1 );
    
    vec3 vNormal = 
        Scene_GetDistance( vPos + e.yxx * fDelta, iInsideObject ).fDist * e.yxx + 
        Scene_GetDistance( vPos + e.xxy * fDelta, iInsideObject ).fDist * e.xxy + 
        Scene_GetDistance( vPos + e.xyx * fDelta, iInsideObject ).fDist * e.xyx + 
        Scene_GetDistance( vPos + e.yyy * fDelta, iInsideObject ).fDist * e.yyy;
    
    if ( dot( vNormal, vNormal ) < 0.00001 )
    {
        return vec3(0, 1, 0);
    }
    
    return normalize( vNormal );
}    
    
SceneResult Scene_Trace( const in vec3 vRayOrigin, const in vec3 vRayDir, float minDist, float maxDist, int iInsideObject )
{	
    SceneResult result;
    result.fDist = 0.0;
    result.vUVW = vec3(0.0);
    result.iObjectId = -1;
    
	float t = minDist;
	const int kRaymarchMaxIter = 96;
	for(int i=0; i<kRaymarchMaxIter; i++)
	{		
		result = Scene_GetDistance( vRayOrigin + vRayDir * t, iInsideObject );
        t += result.fDist;

        if ( abs(result.fDist) < 0.001 )
		{
			break;
		}		

        if ( ( iInsideObject == -1 ) && (abs(result.fDist) > 0.1) )
        {
            result.iObjectId = -1;
        }
        
        if ( t > maxDist )
        {
            result.iObjectId = -1;
	        t = maxDist;
            break;
        }        
	}
    
    result.fDist = t;


    return result;
}    

float Scene_TraceShadow( const in vec3 vRayOrigin, const in vec3 vRayDir, const in float fMinDist, const in float fLightDist )
{
    //return 1.0;
    //return ( Scene_Trace( vRayOrigin, vRayDir, 0.1, fLightDist ).fDist < fLightDist ? 0.0 : 1.0;
    
	float res = 1.0;
    float t = fMinDist;
    for( int i=0; i<16; i++ )
    {
		float h = Scene_GetDistance( vRayOrigin + vRayDir * t, -1 ).fDist;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.0001 || t>fLightDist ) break;
    }
    return clamp( res, 0.0, 1.0 );    
}

float Scene_GetAmbientOcclusion( const in vec3 vPos, const in vec3 vDir )
{
    float fOcclusion = 0.0;
    float fScale = 1.0;
    for( int i=0; i<5; i++ )
    {
        float fOffsetDist = 0.01 + 1.0*float(i)/4.0;
        vec3 vAOPos = vDir * fOffsetDist + vPos;
        float fDist = Scene_GetDistance( vAOPos, -1 ).fDist;
        fOcclusion += (fOffsetDist - fDist) * fScale;
        fScale *= 0.4;
    }
    
    return clamp( 1.0 - 2.0*fOcclusion, 0.0, 1.0 );
}

///////////////////////////
// Lighting
///////////////////////////
    
struct SurfaceInfo
{
    vec3 vPos;
    vec3 vNormal;
    vec3 vBumpNormal;    
    vec3 vAlbedo;
    vec3 vR0;
    float fSmoothness;
    vec3 vEmissive;
   	float fTransparency;
    float fRefractiveIndex;
};
    
SurfaceInfo Scene_GetSurfaceInfo( const in vec3 vRayOrigin,  const in vec3 vRayDir, SceneResult traceResult, int iInsideObject );

struct SurfaceLighting
{
    vec3 vDiffuse;
    vec3 vSpecular;
};
    
SurfaceLighting Scene_GetSurfaceLighting( const in vec3 vRayDir, in SurfaceInfo surfaceInfo );

float Light_GIV( float dotNV, float k)
{
	return 1.0 / ((dotNV + 0.0001) * (1.0 - k)+k);
}

void Light_Add(inout SurfaceLighting lighting, SurfaceInfo surface, const in vec3 vViewDir, const in vec3 vLightDir, const in vec3 vLightColour)
{
	float fNDotL = clamp(dot(vLightDir, surface.vBumpNormal), 0.0, 1.0);
	
	lighting.vDiffuse += vLightColour * fNDotL;
    
	vec3 vH = normalize( -vViewDir + vLightDir );
	float fNdotV = clamp(dot(-vViewDir, surface.vBumpNormal), 0.0, 1.0);
	float fNdotH = clamp(dot(surface.vBumpNormal, vH), 0.0, 1.0);
    
	float alpha = 1.0 - surface.fSmoothness;
	// D

	float alphaSqr = alpha * alpha;
	float denom = fNdotH * fNdotH * (alphaSqr - 1.0) + 1.0;
	float d = alphaSqr / (PI * denom * denom);

	float k = alpha / 2.0;
	float vis = Light_GIV(fNDotL, k) * Light_GIV(fNdotV, k);

	float fSpecularIntensity = d * vis * fNDotL;    
	lighting.vSpecular += vLightColour * fSpecularIntensity;    
}

void Light_AddPoint(inout SurfaceLighting lighting, SurfaceInfo surface, const in vec3 vViewDir, const in vec3 vLightPos, const in vec3 vLightColour)
{    
    vec3 vPos = surface.vPos;
	vec3 vToLight = vLightPos - vPos;	
    
	vec3 vLightDir = normalize(vToLight);
	float fDistance2 = dot(vToLight, vToLight);
	float fAttenuation = 100.0 / (fDistance2);
	
	float fShadowFactor = Scene_TraceShadow( surface.vPos, vLightDir, 0.1, length(vToLight) );
	
	Light_Add( lighting, surface, vViewDir, vLightDir, vLightColour * fShadowFactor * fAttenuation);
}

void Light_AddDirectional(inout SurfaceLighting lighting, SurfaceInfo surface, const in vec3 vViewDir, const in vec3 vLightDir, const in vec3 vLightColour)
{	
	float fAttenuation = 1.0;
	float fShadowFactor = Scene_TraceShadow( surface.vPos, vLightDir, 0.1, 10.0 );
	
	Light_Add( lighting, surface, vViewDir, vLightDir, vLightColour * fShadowFactor * fAttenuation);
}

vec3 Light_GetFresnel( vec3 vView, vec3 vNormal, vec3 vR0, float fGloss )
{
    float NdotV = max( 0.0, dot( vView, vNormal ) );

    return vR0 + (vec3(1.0) - vR0) * pow( 1.0 - NdotV, 5.0 ) * pow( fGloss, 20.0 );
}

void Env_AddPointLightFlare(inout vec3 vEmissiveGlow, const in vec3 vRayOrigin, const in vec3 vRayDir, const in float fIntersectDistance, const in vec3 vLightPos, const in vec3 vLightColour)
{
    vec3 vToLight = vLightPos - vRayOrigin;
    float fPointDot = dot(vToLight, vRayDir);
    fPointDot = clamp(fPointDot, 0.0, fIntersectDistance);

    vec3 vClosestPoint = vRayOrigin + vRayDir * fPointDot;
    float fDist = length(vClosestPoint - vLightPos);
	vEmissiveGlow += sqrt(vLightColour * 0.05 / (fDist * fDist));
}

void Env_AddDirectionalLightFlareToFog(inout vec3 vFogColour, const in vec3 vRayDir, const in vec3 vLightDir, const in vec3 vLightColour)
{
	float fDirDot = clamp(dot(vLightDir, vRayDir) * 0.5 + 0.5, 0.0, 1.0);
	float kSpreadPower = 2.0;
	vFogColour += vLightColour * pow(fDirDot, kSpreadPower) * 0.25;
}


///////////////////////////
// Rendering
///////////////////////////

vec4 Env_GetSkyColor( const vec3 vViewPos, const vec3 vViewDir );
vec3 Env_ApplyAtmosphere( const in vec3 vColor, const in vec3 vRayOrigin,  const in vec3 vRayDir, const in float fDist, const int iInsideObject );
vec3 FX_Apply( in vec3 vColor, const in vec3 vRayOrigin,  const in vec3 vRayDir, const in float fDist);

struct RayInfo
{
    vec3 vRayOrigin;
    vec3 vRayDir;
    float fStartDist;
    float fLengthRemaining;
    
    float fRefractiveIndex;
    
    int iObjectId;
    float fDist;
    vec3 vColor;
    vec3 vAmount;
    
    int iChild0;
    int iChild1;
};
    
#define RAY_STACK_SIZE 12

struct ArrRayInfo
{
    vec3 broadcastHelper;

    vec3 vRayOrigin[RAY_STACK_SIZE];
    vec3 vRayDir[RAY_STACK_SIZE];
    float fStartDist[RAY_STACK_SIZE];
    float fLengthRemaining[RAY_STACK_SIZE];
    
    float fRefractiveIndex[RAY_STACK_SIZE];
    
    int iObjectId[RAY_STACK_SIZE];
    float fDist[RAY_STACK_SIZE];
    vec3 vColor[RAY_STACK_SIZE];
    vec3 vAmount[RAY_STACK_SIZE];
    
    int iChild0[RAY_STACK_SIZE];
    int iChild1[RAY_STACK_SIZE];
};

void RayInfo_Clear(int i, inout ArrRayInfo arrRayInfo )
{
    RayInfo rayInfo = RayInfo( vec3(0), vec3(0), 0., -1., 1.0, -1, 0., vec3(0), vec3(1), -1, -1 );

    arrRayInfo.vRayOrigin[i] = rayInfo.vRayOrigin;
    arrRayInfo.vRayDir[i] = rayInfo.vRayDir;
    arrRayInfo.fStartDist[i] = rayInfo.fStartDist;
    arrRayInfo.fLengthRemaining[i] = rayInfo.fLengthRemaining;
    arrRayInfo.fRefractiveIndex[i] = rayInfo.fRefractiveIndex;
    arrRayInfo.iObjectId[i] = rayInfo.iObjectId;
    arrRayInfo.fDist[i] = rayInfo.fDist;
    arrRayInfo.vColor[i] = rayInfo.vColor;
    arrRayInfo.vAmount[i] = rayInfo.vAmount;
    arrRayInfo.iChild0[i] = rayInfo.iChild0;
    arrRayInfo.iChild1[i] = rayInfo.iChild1;
}

void RayStack_Reset(inout ArrRayInfo rayStack)
{
    for ( int i=0; i<RAY_STACK_SIZE; i++)
    {
	    RayInfo_Clear( i, rayStack );
    }
}

RayInfo RayStack_Get( int i, ArrRayInfo rayStack )
{
    RayInfo rayInfo;

    rayInfo.vRayOrigin = rayStack.vRayOrigin[i];
    rayInfo.vRayDir = rayStack.vRayDir[i];
    rayInfo.fStartDist = rayStack.fStartDist[i];
    rayInfo.fLengthRemaining = rayStack.fLengthRemaining[i];
    rayInfo.fRefractiveIndex = rayStack.fRefractiveIndex[i];
    rayInfo.iObjectId = rayStack.iObjectId[i];
    rayInfo.fDist = rayStack.fDist[i];
    rayInfo.vColor = rayStack.vColor[i];
    rayInfo.vAmount = rayStack.vAmount[i];
    rayInfo.iChild0 = rayStack.iChild0[i];
    rayInfo.iChild1 = rayStack.iChild1[i];

	return rayInfo;
}

void RayStack_Set( int i, RayInfo rayInfo, inout ArrRayInfo rayStack )
{
    rayStack.vRayOrigin[i] = rayInfo.vRayOrigin;
    rayStack.vRayDir[i] = rayInfo.vRayDir;
    rayStack.fStartDist[i] = rayInfo.fStartDist;
    rayStack.fLengthRemaining[i] = rayInfo.fLengthRemaining;
    rayStack.fRefractiveIndex[i] = rayInfo.fRefractiveIndex;
    rayStack.iObjectId[i] = rayInfo.iObjectId;
    rayStack.fDist[i] = rayInfo.fDist;
    rayStack.vColor[i] = rayInfo.vColor;
    rayStack.vAmount[i] = rayInfo.vAmount;
    rayStack.iChild0[i] = rayInfo.iChild0;
    rayStack.iChild1[i] = rayInfo.iChild1;
}

vec4 Scene_GetColorAndDepth( const in vec3 vInRayOrigin, const in vec3 vInRayDir )
{
	vec3 vResultColor = vec3(0.0);
            
	SceneResult firstTraceResult;
    
    int ix = 0;
    int stackCurrent = 0;
    int stackEnd = 1;

    ArrRayInfo rayStack;
    RayStack_Reset(rayStack);

    //rayStack.broadcastHelper = vInRayOrigin;

    rayStack.vRayOrigin[0] = vInRayOrigin;
    rayStack.vRayDir[0] = vInRayDir;
    rayStack.fStartDist[0] = 0.0;
    rayStack.fLengthRemaining[0] = kMaxTraceDist;
    rayStack.fRefractiveIndex[0] = 1.0;
    rayStack.vAmount[0] = vec3(1.0);
    rayStack.iChild0[0] = -1;
    rayStack.iChild1[0] = -1;
           
	for( int iPassIndex=0; iPassIndex < RAY_STACK_SIZE; iPassIndex++ )
	{	                
        if ( ix >= RAY_STACK_SIZE )
            break;
        
        stackCurrent = clamp(ix, 0, RAY_STACK_SIZE - 1);
        RayInfo rayInfo = RayStack_Get( stackCurrent, rayStack );
        
        if ( rayInfo.fLengthRemaining <= 0.0 )
            continue;
        
        rayInfo.iChild0 = -1;
        rayInfo.iChild1 = -1;
        
    	SceneResult traceResult = Scene_Trace( rayInfo.vRayOrigin, rayInfo.vRayDir, rayInfo.fStartDist, rayInfo.fLengthRemaining, rayInfo.iObjectId );
        
        rayInfo.fDist = traceResult.fDist;
        
        //if ( iPassIndex == 0 )
        //{
          //  firstTraceResult = traceResult;
        //}
        
		rayInfo.vColor = vec3(0.0);
		vec3 vReflectance = vec3(1.0);

		if( traceResult.iObjectId < 0 )
		{
            rayInfo.vColor = Env_GetSkyColor( rayInfo.vRayOrigin, rayInfo.vRayDir ).rgb;
        }
        else
        {
            SurfaceInfo surfaceInfo = Scene_GetSurfaceInfo( rayInfo.vRayOrigin, rayInfo.vRayDir, traceResult, rayInfo.iObjectId );
            SurfaceLighting surfaceLighting = Scene_GetSurfaceLighting( rayInfo.vRayDir, surfaceInfo );
                
            // calculate reflectance (Fresnel)
            float NdotV = clamp( dot(surfaceInfo.vBumpNormal, -rayInfo.vRayDir ), 0.0, 1.0);
			vReflectance = Light_GetFresnel( -rayInfo.vRayDir, surfaceInfo.vBumpNormal, surfaceInfo.vR0, surfaceInfo.fSmoothness );
			
			rayInfo.vColor = (surfaceInfo.vAlbedo * surfaceLighting.vDiffuse + surfaceInfo.vEmissive) * (vec3(1.0) - vReflectance); 
            
			vec3 vReflectAmount = vReflectance;                
            vec3 vTranmitAmount = vec3(1.0) - vReflectance;
            vTranmitAmount *= surfaceInfo.fTransparency;
            
            bool doReflection = true;
            
            // superhack
            //if ( false )
            if ( surfaceInfo.fTransparency > 0.0 )
            {                
                vec3 vTestAmount = vTranmitAmount * rayInfo.vAmount;
                
                if ( (vTestAmount.x + vTestAmount.y + vTestAmount.z) > 0.01 )
                {
                    RayInfo refractRayInfo;

                    refractRayInfo.vAmount = vTranmitAmount;
                    refractRayInfo.vRayOrigin = surfaceInfo.vPos;
                    refractRayInfo.iObjectId = traceResult.iObjectId;
                                        
					refractRayInfo.vRayDir = refract( rayInfo.vRayDir, surfaceInfo.vBumpNormal, rayInfo.fRefractiveIndex / surfaceInfo.fRefractiveIndex );
                    if ( traceResult.iObjectId == rayInfo.iObjectId )
                    {
                        refractRayInfo.iObjectId = -1;
                    }                    
                    /*
                    if ( (rayInfo.fObjectId != -1.) && (traceResult.fObjectId == rayInfo.fObjectId) )
                    {
                        refractRayInfo.fObjectId = -1.;
                        refractRayInfo.vRayDir = refract( rayInfo.vRayDir, surfaceInfo.vBumpNormal, surfaceInfo.fRefractiveIndex );
                    }
                    else
                    {
                        refractRayInfo.vRayDir = refract( rayInfo.vRayDir, surfaceInfo.vBumpNormal, 1.0 / surfaceInfo.fRefractiveIndex );
                    }*/
                    
                    if ( length( refractRayInfo.vRayDir ) > 0.0 )
                    {
                        refractRayInfo.vRayDir = normalize(refractRayInfo.vRayDir);
                        //refractRayInfo.vRayDir.xz *= -1.0;
                        refractRayInfo.fStartDist = abs(0.1 / dot( refractRayInfo.vRayDir, surfaceInfo.vNormal ));            
                        refractRayInfo.fLengthRemaining = rayInfo.fLengthRemaining - traceResult.fDist;                
                        refractRayInfo.fDist = 1.0 - rayInfo.fDist;
                        refractRayInfo.fRefractiveIndex = surfaceInfo.fRefractiveIndex;

                        rayInfo.iChild1 = stackEnd;

                        //if ( stackEnd < RAY_STACK_SIZE )
                        {
                            rayInfo.vColor *= 1.0 - surfaceInfo.fTransparency;        
                            refractRayInfo.vAmount *= surfaceInfo.fTransparency;
                        }
                        RayStack_Set( stackEnd, refractRayInfo, rayStack );                        
                        stackEnd++;
                        stackEnd = clamp(stackEnd, 0, RAY_STACK_SIZE - 1);
                        
                        doReflection = false;
                    }
                    else
                    {
                        vReflectAmount += vTranmitAmount;
                    }
                }
            }
                        
            // Reflect Ray
            if ( doReflection )
            {
                vec3 vTestAmount = vReflectAmount * rayInfo.vAmount;
                
                if ( (vTestAmount.x + vTestAmount.y + vTestAmount.z) > 0.01 )
                {                
                    RayInfo reflectRayInfo;

                    reflectRayInfo.vAmount = vReflectAmount;
                    reflectRayInfo.vRayOrigin = surfaceInfo.vPos;
                    reflectRayInfo.vRayDir = normalize( reflect( rayInfo.vRayDir, surfaceInfo.vBumpNormal ) );
                    reflectRayInfo.iObjectId = rayInfo.iObjectId;
                    reflectRayInfo.fStartDist = abs(0.01 / dot( reflectRayInfo.vRayDir, surfaceInfo.vNormal ));            
                    reflectRayInfo.fLengthRemaining = rayInfo.fLengthRemaining - traceResult.fDist;                
                    reflectRayInfo.fRefractiveIndex = rayInfo.fRefractiveIndex;

                    rayInfo.iChild0 = stackEnd;

                    RayStack_Set( stackEnd, reflectRayInfo, rayStack );
                    stackEnd++;
                    stackEnd = clamp(stackEnd, 0, RAY_STACK_SIZE - 1);
                }
            }
            
            rayInfo.vColor += surfaceLighting.vSpecular * vReflectance;            
        }
        
		RayStack_Set( stackCurrent, rayInfo, rayStack );
                
        ix++;
    }
        
	for( int iStackPos=(RAY_STACK_SIZE-1); iStackPos >= 0; iStackPos-- )
	{	    
        RayInfo rayInfo = RayStack_Get( iStackPos, rayStack );

        if ( rayInfo.fLengthRemaining <= 0.0 )
            continue;

        // Accumulate colors from child rays
        
        if ( rayInfo.iChild0 >= 0 )
        {
	        RayInfo childRayInfo = RayStack_Get( rayInfo.iChild0, rayStack );
            if ( childRayInfo.fDist > 0.0 )
            {
            	rayInfo.vColor += childRayInfo.vAmount * childRayInfo.vColor;
            }
        }

        if ( rayInfo.iChild1 >= 0 )
        {
	        RayInfo childRayInfo = RayStack_Get( rayInfo.iChild1, rayStack );
            if ( childRayInfo.fDist > 0.0 )
            {
	            rayInfo.vColor += childRayInfo.vAmount * childRayInfo.vColor;
            }
        }
        
		rayInfo.vColor = Env_ApplyAtmosphere( rayInfo.vColor, rayInfo.vRayOrigin, rayInfo.vRayDir, rayInfo.fDist, rayInfo.iObjectId );
		rayInfo.vColor = FX_Apply( rayInfo.vColor, rayInfo.vRayOrigin, rayInfo.vRayDir, rayInfo.fDist );
            
		RayStack_Set( iStackPos, rayInfo, rayStack );        
    }    
    
    if ( firstTraceResult.iObjectId >= MAT_FG_BEGIN )
    {
        firstTraceResult.fDist = -firstTraceResult.fDist;
    }
        
    return vec4( rayStack.vColor[0], rayStack.fDist[0] );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////
// Utility Functions
///////////////////////////


/////////////////////////
// Scene Description
/////////////////////////

// Materials

#define MAT_SKY		 	-1
#define MAT_DEFAULT 	0
#define MAT_GLASS	 	1
#define MAT_WINE	 	2
#define MAT_STEEL	 	3
#define MAT_GLOSS_PAINT	 4


SurfaceInfo Scene_GetSurfaceInfo( const in vec3 vRayOrigin,  const in vec3 vRayDir, SceneResult traceResult, int iInsideObject )
{
    SurfaceInfo surfaceInfo;
    
    surfaceInfo.vPos = vRayOrigin + vRayDir * (traceResult.fDist);
    
    surfaceInfo.vNormal = Scene_GetNormal( surfaceInfo.vPos, iInsideObject ); 
    surfaceInfo.vBumpNormal = surfaceInfo.vNormal;
    surfaceInfo.vAlbedo = vec3(1.0);
    surfaceInfo.vR0 = vec3( 0.01 );
    surfaceInfo.fSmoothness = 1.0;
    surfaceInfo.vEmissive = vec3( 0.0 );
    
    surfaceInfo.fTransparency = 0.0;
    surfaceInfo.fRefractiveIndex = 1.0;
        
    if ( traceResult.iObjectId == MAT_DEFAULT )
    {
    	surfaceInfo.vR0 = vec3( 0.02 );
	    surfaceInfo.vAlbedo = textureLod( bufferA_iChannel2, traceResult.vUVW.xz * 0.5, 0.0 ).rgb;
        surfaceInfo.vAlbedo = surfaceInfo.vAlbedo * surfaceInfo.vAlbedo;
                        
    	surfaceInfo.fSmoothness = clamp( 1.0 - surfaceInfo.vAlbedo.r * surfaceInfo.vAlbedo.r * 2.0, 0.0, 1.0);
        
        surfaceInfo.vAlbedo *= 0.25;
        
        /*surfaceInfo.vBumpNormal.x += surfaceInfo.vAlbedo.r;
        surfaceInfo.vBumpNormal.z += surfaceInfo.vAlbedo.g;
        surfaceInfo.vAlbedo = mix( vec3(0,.1,0), vec3(0.1, 0.8, 0.2), surfaceInfo.vAlbedo);
        surfaceInfo.fSmoothness = 0.0;*/
        
        //surfaceInfo.vR0 = vec3(clamp(surfaceInfo.vAlbedo.r - surfaceInfo.vPos.x, 0.0, 1.0));
        
        float fDist = length( surfaceInfo.vPos.xz );
        
        float fCheckerAmount = clamp( 1.0 - fDist * 0.1 + surfaceInfo.vAlbedo.r * 0.5, 0.0, 1.0);
        
        vec3 vChecker;
        float fChecker = step(fract((floor(traceResult.vUVW.x) + floor(traceResult.vUVW.z)) * 0.5), 0.25);
        if ( fChecker > 0.0 )
        {
            vChecker = vec3(1.0);
        }
        else
        {
            vChecker = vec3(0.1);
        }
                
        surfaceInfo.vAlbedo = mix(surfaceInfo.vAlbedo, vChecker, fCheckerAmount);
        surfaceInfo.vR0 = mix(surfaceInfo.vR0, vec3(0.1), fCheckerAmount);
        surfaceInfo.fSmoothness =  mix(surfaceInfo.fSmoothness, 1.0, fCheckerAmount);
        
    }

  /*  if ( traceResult.fObjectId == MAT_STEEL )
    {
	    surfaceInfo.vAlbedo = texture( bufferA_iChannel2, traceResult.vUVW.xz ).rgb;
        surfaceInfo.vAlbedo = surfaceInfo.vAlbedo * surfaceInfo.vAlbedo;
        
    	surfaceInfo.fSmoothness = surfaceInfo.vAlbedo.r;//clamp( surfaceInfo.vAlbedo.r, 0.0, 1.0);                
        
        surfaceInfo.vAlbedo = surfaceInfo.vAlbedo * 0.5;
                        
    	surfaceInfo.vR0 = vec3( surfaceInfo.vAlbedo.r * surfaceInfo.vAlbedo.r * 0.8 );

        vec3 vDirt = texture( bufferA_iChannel3, traceResult.vUVW.xz ).rgb;
        vDirt = vDirt * vDirt;
        surfaceInfo.vAlbedo = vDirt * ( 1.0 - surfaceInfo.vAlbedo.r) * 0.1;//mix( surfaceInfo.vAlbedo, vDirt, 1.0 - surfaceInfo.vAlbedo.r );
        
    }*/
    
    if ( traceResult.iObjectId == MAT_GLOSS_PAINT )
    {
        //float fChecker = step(fract((floor(traceResult.vUVW.x) + floor(traceResult.vUVW.z)) * 0.5), 0.25);
        float fStripe = step( fract( dot( traceResult.vUVW * 5.0, vec3(1.0, 0.2, 0.4) ) ), 0.5 );
        if ( fStripe > 0.0 )
        {
	        surfaceInfo.vAlbedo = vec3(0.1, 0.05, 1.0);
        }
        else
        {
	        surfaceInfo.vAlbedo = vec3(1.0, 0.05, 0.1);
        }        
    }
    
    
    if ( traceResult.iObjectId == MAT_GLASS )
    {
    	surfaceInfo.vR0 = vec3( 0.02 );

        vec3 vAlbedo;
        vAlbedo = textureLod( bufferA_iChannel3, traceResult.vUVW.xy * 2.0, 0.0 ).rgb;
        vAlbedo = vAlbedo * vAlbedo;
        vAlbedo *= vec3(1.0, 0.1, 0.5);
                
        surfaceInfo.vAlbedo = vAlbedo;
        
    	//surfaceInfo.fSmoothness = 0.9;//clamp( 1.0 - surfaceInfo.vAlbedo.r * surfaceInfo.vAlbedo.r * 2.0, 0.0, 1.0);
    	surfaceInfo.fSmoothness = clamp( 1.0 - surfaceInfo.vAlbedo.r * 2.0, 0.0, 0.9);

        surfaceInfo.fTransparency = clamp( surfaceInfo.vPos.y * 5.0 + surfaceInfo.vAlbedo.g * 3.0, 0.0, 1.0);

        if ( surfaceInfo.vPos.x < 0.0 || surfaceInfo.vPos.z < 0.0 )
        {
        	surfaceInfo.vAlbedo = vec3(0.0);                
    		surfaceInfo.fSmoothness = 0.9;
            surfaceInfo.fTransparency = 1.0 - vAlbedo.g * 0.2;
        }
        
        surfaceInfo.fRefractiveIndex = 1.5;
        
        
        float fLipStickDist = length(surfaceInfo.vPos - vec3(0.3, 3.1, -2.9)) - 1.0;
        if ( fLipStickDist < 0.0 )
        {            
            surfaceInfo.fTransparency = clamp(1.0 +fLipStickDist * 2.0- vAlbedo.r, 0.0, 1.0);
            surfaceInfo.vAlbedo = vec3(0.5,0,0) * (1.0 - surfaceInfo.fTransparency);
        }
        
        /*surfaceInfo.vBumpNormal.x += surfaceInfo.vAlbedo.r;
        surfaceInfo.vBumpNormal.z += surfaceInfo.vAlbedo.g;
        surfaceInfo.vAlbedo = mix( vec3(0,.1,0), vec3(0.1, 0.8, 0.2), surfaceInfo.vAlbedo);
        surfaceInfo.fSmoothness = 0.0;*/
    }

    if ( traceResult.iObjectId == MAT_WINE )
    {
    	surfaceInfo.vR0 = vec3( 0.02 );
        surfaceInfo.vAlbedo = vec3(0.0);
                
    	surfaceInfo.fSmoothness = 0.9;
        
        surfaceInfo.fTransparency = 1.0;
        surfaceInfo.fRefractiveIndex = 1.3;        
        
        /*surfaceInfo.vBumpNormal.x += surfaceInfo.vAlbedo.r;
        surfaceInfo.vBumpNormal.z += surfaceInfo.vAlbedo.g;
        surfaceInfo.vAlbedo = mix( vec3(0,.1,0), vec3(0.1, 0.8, 0.2), surfaceInfo.vAlbedo);
        surfaceInfo.fSmoothness = 0.0;*/
    }
    
    if ( traceResult.iObjectId == iInsideObject )
    {
		surfaceInfo.fRefractiveIndex = 1.0;            
    }
    
    return surfaceInfo;
}

// Scene Description
float GetDistanceMug( const in vec3 vPos )
{
	float fDistCylinderOutside = length(vPos.xz) - 1.0;
	float fDistCylinderInterior = length(vPos.xz) - 0.9;
	float fTop = vPos.y - 1.0;
       
	float r1 = 0.6;
	float r2 = 0.15;
	vec2 q = vec2(length(vPos.xy + vec2(1.2, -0.1))-r1,vPos.z);
	float fDistHandle = length(q)-r2;
       
	float fDistMug = max(max(min(fDistCylinderOutside, fDistHandle), fTop), -fDistCylinderInterior);
	return fDistMug;
}

float SmoothMin( float a, float b, float k )
{
	//return min(a,b);
	
	
    //float k = 0.06;
	float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
	return mix( b, a, h ) - k*h*(1.0-h);
}

float GetDistanceWine( vec3 vPos )
{
    vec3 vLocalPos = vPos;
    vLocalPos.y -= 2.0;
    
    vec2 vPos2 = vec2(length(vLocalPos.xz), vLocalPos.y);
    
    vec2 vSphOrigin = vec2(0);
    vec2 vSphPos = vPos2 - vSphOrigin;   
    
    float fBowlDistance = length( vSphPos ) -  0.6 + 0.01;
    
    vec3 vWaterNormal = vec3(0,1,0);
    
    vWaterNormal.x = sin( iTime * 5.0) * 0.01;
    vWaterNormal.z = cos( iTime * 5.0) * 0.01;
    
    vWaterNormal = normalize( vWaterNormal );
    float fWaterLevel = dot(vLocalPos, vWaterNormal) - 0.1;
        
    return max( fBowlDistance, fWaterLevel );
}

float GetDistanceWineGlass( vec3 vPos )
{
    vec2 vPos2 = vec2(length(vPos.xz), vPos.y);
    
    vec2 vSphOrigin = vec2(0,2.0);
    vec2 vSphPos = vPos2 - vSphOrigin;
    
    vec2 vClosest = vSphPos;
    
    if ( vClosest.y > 0.3 ) vClosest.y = 0.3;
    vClosest = normalize(vClosest) * 0.6;
    
    float fBowlDistance = distance( vClosest, vSphPos ) - 0.015;
    
    vec2 vStemClosest = vPos2;
    vStemClosest.x = 0.0;    
    vStemClosest.y = clamp(vStemClosest.y, 0.0, 1.35);
    
    float fStemRadius = vStemClosest.y - 0.5;
    fStemRadius = fStemRadius * fStemRadius * 0.02 + 0.03;
    
    float fStemDistance = distance( vPos2, vStemClosest ) - fStemRadius;
    
    
    vec2 norm = normalize( vec2( 0.4, 1.0 ) );
    vec2 vBaseClosest = vPos2;
    float fBaseDistance = dot( vPos2 - vec2(0.0, 0.1), norm ) - 0.2;
    fBaseDistance = max( fBaseDistance, vPos2.x - 0.5 ); 

    float fDistance = SmoothMin(fBowlDistance, fStemDistance, 0.2);
    fDistance = SmoothMin(fDistance, fBaseDistance, 0.2);
    
    fDistance = max( fDistance, vSphPos.y - 0.5 );
        
    return fDistance;
}

float GetDistanceBowl( vec3 vPos )
{    
    vec2 vPos2 = vec2(length(vPos.xz), vPos.y);
    
    vec2 vSphOrigin = vec2(0,1.0 - 0.3 + 0.03);
    vec2 vSphPos = vPos2 - vSphOrigin;
    
    vec2 vClosest = vSphPos;
    
    if ( vClosest.y > 0.1 ) vClosest.y = 0.1;
    if ( vClosest.y < -0.7 ) vClosest.y = -0.7;
    
    float r = sqrt( 1.0 - vClosest.y * vClosest.y);    
    vClosest.x = r;        
    
    float fBowlDistance = distance( vClosest, vSphPos );   
    
    vClosest = vSphPos;
    vClosest.y = -0.7;    
    r = sqrt( 1.0 - vClosest.y * vClosest.y);    
    vClosest.x = min( vClosest.x, r ); 
    
    float fBaseDistance = distance( vClosest, vSphPos );
    
    fBowlDistance = min( fBowlDistance, fBaseDistance );   
    
    return fBowlDistance- 0.03;
}


SceneResult Scene_GetDistance( const vec3 vPos, const int iInsideObject )
{
    SceneResult result;
    
    vec3 vWineGlassPos = vec3(0.0, 0.0, -2.0);
    vec3 vBowlPos = vec3(1.0, 0.0, 1.0 );

	result.fDist = vPos.y;
    result.vUVW = vPos;
	result.iObjectId = MAT_DEFAULT;

        
    SceneResult sphereResult1;
    
    
    vec3 vSphere1Pos = vBowlPos + vec3(0.4, 0.5, 0.2);
    sphereResult1.vUVW = vPos - vSphere1Pos;
    sphereResult1.fDist = min( result.fDist, length(vPos - vSphere1Pos) - 0.4);
	sphereResult1.iObjectId = MAT_GLOSS_PAINT;
    Scene_Union( result, sphereResult1 );    
    

    vec3 vSphere2Pos = vec3(2.2, 0.5, -0.9);
    SceneResult sphereResult2;
    sphereResult2.vUVW = (vPos - vSphere2Pos).zyx;
	sphereResult2.fDist = length(vPos - vSphere2Pos) - 0.5;
	sphereResult2.iObjectId = MAT_GLOSS_PAINT;
    Scene_Union( result, sphereResult2 );
    
    if ( result.fDist > 10.0 )
    {
        result.iObjectId = MAT_SKY;
    }
    
    SceneResult wineResult;
    wineResult.vUVW = vPos;
	wineResult.iObjectId = MAT_WINE;    
    wineResult.fDist = GetDistanceWine( vPos - vWineGlassPos );
    
    
    float fRadius = 1.0;
    float fHeight = 1.0;
    
    SceneResult glassResult;
    glassResult.iObjectId = MAT_GLASS;
    glassResult.fDist = length(vPos - vec3(-2.0,fHeight,1.0)) - fRadius;
    
    glassResult.fDist = min(glassResult.fDist, GetDistanceBowl( vPos - vBowlPos));
        
    glassResult.vUVW = vPos.xzy;
    glassResult.fDist = min( glassResult.fDist, GetDistanceWineGlass(vPos - vWineGlassPos ) );    

    //glassResult.fDist = min( glassResult.fDist, GetDistanceMug(vPos - vec3(-2.0, 1.0, -2.0) ) );    

    Scene_Trim( wineResult, glassResult );
    wineResult.fDist -= 0.0001;
    
    if ( iInsideObject == MAT_GLASS )
    {
        glassResult.fDist = -glassResult.fDist;
    }

    if ( iInsideObject == MAT_WINE )
    {
        wineResult.fDist = -wineResult.fDist;
    }
        
    Scene_Union( result, glassResult );
    Scene_Union( result, wineResult );
    
    return result;
}



// Scene Lighting

vec3 g_vSunDir = normalize(vec3(1.0, 0.4, 0.5));
vec3 g_vSunColor = vec3(1, 0.7, 0.5) * 5.0;
vec3 g_vAmbientColor = vec3(0.8, 0.2, 0.1);

SurfaceLighting Scene_GetSurfaceLighting( const in vec3 vViewDir, in SurfaceInfo surfaceInfo )
{
    SurfaceLighting surfaceLighting;
    
    surfaceLighting.vDiffuse = vec3(0.0);
    surfaceLighting.vSpecular = vec3(0.0);    
    
    Light_AddDirectional( surfaceLighting, surfaceInfo, vViewDir, g_vSunDir, g_vSunColor );
    
    //Light_AddPoint( surfaceLighting, surfaceInfo, vViewDir, vec3(1.4, 1.0, 5.8), vec3(1,1,1) );
    
    float fAO = Scene_GetAmbientOcclusion( surfaceInfo.vPos, surfaceInfo.vNormal );
    // AO
    surfaceLighting.vDiffuse += fAO * (surfaceInfo.vBumpNormal.y * 0.5 + 0.5) * g_vAmbientColor;
    
    return surfaceLighting;
}

// Environment

vec4 Env_GetSkyColor( const vec3 vViewPos, const vec3 vViewDir )
{
	vec4 vResult = vec4( 0.0, 0.0, 1.0, kFarDist );
    
    vec3 vEnvMap = textureLod( bufferA_iChannel1, vViewDir, 0.0 ).rgb;
    vEnvMap = vEnvMap * vEnvMap;
    float kEnvmapExposure = 0.999;
    vResult.rgb = -log2(1.0 - vEnvMap * kEnvmapExposure);
        
    /*
	vec4 vResult = vec4( 0.0, 0.0, 0.0, kFarDist );
	
    float fElevation = atan( vViewDir.y, length(vViewDir.xz) );
    float fHeading = atan( vViewDir.x, vViewDir.z );
    
    float fSkyElevationMin = -PI * 0.125;
    float fSkyElevationMax = PI * 0.5;

    float fScaledElevation = 0.5 * ((fElevation - fSkyElevationMin) / (fSkyElevationMax - fSkyElevationMin));
    if (fHeading < 0.0) fScaledElevation = 1.0 - fScaledElevation;
    vec2 vUV = vec2( fract(fHeading / PI), fScaledElevation );
    
    vResult = texture( bufferA_iChannel3, vUV );
    
    //vResult = mix( vec3(0.02, 0.04, 0.06), vec3(0.1, 0.3, 0.8) * 3.0, vViewDir.y * 0.5 + 0.5 );
	
    // Sun
    float NdotV = dot( g_vSunDir, vViewDir );
    vResult.rgb += smoothstep( cos(radians(.7)), cos(radians(.5)), NdotV ) * g_vSunColor * 5000.0;
    
	*/

    return vResult;	
}

float Env_GetFogFactor(const in vec3 vRayOrigin,  const in vec3 vRayDir, const in float fDist, const float fInsideObject)
{    
	float kFogDensity = 0.00001;
	return exp(fDist * -kFogDensity);	
}

vec3 Env_GetFogColor(const in vec3 vDir, const float fInsideObject)
{    
	return vec3(0.2, 0.5, 0.6) * 2.0;		
}

vec3 Env_ApplyAtmosphere( const in vec3 vColor, const in vec3 vRayOrigin,  const in vec3 vRayDir, const in float fDist, const int iInsideObject )
{
    //return vColor;
    vec3 vResult = vColor;
    
    /*
	float fFogFactor = Env_GetFogFactor( vRayOrigin, vRayDir, fDist, fInsideObject );
	vec3 vFogColor = Env_GetFogColor( vRayDir, fInsideObject );	
	Env_AddDirectionalLightFlareToFog( vFogColor, vRayDir, g_vSunDir, g_vSunColor * 3.0);    
    vResult = mix( vFogColor, vResult, fFogFactor );
	*/
    
    // Glass extinction
    if ( iInsideObject == MAT_GLASS || iInsideObject == MAT_WINE )
    {
        vec3 vExtCol = vec3(0);

        if ( iInsideObject == MAT_WINE )
        {
            vExtCol = vec3(1.0) - vec3(1.0, 0.5, 0.01);
        }
        else
        {
            if ( vRayOrigin.z > 0.0 )
            {
                if ( vRayOrigin.x < 0.0 )
                {
                    vExtCol = vec3(1.0) - vec3(0.01, 0.01, 1.0);
                }
                else
                {
                    vExtCol = vec3(1.0) - vec3(1.0, 0.2, 0.8);
                    vExtCol *= 20.0;
                }
            }
        }
        
		vResult *= exp(fDist * -vExtCol);	
    }
    

    return vResult;	    
}


vec3 FX_Apply( in vec3 vColor, const in vec3 vRayOrigin,  const in vec3 vRayDir, const in float fDist)
{    
    return vColor;
}

float fPlaneInFocus = 5.0;

float GetCoC( float fDistance, float fPlaneInFocus )
{
	// http://http.developer.nvidia.com/GPUGems/gpugems_ch23.html

    float fAperture = 0.05;
    float fFocalLength = 0.8;
  
	return abs(fAperture * (fFocalLength * (fDistance - fPlaneInFocus)) /
          (fDistance * (fPlaneInFocus - fFocalLength)));  
}

vec4 MainCommon( vec3 vRayOrigin, vec3 vRayDir, float fShade )
{
	vec4 vColorLinAmdDepth = Scene_GetColorAndDepth( vRayOrigin, vRayDir );    
    vColorLinAmdDepth.rgb = max( vColorLinAmdDepth.rgb, vec3(0.0) );
        
    vec4 vFragColor = vColorLinAmdDepth;
    
    vFragColor.rgb *= fShade;
    
    vFragColor.a = GetCoC( vColorLinAmdDepth.w, fPlaneInFocus );    
    
    return vFragColor;
}

float GetVignetting( const in vec2 vUV, float fScale, float fPower, float fStrength )
{
	vec2 vOffset = (vUV - 0.5) * sqrt(2.0) * fScale;
	
	float fDist = max( 0.0, 1.0 - length( vOffset ) );
    
	float fShade = 1.0 - pow( fDist, fPower );
    
    fShade = 1.0 - fShade * fStrength;

	return fShade;
}

void fillBufferA( out vec4 vFragColor, in vec2 vFragCoord )
{
    vec2 vUV = vFragCoord.xy / iResolution.xy; 

    CameraState cam;
    
    float fAngle = (iMouse.x / iResolution.x) * radians(360.0);
    float fElevation = (iMouse.y / iResolution.y) * radians(90.0);

    if ( iMouse.x <= 0.0 )
    {
        fAngle = -2.3;
        fElevation = 0.5;
    }
    
    float fDist = 6.0;
    
    cam.vPos = vec3(sin(fAngle) * fDist * cos(fElevation),sin(fElevation) * fDist,cos(fAngle) * fDist * cos(fElevation));
    cam.vTarget = vec3(0,0.5,0);
    cam.fFov = 20.0;
    
    vec3 vRayOrigin, vRayDir;
    Cam_GetCameraRay( vUV, cam, vRayOrigin, vRayDir );
 
    float fShade = GetVignetting( vUV, 0.7, 2.0, 1.0 );
    
    vFragColor= MainCommon( vRayOrigin, vRayDir, fShade );
}

// Multiple Transparency - @P_Malin
// @P_Malin


vec3 Tonemap( vec3 x )
{
    float a = 0.010;
    float b = 0.132;
    float c = 0.010;
    float d = 0.163;
    float e = 0.101;

    return ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e );
}

vec3 ColorGrade( vec3 vColor )
{
    vec3 vHue = vec3(1.0, .7, .2);
    
    vec3 vGamma = 1.0 + vHue * 0.6;
    vec3 vGain = vec3(.9) + vHue * vHue * 8.0;
    
    vColor *= 2.0;
    
    float fMaxLum = 100.0;
    vColor /= fMaxLum;
    vColor = pow( vColor, vGamma );
    vColor *= vGain;
    vColor *= fMaxLum;  
    return vColor;
}

// Depth of field pass

#define BLUR_TAPS 32

float fGolden = 3.141592 * (3.0 - sqrt(5.0));

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 vUV = fragCoord.xy / iResolution.xy;

    vec4 vSample = textureLod( iChannel0, vUV, 0.0 ).rgba;
	float fCoC = vSample.w;
    	
	vec3 vResult = vSample.rgb;
    float fTot = 0.0;
        
    vec2 vangle = vec2(0.0, fCoC); // Start angle
    
    if ( abs(fCoC) > 0.0 )
    {
        vResult.rgb  *= fCoC;
        fTot += fCoC;

        float fBlurTaps = float(BLUR_TAPS);

        for(int i=1; i<BLUR_TAPS; i++)
        {
            // http://blog.marmakoide.org/?p=1
            float t = float(i) / fBlurTaps;
            float fTheta = t * fBlurTaps * fGolden;
            float fRadius = fCoC * sqrt( t * fBlurTaps ) / sqrt( fBlurTaps );        

            vec2 vTapUV = vUV + vec2( sin(fTheta), cos(fTheta) ) * fRadius;

            vec4 vTapSample = textureLod( iChannel0, vTapUV, 0.0 ).rgba;
            {
                float fCoC2 = vTapSample.w;
                float fWeight = max( 0.001, fCoC2 );

                vResult += vTapSample.rgb * fWeight;
                fTot += fWeight;
            }
        }
        vResult /= fTot;
    }
        
	fragColor = vec4(vResult, 1.0);    
    
    float fExposure = 3.0;    
    
    fragColor.rgb = fragColor.rgb * fExposure;
    
    fragColor.rgb = ColorGrade( fragColor.rgb );
        
    fragColor.rgb = Tonemap( fragColor.rgb );
}
