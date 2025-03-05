// Created by XORXOR, 2016
// Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
// https://www.shadertoy.com/view/Mt3XWH
//
// Replaying mocap data with hand-tuned shape animation for Codevember 2016.
// Inspired by Universal Everything's Walking City
// https://vimeo.com/85596568
// and makio64's Codevember sketches
// http://makiopolis.com/codevember/
//
// Soundtrack by Simon Pyke
// https://soundcloud.com/freefarm/walking-city


// I don't believe there's no better way to read an array with a non-constant index. Is there?
#define returnPos() vec3 p0, p1; for(int i=0; i<NUM_FRAMES-1; i++){ if(i==n) { p0=p[i]; p1=p[i+1]; break;}} return mix( p0, p1, f );

#define NUM_FRAMES 21

vec3 getSpine2( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(0.0368,9.18,-0.389); p[1] = vec3(0.036,9.19,-0.386); p[2] = vec3(0.0326,9.19,-0.367); p[3] = vec3(0.0294,9.2,-0.347); p[4] = vec3(0.0272,9.2,-0.33); p[5] = vec3(0.0276,9.2,-0.334); p[6] = vec3(0.029,9.2,-0.345); p[7] = vec3(0.0307,9.19,-0.358); p[8] = vec3(0.0327,9.19,-0.371); p[9] = vec3(0.0346,9.19,-0.383); p[10] = vec3(0.0363,9.18,-0.389); p[11] = vec3(0.0377,9.19,-0.378); p[12] = vec3(0.0388,9.19,-0.357); p[13] = vec3(0.0395,9.2,-0.337); p[14] = vec3(0.0397,9.2,-0.33); p[15] = vec3(0.0396,9.2,-0.336); p[16] = vec3(0.0394,9.2,-0.347); p[17] = vec3(0.0391,9.19,-0.359); p[18] = vec3(0.0387,9.19,-0.371); p[19] = vec3(0.0381,9.19,-0.381); p[20] = vec3(0.0373,9.19,-0.388);
	returnPos();
}

vec3 getLWrist( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(0.829,9.28,2.77); p[1] = vec3(0.817,9.29,2.74); p[2] = vec3(0.809,9.32,2.55); p[3] = vec3(0.972,9.31,2.36); p[4] = vec3(1.27,9.22,2.12); p[5] = vec3(1.62,9.05,1.83); p[6] = vec3(2,8.84,1.38); p[7] = vec3(2.32,8.63,0.77); p[8] = vec3(2.5,8.45,0.0625); p[9] = vec3(2.52,8.33,-0.505); p[10] = vec3(2.54,8.27,-0.816); p[11] = vec3(2.8,8.25,-0.958); p[12] = vec3(3.29,8.36,-0.936); p[13] = vec3(3.8,8.61,-0.774); p[14] = vec3(4.12,8.92,-0.361); p[15] = vec3(4.16,9.19,0.259); p[16] = vec3(3.88,9.24,1.01); p[17] = vec3(3.35,9.25,1.72); p[18] = vec3(2.67,9.28,2.27); p[19] = vec3(1.93,9.33,2.62); p[20] = vec3(1.21,9.33,2.77);
	returnPos();
}

vec3 getHead( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-0.181,12.1,-0.778); p[1] = vec3(-0.163,12.1,-0.763); p[2] = vec3(-0.0746,12.2,-0.687); p[3] = vec3(0.0121,12.2,-0.6); p[4] = vec3(0.0886,12.2,-0.514); p[5] = vec3(0.138,12.2,-0.444); p[6] = vec3(0.193,12.2,-0.425); p[7] = vec3(0.252,12.2,-0.467); p[8] = vec3(0.303,12.2,-0.564); p[9] = vec3(0.337,12.1,-0.667); p[10] = vec3(0.354,12.1,-0.727); p[11] = vec3(0.349,12.1,-0.647); p[12] = vec3(0.311,12.2,-0.484); p[13] = vec3(0.234,12.2,-0.361); p[14] = vec3(0.141,12.2,-0.317); p[15] = vec3(0.0452,12.2,-0.353); p[16] = vec3(-0.0365,12.2,-0.417); p[17] = vec3(-0.102,12.2,-0.504); p[18] = vec3(-0.149,12.2,-0.603); p[19] = vec3(-0.175,12.2,-0.696); p[20] = vec3(-0.183,12.1,-0.762);
	returnPos();
}

vec3 getSpine1( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(1.75e-08,8.01,-0.0366); p[1] = vec3(1.75e-08,8.01,-0.0366); p[2] = vec3(1.75e-08,8.01,-0.0366); p[3] = vec3(1.75e-08,8.01,-0.0366); p[4] = vec3(1.75e-08,8.01,-0.0366); p[5] = vec3(1.75e-08,8.01,-0.0366); p[6] = vec3(1.75e-08,8.01,-0.0366); p[7] = vec3(1.75e-08,8.01,-0.0366); p[8] = vec3(1.75e-08,8.01,-0.0366); p[9] = vec3(1.75e-08,8.01,-0.0366); p[10] = vec3(1.75e-08,8.01,-0.0366); p[11] = vec3(1.75e-08,8.01,-0.0366); p[12] = vec3(1.75e-08,8.01,-0.0366); p[13] = vec3(1.75e-08,8.01,-0.0366); p[14] = vec3(1.75e-08,8.01,-0.0366); p[15] = vec3(1.75e-08,8.01,-0.0366); p[16] = vec3(1.75e-08,8.01,-0.0366); p[17] = vec3(1.75e-08,8.01,-0.0366); p[18] = vec3(1.75e-08,8.01,-0.0366); p[19] = vec3(1.75e-08,8.01,-0.0366); p[20] = vec3(1.75e-08,8.01,-0.0366);
	returnPos();
}

vec3 getRElbow( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-1.13,9.51,-3.53); p[1] = vec3(-1.18,9.53,-3.52); p[2] = vec3(-1.46,9.69,-3.48); p[3] = vec3(-1.93,9.89,-3.3); p[4] = vec3(-2.52,10.1,-2.89); p[5] = vec3(-3.04,10.3,-2.25); p[6] = vec3(-3.35,10.3,-1.48); p[7] = vec3(-3.4,10.1,-0.669); p[8] = vec3(-3.14,9.84,0.164); p[9] = vec3(-2.68,9.58,0.756); p[10] = vec3(-2.26,9.38,1.03); p[11] = vec3(-2.09,9.17,1.04); p[12] = vec3(-2.13,8.94,0.857); p[13] = vec3(-2.22,8.71,0.521); p[14] = vec3(-2.31,8.53,0.0537); p[15] = vec3(-2.38,8.43,-0.531); p[16] = vec3(-2.43,8.47,-1.25); p[17] = vec3(-2.35,8.65,-2); p[18] = vec3(-2.12,8.93,-2.67); p[19] = vec3(-1.77,9.21,-3.16); p[20] = vec3(-1.36,9.43,-3.45);
	returnPos();
}

vec3 getLShoulder( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(0.905,10.8,-0.53); p[1] = vec3(0.918,10.8,-0.517); p[2] = vec3(0.977,10.8,-0.456); p[3] = vec3(1.03,10.7,-0.407); p[4] = vec3(1.09,10.7,-0.385); p[5] = vec3(1.12,10.6,-0.407); p[6] = vec3(1.16,10.6,-0.472); p[7] = vec3(1.19,10.6,-0.574); p[8] = vec3(1.21,10.6,-0.698); p[9] = vec3(1.22,10.5,-0.807); p[10] = vec3(1.22,10.5,-0.869); p[11] = vec3(1.21,10.5,-0.83); p[12] = vec3(1.2,10.6,-0.723); p[13] = vec3(1.16,10.6,-0.623); p[14] = vec3(1.12,10.7,-0.557); p[15] = vec3(1.07,10.7,-0.529); p[16] = vec3(1.02,10.8,-0.513); p[17] = vec3(0.983,10.8,-0.508); p[18] = vec3(0.95,10.8,-0.514); p[19] = vec3(0.926,10.8,-0.525); p[20] = vec3(0.91,10.8,-0.532);
	returnPos();
}

vec3 getLKnee( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(0.785,3.85,-0.624); p[1] = vec3(0.943,3.84,-0.57); p[2] = vec3(1.27,3.83,0.0486); p[3] = vec3(1.71,4.02,0.562); p[4] = vec3(2.12,4.33,0.882); p[5] = vec3(2.29,4.51,1.01); p[6] = vec3(2.37,4.79,1.31); p[7] = vec3(2.36,5.21,1.7); p[8] = vec3(2.09,5.31,1.97); p[9] = vec3(1.51,5.01,2.09); p[10] = vec3(0.769,4.38,1.67); p[11] = vec3(0.454,4.24,1.45); p[12] = vec3(0.446,4.23,1.41); p[13] = vec3(0.504,3.96,0.814); p[14] = vec3(0.562,3.87,0.356); p[15] = vec3(0.598,3.87,0.156); p[16] = vec3(0.623,3.87,-0.0922); p[17] = vec3(0.654,3.89,-0.36); p[18] = vec3(0.702,3.91,-0.605); p[19] = vec3(0.767,3.9,-0.661); p[20] = vec3(0.819,3.86,-0.619);
	returnPos();
}

vec3 getLElbow( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(2.32,9.16,1.03); p[1] = vec3(2.32,9.11,1.01); p[2] = vec3(2.33,8.88,0.885); p[3] = vec3(2.36,8.66,0.652); p[4] = vec3(2.42,8.46,0.287); p[5] = vec3(2.5,8.37,-0.181); p[6] = vec3(2.53,8.35,-0.795); p[7] = vec3(2.43,8.41,-1.51); p[8] = vec3(2.13,8.54,-2.2); p[9] = vec3(1.78,8.67,-2.65); p[10] = vec3(1.64,8.78,-2.87); p[11] = vec3(1.87,9,-2.92); p[12] = vec3(2.34,9.33,-2.79); p[13] = vec3(2.73,9.63,-2.53); p[14] = vec3(3.05,9.82,-2.18); p[15] = vec3(3.27,9.84,-1.75); p[16] = vec3(3.38,9.68,-1.18); p[17] = vec3(3.33,9.49,-0.559); p[18] = vec3(3.14,9.35,0.0245); p[19] = vec3(2.85,9.27,0.518); p[20] = vec3(2.52,9.21,0.885);
	returnPos();
}

vec3 getRHip( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-0.847,6.35,-0.0933); p[1] = vec3(-0.847,6.35,-0.0938); p[2] = vec3(-0.85,6.35,-0.0979); p[3] = vec3(-0.856,6.36,-0.109); p[4] = vec3(-0.863,6.37,-0.127); p[5] = vec3(-0.87,6.38,-0.149); p[6] = vec3(-0.877,6.38,-0.173); p[7] = vec3(-0.882,6.38,-0.198); p[8] = vec3(-0.886,6.38,-0.222); p[9] = vec3(-0.889,6.37,-0.246); p[10] = vec3(-0.89,6.36,-0.266); p[11] = vec3(-0.89,6.35,-0.277); p[12] = vec3(-0.887,6.34,-0.277); p[13] = vec3(-0.882,6.33,-0.262); p[14] = vec3(-0.875,6.31,-0.24); p[15] = vec3(-0.867,6.3,-0.214); p[16] = vec3(-0.858,6.29,-0.185); p[17] = vec3(-0.851,6.29,-0.157); p[18] = vec3(-0.847,6.3,-0.131); p[19] = vec3(-0.845,6.31,-0.11); p[20] = vec3(-0.846,6.33,-0.0968);
	returnPos();
}

vec3 getRShoulder( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-1.03,10.5,-1.06); p[1] = vec3(-1.02,10.5,-1.05); p[2] = vec3(-0.963,10.6,-1.02); p[3] = vec3(-0.912,10.7,-0.96); p[4] = vec3(-0.873,10.7,-0.88); p[5] = vec3(-0.856,10.8,-0.789); p[6] = vec3(-0.839,10.8,-0.714); p[7] = vec3(-0.818,10.8,-0.668); p[8] = vec3(-0.798,10.8,-0.656); p[9] = vec3(-0.783,10.8,-0.662); p[10] = vec3(-0.775,10.8,-0.666); p[11] = vec3(-0.781,10.8,-0.612); p[12] = vec3(-0.806,10.8,-0.531); p[13] = vec3(-0.852,10.8,-0.485); p[14] = vec3(-0.902,10.7,-0.501); p[15] = vec3(-0.95,10.7,-0.573); p[16] = vec3(-0.988,10.6,-0.664); p[17] = vec3(-1.01,10.6,-0.767); p[18] = vec3(-1.03,10.6,-0.873); p[19] = vec3(-1.04,10.5,-0.967); p[20] = vec3(-1.03,10.5,-1.04);
	returnPos();
}

vec3 getRWrist( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-1.6,8.04,-1.82); p[1] = vec3(-1.65,8.05,-1.84); p[2] = vec3(-1.99,8.08,-1.94); p[3] = vec3(-2.54,8.13,-1.96); p[4] = vec3(-3.3,8.3,-1.75); p[5] = vec3(-3.96,8.55,-1.12); p[6] = vec3(-4.18,8.71,-0.0801); p[7] = vec3(-3.77,8.73,1.11); p[8] = vec3(-2.76,8.78,2.16); p[9] = vec3(-1.57,8.95,2.67); p[10] = vec3(-0.714,9.18,2.71); p[11] = vec3(-0.359,9.3,2.55); p[12] = vec3(-0.396,9.31,2.32); p[13] = vec3(-0.576,9.29,2.02); p[14] = vec3(-0.845,9.19,1.69); p[15] = vec3(-1.15,9,1.32); p[16] = vec3(-1.47,8.65,0.829); p[17] = vec3(-1.72,8.32,0.182); p[18] = vec3(-1.82,8.12,-0.545); p[19] = vec3(-1.78,8.03,-1.19); p[20] = vec3(-1.66,8.03,-1.65);
	returnPos();
}

vec3 getRAnkle( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-0.994,1.24,2.41); p[1] = vec3(-0.587,1.02,2.41); p[2] = vec3(-0.478,0.89,1.7); p[3] = vec3(-0.466,0.664,0.998); p[4] = vec3(-0.4,0.473,0.322); p[5] = vec3(-0.354,0.463,-0.229); p[6] = vec3(-0.338,0.502,-0.778); p[7] = vec3(-0.343,0.586,-1.36); p[8] = vec3(-0.374,0.715,-1.93); p[9] = vec3(-0.412,0.938,-2.51); p[10] = vec3(-0.464,1.33,-3.08); p[11] = vec3(-0.46,1.96,-3.65); p[12] = vec3(-0.48,2.61,-3.87); p[13] = vec3(-0.531,2.99,-3.45); p[14] = vec3(-0.503,2.98,-2.8); p[15] = vec3(-0.587,2.74,-2.06); p[16] = vec3(-0.745,2.89,-1.29); p[17] = vec3(-0.861,2.99,-0.664); p[18] = vec3(-0.918,2.69,-0.0864); p[19] = vec3(-0.881,1.83,0.69); p[20] = vec3(-0.72,1.2,1.79);
	returnPos();
}

vec3 getLHip( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(0.892,6.35,-0.129); p[1] = vec3(0.892,6.35,-0.129); p[2] = vec3(0.889,6.34,-0.125); p[3] = vec3(0.883,6.33,-0.114); p[4] = vec3(0.876,6.32,-0.0966); p[5] = vec3(0.867,6.31,-0.0752); p[6] = vec3(0.857,6.31,-0.0521); p[7] = vec3(0.848,6.31,-0.0284); p[8] = vec3(0.839,6.32,-0.00525); p[9] = vec3(0.83,6.32,0.0163); p[10] = vec3(0.823,6.33,0.0347); p[11] = vec3(0.819,6.34,0.0452); p[12] = vec3(0.822,6.35,0.0448); p[13] = vec3(0.832,6.37,0.0323); p[14] = vec3(0.845,6.38,0.0128); p[15] = vec3(0.858,6.39,-0.0113); p[16] = vec3(0.871,6.4,-0.0381); p[17] = vec3(0.882,6.41,-0.0654); p[18] = vec3(0.89,6.4,-0.0911); p[19] = vec3(0.893,6.38,-0.112); p[20] = vec3(0.893,6.36,-0.126);
	returnPos();
}

vec3 getRKnee( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(-1.22,4.58,1.71); p[1] = vec3(-0.858,4.26,1.37); p[2] = vec3(-0.798,4.27,1.37); p[3] = vec3(-0.742,4.07,0.989); p[4] = vec3(-0.691,3.88,0.358); p[5] = vec3(-0.675,3.85,0.0894); p[6] = vec3(-0.676,3.84,-0.139); p[7] = vec3(-0.683,3.85,-0.405); p[8] = vec3(-0.702,3.88,-0.674); p[9] = vec3(-0.736,3.9,-0.837); p[10] = vec3(-0.799,3.89,-0.849); p[11] = vec3(-0.891,3.88,-0.876); p[12] = vec3(-1.11,3.86,-0.778); p[13] = vec3(-1.54,3.87,-0.371); p[14] = vec3(-1.86,3.98,0.0739); p[15] = vec3(-2.03,4.17,0.562); p[16] = vec3(-2.14,4.56,1.18); p[17] = vec3(-2.13,4.98,1.61); p[18] = vec3(-1.97,5.14,1.84); p[19] = vec3(-1.58,4.88,1.86); p[20] = vec3(-1.2,4.57,1.7);
	returnPos();
}

vec3 getLAnkle( int n, float f )
{
	vec3 p[ NUM_FRAMES ];
	p[0] = vec3(0.645,1.27,-2.85); p[1] = vec3(0.941,1.62,-3.15); p[2] = vec3(1.01,2.52,-3.04); p[3] = vec3(1.03,3.06,-2.54); p[4] = vec3(1.13,3.11,-2.02); p[5] = vec3(1.25,2.87,-1.66); p[6] = vec3(1.35,2.77,-1.08); p[7] = vec3(1.32,2.86,-0.344); p[8] = vec3(1.02,2.5,0.608); p[9] = vec3(0.784,1.73,1.76); p[10] = vec3(0.342,1.1,2.54); p[11] = vec3(-0.128,0.972,2.24); p[12] = vec3(0.0519,0.846,1.48); p[13] = vec3(0.129,0.568,0.847); p[14] = vec3(0.204,0.475,0.281); p[15] = vec3(0.266,0.485,-0.211); p[16] = vec3(0.301,0.534,-0.758); p[17] = vec3(0.355,0.618,-1.31); p[18] = vec3(0.454,0.738,-1.85); p[19] = vec3(0.602,0.952,-2.39); p[20] = vec3(0.721,1.22,-2.78);
	returnPos();
}

vec3 head, spine1, spine2;
vec3 lHip, lKnee, lAnkle, lShoulder, lElbow, lWrist;
vec3 rHip, rKnee, rAnkle, rShoulder, rElbow, rWrist;

float smin( float a, float b )
{
    const float k = 2.9;
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float tRidgeStrength = 0.2;
float tRidgeFreq = 10.0;
vec3 tLatticeRadii = vec3( 0.0 );
float tR = 1.2;

vec2 map( vec3 p )
{
    float plane = abs( p.y + 0.9 );

    vec3 q = mod( p, 1.0 ) - 0.5;
    float lattice = min( length( q.xy ) - tLatticeRadii.x,
                    min( length( q.yz ) - tLatticeRadii.y,
                         length( q.xz ) - tLatticeRadii.z ) );
    
    p.y -= tRidgeStrength * ( 2.0 - p.y * 0.1 ) * ( 1.0 +  sin( p.y * tRidgeFreq ) );

    float d = length( p - head ) - tR;
    d = smin( d, length( p - spine1 ) - tR );
    d = smin( d, length( p - spine2 ) - tR );
    d = smin( d, length( p - lHip ) - tR );
    d = smin( d, length( p - lKnee ) - tR );
    d = smin( d, length( p - lAnkle ) - 1.5 * tR );
    d = smin( d, length( p - lShoulder ) - tR );
    d = smin( d, length( p - lElbow ) - 0.8 * tR );
    d = smin( d, length( p - lWrist ) - 0.7 * tR );
    
    d = smin( d, length( p - rHip ) - tR );
    d = smin( d, length( p - rKnee ) - tR );
    d = smin( d, length( p - rAnkle ) - 1.5 * tR );
    d = smin( d, length( p - rShoulder ) - tR );
    d = smin( d, length( p - rElbow ) - 0.8 * tR );
    d = smin( d, length( p - rWrist ) - 0.7 * tR );
    
    d = max( d, -lattice );
    return ( d < plane ) ? vec2( d, 1.0 ) : vec2( plane, 2.0 );
}

float calcShadow( vec3 ro, vec3 rd, float mint, float maxt )
{
    float t = mint;
    float res = 1.0;
    for ( int i = 0; i < 10; i++ )
    {
        float h = map( ro + rd * t ).x;
        res = min( res, 1.1 * h / t );
        t += h;
        if ( ( h < 0.001 ) || ( t > maxt ) )
        {
            break;
        }
    }
    return clamp( res, 0.0, 1.0 );
}

vec2 trace( vec3 ro, vec3 rd )
{
    const float kTMin = 0.01;
    const float kTMax = 200.0;
    const float kEps = 0.001;

    float t = kTMin;
    vec2 res;
    for ( int i = 0; i < 70; i++ )
    {
        vec3 pos = ro + rd * t;
        res = map( pos );
        if ( ( res.x < kEps ) || ( t > kTMax ) )
        {
            break;
        }
        t += res.x * 0.5;
    }

    if ( t < kTMax )
    {
        return vec2( t, res.y );
    }
    else
    {
        return vec2( -1.0 );
    }
}

vec3 calcNormal( vec3 p )
{
    const vec2 e = vec2( 0.005, 0 );
    float dp = map( p ).x;
    return normalize( vec3( dp - map( p - e.xyy ).x,
                            dp - map( p - e.yxy ).x,
                            dp - map( p - e.yyx ).x ) );
}

mat3 calcCamera( vec3 eye, vec3 target )
{
    vec3 cw = normalize( target - eye );
    vec3 cu = cross( cw, vec3( 0, 1, 0 ) );
    vec3 cv = cross( cu, cw );
    return mat3( cu, cv, cw );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float walkTime = mod( 10.0 * iTime, float( NUM_FRAMES - 1 ) );
    int inTime = int( floor( walkTime ) );
    float fTime = fract( walkTime );
    head = getHead( inTime, fTime );
    spine1 = getSpine1( inTime, fTime );
    spine2 = getSpine2( inTime, fTime );
    lHip = getLHip( inTime, fTime );
    lKnee = getLKnee( inTime, fTime );
    lAnkle = getLAnkle( inTime, fTime );
    lShoulder = getLShoulder( inTime, fTime );
	lElbow = getLElbow( inTime, fTime );
	lWrist = getLWrist( inTime, fTime );
    rHip = getRHip( inTime, fTime );
	rKnee = getRKnee( inTime, fTime );
    rAnkle = getRAnkle( inTime, fTime );
	rShoulder = getRShoulder( inTime, fTime );
	rElbow = getRElbow( inTime, fTime );
	rWrist = getRWrist( inTime, fTime );
    
    float trTime = mod( iTime, 140.0 );
    
    tRidgeStrength = 0.2;
	tRidgeFreq = 0.0;
    tLatticeRadii = vec3( 0.0 );
    tR = 1.2;
    
    if ( trTime < 5.0 )
    {
        tRidgeStrength = 0.2;
		tRidgeFreq = 0.0;
		tLatticeRadii = vec3( 0.0 ); 
    }
    else
    if ( trTime < 10.0 )
    {
        tRidgeFreq = 10.0 * smoothstep( 5.0, 10.0, trTime );  
    }
    else
    if ( trTime < 20.0 )
    {
        tRidgeFreq = 10.0;
    }
    else
    if ( trTime < 40.0 )
    {
        tRidgeFreq = 10.0;
        tLatticeRadii = mix( vec3( 0.0 ), vec3( 0.3, 0.0, 0.0 ),
                       		 smoothstep( 20.0, 40.0, trTime ) );
    }
    else
    if ( trTime < 60.0 )
    {
        tRidgeFreq = 10.0;
        tLatticeRadii  = mix( vec3( 0.3, 0.0, 0.0 ), vec3( 0.0, 0.0, 0.6 ),
                              smoothstep( 40.0, 60.0, trTime ) );
    }
    else
    if ( trTime < 70.0 )
    {
        float t = smoothstep( 60.0, 70.0, trTime );
        tRidgeFreq = 10.0 - 10.0 * t;
        tLatticeRadii = mix( vec3( 0.0, 0.0, 0.6 ), vec3( 0.0 ), t );
        tR = mix( 1.2, 2.25, t ); 
    }
    else
    if ( trTime < 90.0 )
    {
        float t = smoothstep( 70.0, 90.0, trTime );
        tRidgeFreq = 10.0;
        tLatticeRadii = vec3( 0.0 );
        tR = 2.25;
        tRidgeStrength = 0.2 * t;
    }
    else
    if ( trTime < 100.0 )
    {
        float t = smoothstep( 90.0, 100.0, trTime );
        tRidgeFreq = 10.0;
        tLatticeRadii = mix( vec3( 0.0 ), vec3( 0.5 ), t );
        tR = 2.25;
    }
    else
    if ( trTime < 120.0 )
    {
        float t = smoothstep( 100.0, 120.0, trTime );
        tRidgeFreq = 10.0;
        tLatticeRadii = vec3( 0.5 );
        tR = mix( 2.25, 1.2, t );
    }
    else
    if ( trTime < 140.0 )
    {
        float t = smoothstep( 120.0, 140.0, trTime );
        tRidgeFreq = 10.0 - 10.0 * t;
        tLatticeRadii = mix( vec3( 0.5 ), vec3( 0.0 ), t );
    }

    vec2 mo = vec2( 0.95, -0.2 );
    if ( iMouse.z > 0.5 )
    {
        mo = 2.0 * iMouse.xy / iResolution.xy - 1.0;
        mo *= 3.14159 * vec2( 0.4, 0.1 );
    }
    mo += 3.14159 * 0.5;

    vec3 eye = vec3( 40.0 * cos( mo.x ), 30.0 + 20.0 * cos( mo.y ), 40.0 * sin( mo.x ) );
    vec3 target = vec3( 0.0, 6.0, 0.0 );
    
    mat3 cam = calcCamera( eye, target );

 	vec2 uv = ( fragCoord.xy - 0.5 * iResolution.xy ) / iResolution.y;
    vec3 rd = cam * normalize( vec3( uv, 2.0 ) );

    vec3 col = vec3( 1.0 );
    
    vec2 res = trace( eye, rd );
    if ( res.x > 0.0 )
    {
        vec3 pos = eye + rd * res.x;
        vec3 nor = calcNormal( pos );
        vec3 ldir = normalize( vec3( -10.5, 20.8, 24.0 ) );
        
        if ( res.y < 1.5 )
        {
        	col = 0.5 + 0.5 * nor;
            float dif = max( dot( nor, ldir ), 0.0 );
            vec3 ref = reflect( rd, nor );
            float spe = pow( clamp( dot( ref, ldir ), 0.0, 1.0 ), 15.0 );

            col *= ( 0.3 + 0.7 * dif );
            float edge = pow( 1.0 - dot( -rd, nor ), 1.1 );
        	col += 0.8 * edge + spe;
        }
      
        float sh = calcShadow( pos, ldir, 0.1, 30.0 );
        col *= ( 0.5 + sh );
    }
    
	fragColor = vec4( col, 1.0 );
}