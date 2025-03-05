// Fork of "Football" by . https://shadertoy.com/view/llKcR3
// 2019-07-25 21:23:38
// Modified from https://www.shadertoy.com/view/Xds3zN by iq.
//

/*
 Some path funtion: timefly(t) returns a 2d pivot
 pasaR(t) and pasaL(t) modifies time to get 
 initial foot targets when fead to timefly()
 
*/

#define AA 1
# define PI 3.14159265359
# define PHI 1.618033988749895
# define TAU 6.283185307179586
vec3 rightFoot;
vec3 leftFoot;
vec3 rightToe;
vec3 leftToe;
vec3 rightHand;
vec3 leftHand;
vec3 rightFootT;
vec3 leftFootT;
vec3 rightHandT;
vec3 leftHandT;
vec3 rightToeT;
vec3 leftToeT;
vec3 rightE; // Elbow
vec3 leftE;
vec3 rightK;//Knee
vec3 leftK;
vec3 rightH; //Hip
vec3 leftH;
vec3 rightS;// Shoulder
vec3 leftS;
vec3 pelvis;
vec3 torso;
vec3 head;
vec3 target;

	float pasa = 1.; // steps overlaping airtime
	float legmax = .89; // max extention
	float leg = .89+0.005; // actual max length
	float armmax = .7;// max extention
	float arm = .7 +.012;// actual max length
  	float toemax = 1.1;// max extention toe from hip
    float footlift=0.19; //lift height later multiplied by speed


# define PLOTPATH 0
 

//------------------------------------------------------------------
float sdPlane(vec3 p) {
	return p.y;
}
float sdSphere(vec3 p, float s) {
	return length(p) - s;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
	vec3 pa = p - a, ba = b - a;
	float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
	return length(pa - ba * h) - r;
}
float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float pathterrain(float x,float z){
    // Common height function for path and terrain
    return 
        sin(x*.5 )*1.+cos(z*.3 )*0.3
        +cos(x*3.+z )*0.1+sin(x-z*.2 )*0.2
        
        ;}
 vec3 timefly(float t) {
    // main path Called from many places
    t*=.80;
	t += (.125 + sin(t * .125));
	vec3 v =
	vec3(sin(t / 50.) * 20., 0., cos(t / 25.) * 24.) +
		vec3(sin(t / 17.1) * 07., 0., cos(t / 17.1) * 05.) +
		vec3(sin(t / 8.1) * 6., 0., cos(t / 8.1) * 8.) +
		vec3(cos(t / 3.) * 3.,0., sin(t / 3.) * 2.)
        +vec3(cos(t  )*2.,0., sin(t  )*2. );
    v.y=pathterrain(v.x,v.z);
    return v        ;
} 
float pasaR(float x){
return max(x + fract(x + 0.25) * pasa - pasa, floor(x + 0.25) - 0.25) + 0.25;
    //gait function 
}
    
float pasaL(float x){
return max(x + fract(x - 0.25) * pasa - pasa, floor(x - 0.25) + 0.25) + 0.25;
   //gait function 
}



float lpnorm(vec3 p, float s) {
	return pow(
		(
			pow(abs(p.x), s) +
			pow(abs(p.y), s) +
			pow(abs(p.z), s)),
		1.0 / s);
}

 
//------------------------------------------------------------------
vec2 opU(vec2 d1, vec2 d2) {
	return (d1.x < d2.x) ? d1 : d2;
}
float smin(float a, float b, float k)
{
	float h = clamp(.5 + .5*(a-b)/k, 0., 1.);
	return mix(a, b, h) - k*h*(1.-h);
}
vec2 bodyPlan(vec3 pos) {
	float res;
	res =  sdSphere(pos - leftFoot, .07);
	res = min(res, sdSphere(pos - leftHand, .06));
	res = min(res, sdSphere(pos - leftH, .09));
	res = min(res, sdSphere(pos - leftK, .08));
	res = min(res, sdSphere(pos - leftE, .08));
	res = min(res, sdSphere(pos - leftS, .07));	

    

    res = min(res, sdSphere(pos - rightFoot, .07));
	res = min(res, sdSphere(pos - rightHand, .06));
	res = min(res, sdSphere(pos - rightH, .09));
	res = min(res, sdSphere(pos - rightS, .07));
	res = min(res, sdSphere(pos - rightK, .08));
	res = min(res, sdSphere(pos - rightE, .08));
    
    	res = min(res, sdSphere(pos - target, .2));
	//res = min(res, sdSphere(pos - head, .16));

 	
   

    

    
    res = min(res, sdCapsule(pos ,rightToe,rightFoot, .06));
    res = smin(res, sdRoundedCylinder(pos - rightToe, .04, .02, .03 ),0.06 );

    res = min(res, sdCapsule(pos ,rightK,rightFoot, .06));
    res = min(res, sdCapsule(pos ,rightK,rightH, .07));   
    res = min(res, sdCapsule(pos ,rightE,rightHand, .05));
    res = min(res, sdCapsule(pos ,rightE,rightS, .06));
    res = min(res, sdCapsule(pos ,torso,rightS, .08));
    
    res = min(res, sdCapsule(pos ,leftToe,leftFoot, .06));
    res = smin(res, sdRoundedCylinder(pos - leftToe, .04, .02, .03 ),0.06);// todo rotate to grund normal

    res = min(res, sdCapsule(pos ,leftK,leftFoot, .06));
    res = min(res, sdCapsule(pos ,leftK,leftH, .07));   
    res = min(res, sdCapsule(pos ,leftE,leftHand, .05));
    res = min(res, sdCapsule(pos ,leftE,leftS, .06));
    res = min(res, sdCapsule(pos ,torso,leftS, .08));
    
    res = smin(res, sdSphere(pos - torso, .14),0.025);
    res = smin(res, sdSphere(pos - pelvis, .16),0.025);
    
    res = smin(res, sdCapsule(pos ,pelvis,torso, .13),0.025);
	res = min(res, sdCapsule(pos ,head,torso, .02)); 
    
    
    
#if PLOTPATH > 0
    for(int i=PLOTPATH;i>-PLOTPATH/2;i--)
    {
        res = min(res, sdSphere(pos- timefly(iTime+float(i)*0.5), .04));
    }
#endif
    //float x=iTime;
    // res= min(res, sdCapsule( pos, timefly(x),timefly(x+1.) , .06125));
    // res= min(res, sdCapsule( pos, timefly(x)-perpr*-0.25,timefly(x)-perpl*0.25 , .06125));
	return vec2(res, 2.0);
}
vec2 map( in vec3 pos) {
	vec2 res = vec2(pos.y-pathterrain(pos.x,pos.z), 1.0);
	res = opU(res, bodyPlan(pos));
	return res;
}
vec2 castRay( in vec3 ro, in vec3 rd) {
	float tmin = 1.0;
	float tmax = 30.0;
	float t = tmin;
	float m = -1.0;
	for (int i = 0; i < 80; i++) {
		float precis = 0.0001 * t;
		vec2 res = map(ro + rd * t);
		if (res.x < precis || t > tmax) break;
		t += res.x * .7;
		m = res.y;
	}
	if (t > tmax) m = -1.0;
	return vec2(t, m);
}
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax) {
	float res = 1.0;
	float t = mint;
	for (int i = 0; i < 32; i++) {
		float h = map(ro + rd * t).x;
		res = min(res, 8.0 * h / t);
		t += clamp(h, 0.02, 0.10);
		if (res < 0.005 || t > tmax) break;
	}
	return clamp(res, 0.0, 1.0);
}
vec3 calcNormal( in vec3 pos) {
	vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
	return normalize(e.xyy * map(pos + e.xyy).x +
		e.yyx * map(pos + e.yyx).x +
		e.yxy * map(pos + e.yxy).x +
		e.xxx * map(pos + e.xxx).x);
}
float calcAO( in vec3 pos, in vec3 nor) {
	float occ = 0.0;
	float sca = 1.0;
	for (int i = 0; i < 5; i++) {
		float hr = 0.01 + 0.12 * float(i) / 4.0;
		vec3 aopos = nor * hr + pos;
		float dd = map(aopos).x;
		occ += -(dd - hr) * sca;
		sca *= 0.95;
	}
	return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}
// https://iquilezles.org/articles/checkerfiltering
float checkersGradBox2( in vec2 p) {
	// filter kernel
	vec2 w = fwidth(p) + 0.001;
	// analytical integral (box filter)
	vec2 i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
	// xor pattern
	return 0.5 - 0.5 * i.x * i.y;
}
float checkersGradBox( in vec2 p) {
 
	return  checkersGradBox2(p) -checkersGradBox2(p-0.03 )*0.4 ;
}
vec3 render( in vec3 ro, in vec3 rd) {
	vec3 col = vec3(0.7, 0.9, 1.0) + rd.y * 0.8;
	vec2 res = castRay(ro, rd);
	float t = res.x;
	float m = res.y;
	if (m > -0.5) {
		vec3 pos = ro + t * rd;
		vec3 nor = calcNormal(pos);
		vec3 ref = reflect(rd, nor);
		// material        
		col = 0.45 + 0.35 * sin(vec3(0.05, 0.08, 0.10) * (m - 1.0));
		if (m < 1.5) {
			float f = checkersGradBox(1.2 * pos.xz);
			col = 0.3 + f * vec3(0.3);
		}
		if (m >= 2.0) {
			col = vec3(0.6);
		}
		if (m >= 3.0) {
			col = vec3(0.07);
		}
		// lighting        
		float occ = calcAO(pos, nor);
		vec3 lig = normalize(vec3(0.2, 0.7, 0.6));
		vec3 hal = normalize(lig - rd);
		float amb = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
		float dif = clamp(dot(nor, lig), 0.0, 1.0);
		float bac = clamp(dot(nor, normalize(vec3(-lig.x, 0.0, -lig.z))), 0.0, 1.0) * clamp(1.0 - pos.y, 0.0, 1.0);
		float dom = smoothstep(-0.1, 0.1, ref.y);
		float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);
		dif *= calcSoftshadow(pos, lig, 0.02, 2.5);
		dom *= calcSoftshadow(pos, ref, 0.02, 2.5);
		float spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0) *
			dif *
			(0.04 + 0.96 * pow(clamp(1.0 + dot(hal, rd), 0.0, 1.0), 5.0));
		vec3 lin = vec3(0.0);
		lin += 1.30 * dif * vec3(1.00, 0.80, 0.55);
		lin += 0.20 * amb * vec3(0.40, 0.60, 1.00) * occ;
		lin += 0.20 * dom * vec3(0.40, 0.60, 1.00) * occ;
		lin += 0.30 * bac * vec3(0.25, 0.25, 0.25) * occ;
		lin += 0.35 * fre * vec3(1.00, 1.00, 1.00) * occ;
		col = col * lin;
		col += 10.00 * spe * vec3(1.00, 0.90, 0.70);
		col = mix(col, vec3(0.8, 0.9, 1.0), 1.0 - exp(-0.0002 * t * t * t));
	}
	return vec3(clamp(col, 0.0, 1.0));
}
mat3 setCamera( in vec3 ro, in vec3 ta, float cr) {
	vec3 cw = normalize(ta - ro);
	vec3 cp = vec3(sin(cr), cos(cr), 0.0);
	vec3 cu = normalize(cross(cw, cp));
	vec3 cv = normalize(cross(cu, cw));
	return mat3(cu, cv, cw);
}





void setup() {
	float x = iTime   ;//Time manipulations moved to timefly
      
    
    // filter gait slightly for less stabby foot placement, too much generates skating
    float filt=18.;
	float left = 0.025+ mix(pasaR(floor(x*filt)/filt) ,pasaR(ceil(x*filt)/filt), ( fract(x*filt)));
	float right =0.025+ mix(pasaL(floor(x*filt)/filt) ,pasaL(ceil(x*filt)/filt), ( fract(x*filt)));
	
    
    float ahead=1.1;
    vec3 dif = (timefly(x + ahead) - timefly(x))/ahead; //delta x+1
	float speed = length(dif); 
     ahead = clamp(0.8,1.1,1.3-speed);
     dif = (timefly(x + ahead) - timefly(x))/ahead; //delta x+1
	 speed = length(dif); 
    
    
    
    vec3 nextdif = (timefly(x + ahead+.5) - timefly(x + .5))/ahead; 
	vec3 lean = (nextdif - dif*2.); // bank into turns

    //
      ahead=speed;
      dif = (timefly(x + ahead) - timefly(x))/ahead; //delta x+1
	  nextdif = (timefly(x + ahead+.5) - timefly(x + .5))/ahead; 
	  lean = (nextdif - dif*2.); // bank into turns

     
    
	float nextSpeed = length(timefly(x + 1.2) - timefly(x + .2));
   


    vec3 dir = normalize(dif); //Path direction 
    vec3 nextdir = normalize(nextdif); //Path direction 
	vec3 dirr = normalize(timefly(right + 1.) - timefly(right)); //Path direction Foot specific
	vec3 dirl = normalize(timefly(left + 1.) - timefly(left));
    
	vec3 perp = cross(dir,vec3(0,-1,0));// perpendicular to main path
	vec3 perpl = cross(dirl,vec3(0,-1,0));// perpendicular to intervalled step path
	vec3 perpr = cross(dirr,vec3(0,-1,0));
    
    target =(timefly(x+1.5))
               
            +(vec3(0,.4,0)+lean*1.6+dir*0.25 )*(.09/clamp(speed , 0.05, 4.5));// rolling head

       
    target.y=pathterrain(target.x,target.z);// fix for rolling head collision 
    
    target +=
        +( vec3(0,0.14+abs(sin(x*7.)*0.3),0)) ;
  
      
    vec3 tfx= timefly(x)  ; // Pelvis   path
    vec3 tfr= timefly(right) ;//intervalled step path
    vec3 tfl= timefly(left) ; //intervalled step path
    
    // foot lift component
	vec3 leftlift = vec3(0, min(0., sin(x * TAU + 1.57) * footlift * clamp(speed, 0.05, 1.5)), 0);
	vec3 rightlift = vec3(0, min(0., sin(x * TAU - 1.57) * footlift * clamp(speed, 0.05, 1.5)), 0);
 
    
    // setup targets
	rightFootT = tfr + perpr * -0.16 - rightlift;
	leftFootT = tfl + perpl * 0.16 - leftlift;
    rightToeT = tfr  + perpr * -0.19  +dir*0.172 - rightlift*0.6;
	leftToeT = tfl  + perpl * 0.19  +dir*0.172- leftlift*0.7;
    // ground collision feet and toes
    rightFootT.y=max(pathterrain(rightFootT.x,rightFootT.z),    rightFootT.y);
    leftFootT.y=max(pathterrain(leftFootT.x,leftFootT.z),    leftFootT.y);
    rightToeT.y=max(pathterrain(rightToeT.x,rightToeT.z),    rightToeT.y);
    leftToeT.y=max(pathterrain(leftToeT.x,leftToeT.z),    leftToeT.y);
    
    

    
    
	pelvis = tfx 
        + (lean  ) * clamp(nextSpeed, 0.01, .5) * 0.1 // lean into turn
        + vec3(0, .9 + cos(x * TAU * 2.) * 0.02 * speed, 0) // bob u/d with step
		+ dir * 0.1 * (-0.45 + speed) // lean in to run
		+ perpr * sin(x * TAU) * 0.025 * speed // bob l/R with step
 		+ (vec3(0,-1.,0) )*(.02/clamp(speed , 0.15, 4.5))// bend when head is close
;
    // spine component
	vec3 spine = normalize(
		 (lean  ) * clamp(nextSpeed, 0.2, .5) * 0.1 // lean into turn
		+ vec3(0, 0.3 + cos(x * TAU * 2.) * 0.0125 * speed, 0)// bob u/d with step
		+ dir * 0.05 * (-0.25 + nextSpeed)  // lean in to run
        +(vec3(0,-1.,0)+dir)*(.05/clamp(speed , 0.15, 4.5))// bend when head is close
        + perpr * cos(x * TAU) * 0.025 * speed// bob l/R with step
	);
     
    torso = pelvis + spine * 0.3;
    
    

	// Hips
    rightH = pelvis + perp * -0.11 - rightlift * 0.1 - spine * 0.08 + dir * -0.025;
	leftH = pelvis + perp * 0.11 - leftlift * 0.1 - spine * 0.08 + dir * -0.025;
    
    // Feet
	rightFoot = rightH + normalize(rightFootT - rightH) * min(legmax, length(rightFootT - rightH));
	leftFoot = leftH + normalize(leftFootT - leftH) * min(legmax, length(leftFootT - leftH));
	
    rightToe = rightH + normalize(rightToeT - rightH) * min(toemax, length(rightToeT - rightH));
	leftToe = leftH + normalize(leftToeT - leftH) * min(toemax, length(leftToeT - leftH));
    
    // Shoulder
	rightS = torso + perp * -0.2   + spine * 0.05;
	leftS = torso + perp * 0.2  + spine * 0.05;
    
    // Hand Target
    rightHandT=(rightS +  normalize(
			+perpr * -0.06 
			+vec3(0, -0.4, 0) 
			+dir * 0.3 * cos(.25 + x * TAU) * (clamp(speed, 0.0, 2.) * 0.25)
 			) 
            * armmax 
			+vec3(0, 0.2, 0) * clamp(speed - 0.6, 0., 1.) )// lift alittle with speed
        	+( target -rightS)*(1.-smoothstep(0.,1.2,(1.+sin(x*1. ))))*0.3;// reach for head 
    
     leftHandT= (leftS + normalize(
			perpl * 0.06 +
			vec3(0, -0.4, 0) +
			dir * 0.3 * cos(.25 + PI + x * TAU) * (clamp(speed, 0.0, 2.) * 0.25)
 		) * armmax +
		vec3(0, 0.2, 0) * clamp(speed - 0.6, 0., 1.))
       +( target -leftS)*(1.-smoothstep(0.,1.2,(1.+sin(x*1.+PI))))*0.3;
    
       rightHand = rightS + normalize(rightHandT - rightS) * min(armmax, length(rightHandT - rightS));
       leftHand = leftS + normalize(leftHandT - leftS) * min(armmax, length(leftHandT - leftS));
 	
     
        rightHand.y=max(pathterrain(rightHand.x,rightHand.z)+.2,    rightHand.y);
    leftHand.y=max(pathterrain(leftHand.x,leftHand.z)+.2,    leftHand.y);
 
    
    
	head = torso +normalize(
		vec3(0, .27, 0) 
		+ normalize(lean) * clamp(nextSpeed, 0.2, 1.) * 0.05 // lean into torn
		+dir * 0.1 * (-0.35 + clamp(speed, 0.5, 2.)) // lean into run
		+perpr * cos(x * TAU) * 0.025 * clamp(speed, 0.5, 2.)
        +(vec3(0,-1.,0)+dir)*(.07/clamp(speed , 0.05, 4.5))// bend when head is close

       )*0.27;// sway with step
    
    // bendy lims IK
    
	rightE = mix(rightS, rightHand, 0.5) - cross(rightS - rightHand, -normalize(perp - dir * 0.5)) *
		sqrt(max(0.0001, arm * arm - length(rightS - rightHand) * length(rightS - rightHand))) * 0.5;
	leftE = mix(leftS, leftHand, 0.5) - cross(leftS - leftHand, -normalize(perp + dir * 0.5)) *
		sqrt(max(0.0001, arm * arm - length(leftS - leftHand) * length(leftS - leftHand))) * 0.5;
	rightK = mix(rightH, rightFoot, 0.5) - cross(rightH - rightFoot, normalize(perp + dir * 0.25)) *
		sqrt(max(0.0001, leg * leg - length(rightH - rightFoot) * length(rightH - rightFoot))) * 0.5;
	leftK = mix(leftH, leftFoot, 0.5) - cross(leftH - leftFoot, normalize(perp - dir * 0.25)) *
		sqrt(max(0.0001, leg * leg - length(leftH - leftFoot) * length(leftH - leftFoot))) * 0.5;
}


void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	setup();
	vec2 mo = iMouse.xy / iResolution.xy;
	float time = .0 + iTime;
	vec3 tot = vec3(0.0);
 #	if AA > 1
	for (int m = 0; m < AA; m++)
		for (int n = 0; n < AA; n++) {
			// pixel coordinates
			vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
			vec2 p = (-iResolution.xy + 2.0 * (fragCoord + o)) / iResolution.y;
 # else
				vec2 p = (-iResolution.xy + 2.0 * fragCoord) / iResolution.y;
 # endif
			// camera	
			vec3 ta = timefly(time) + vec3(0, 0.7, 0);
			vec3 ro = ta + vec3(-0.5 + 3.5 * cos(0.1 * time + 6.0 * mo.x),
				2.0 + 2.0 * mo.y,
				0.5 + 4.0 * sin(0.1 * time + 6.0 * mo.x));
			// camera-to-world transformation
			mat3 ca = setCamera(ro, ta, 0.0);
			// ray direction
			vec3 rd = ca * normalize(vec3(p.xy, 2.5));
			// render	
			vec3 col = render(ro, rd);
			// gamma
			col = pow(col, vec3(0.4545));
			tot += col;
 # if AA > 1
		}
	tot /= float(AA * AA);
 #	endif
	fragColor = vec4(tot, 1.0);
}
