// It's time for another episode of your favorite show :
// -----------------------------------------------------
// E P I C         S P A C E         A D V E N T U R E S
//         ...with Rangiroa and the Commander !
//
// EPISODE 457 :  "Giant Ventifacts Of Calientis V"
//
// -----------------------------------------------------
//
// In the last episode, the Commander has finally found the
// Lost City Of Sandara ! But the place is dead. It's
// been abandonned for centuries. Crushed by the discovery,
// with no hope left, our hero repairs a rocket-powered
// ground-effect vehicle to cross the Great Desert of
// Calientis V : 5000 miles of scorching hot sand mixed with
// salts and sulfur. Meanwhile, our favorite robot girl
// has been captured by the Consortium ! Rangiroa, Queen
// of the Space Pirates, Lunar Lady of Tycho, is being
// brought before the evil Dr Zen for interrogation...
//
// Will our hero save her ? Time is running out. And
// the Great Desert is a very old place, full of weird
// ruins, unspeakable madness, floating temples, giant
// ventifacts, fractal mandeltraps, ghosts, and
// blood-thirsty horrors from a billion years ago...
//
// [Opening] starts !

// Technical notes :
//
// Wanted to make "mesas". Ended up doing strange looking
// ventifacts. Then added space art items. It all started
// coming together.
// "I like it when no plan comes together..."
// I honestly don't now what to say except that this is
// more "painting" than real actual "coding", and that I
// enjoy this process tremendously.
// The more time goes, the more I understand why Iq opened
// that can of worms a long time ago. So thank you for that.
//
// Read the real adventures of Rangiroa and the Commander
// here :

// https://baselunaire.fr/?page_id=554

// 18 episodes already. The concept : a fake lunar radio show
// around 2035 that presents and promotes real Demoscene musics.
// Why ? Well because the Scene is great, and Scene musicians
// are the best, and we should talk about it more often,
// THAT'S WHY !

// The music for this shader is "A Defender Rises" by Skaven252.
// You can download or listen to it on Soundcloud here :
// https://soundcloud.com/skaven252/a-defender-rises
// This guy rules. Seriously.

// Feel free to use this shader to illustrate your latest scifi story
// online, make a video for a music you just composed, or anything.
// Just give proper credit, and provide a link to this page.
// Creative Commons 3.0, etc...

// 01/09/2022 : Added bluish tint to anamorphic streaks, as suggested
//              by leGend/ADAPT (Thanks !)

mat2 r2d( float a ){ float c = cos(a), s = sin(a); return mat2( c, s, -s, c ); }
float noise(vec2 st) { return fract( sin( dot( st.xy, vec2(12.9898,78.9)))*43758.5453123 ); }

#define TimeVar 1.5f*iTime // Let you easily tweak the global speed

// Basic Geometry Functions.

float sdCircle(in vec2 p, float radius, vec2 pos, float prec)
{
      return smoothstep(0.0,prec,radius - length(pos-p));
}

// This belongs to Iq...
float sdTriangle( in vec2 p, in vec2 p0, in vec2 p1, in vec2 p2 )
{
      vec2 e0 = p1-p0, e1 = p2-p1, e2 = p0-p2;
      vec2 v0 = p -p0, v1 = p -p1, v2 = p -p2;
      vec2 pq0 = v0 - e0*clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 );
      vec2 pq1 = v1 - e1*clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 );
      vec2 pq2 = v2 - e2*clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 );
      float s = sign( e0.x*e2.y - e0.y*e2.x );
      vec2 d = min(min(vec2(dot(pq0,pq0), s*(v0.x*e0.y-v0.y*e0.x)),
                       vec2(dot(pq1,pq1), s*(v1.x*e1.y-v1.y*e1.x))),
                       vec2(dot(pq2,pq2), s*(v2.x*e2.y-v2.y*e2.x)));
      return -sqrt(d.x)*sign(d.y);
}

// This belongs to a nice shadertoy coder whose name I lost.
// Please tell me if you read this !
float metaDiamond(vec2 p, vec2 pixel, float r, float s)
{
      vec2 d = abs(r2d(s*TimeVar)*(p-pixel));
      return r / (d.x + d.y);
}

// That's it guys, everything else is mine, as you can
// see by the sudden drop in quality. :D

vec4 drawAtmoGradient(in vec2 v_p)
{
      return mix( vec4(0.0,0.3,0.7,1.0), vec4(1.0,0.8,0.7,1.0), 1.0-v_p.y + 0.7);
}

// Ultra-super-simplified 1D noise with smooth please ?
// We don't need more, really !
float fbm(in vec2 v_p)
{
      float VarX1 = 0.0;
      float VarX2 = 0.0;
      float VarD0 = 0.0;
      float VarS1 = 0.0;
      float Amplitude = 1.0/2.0;
      float Periode   = 2.0;
      VarX1 = Amplitude*floor( Periode*v_p.x);
      VarX2 = Amplitude*floor( Periode*v_p.x + 1.0);
      VarD0 = fract( Periode*v_p.x);
      VarS1 += mix( noise(vec2(VarX1)), noise(vec2(VarX2)), smoothstep( 0.0, 1.0, VarD0));
      return VarS1;
}

float GetMesaMaxHeight(in vec2 v_p)
{
      float MH = 0.98
               + 0.06*fbm(vec2(5.0*v_p.x + 0.25*TimeVar))
               + 0.02*fbm(vec2(40.0*v_p.x + 2.0*TimeVar));
      float Offset = 0.0;
      if( fbm(vec2(10.0*v_p.x + 0.5*TimeVar)) > 0.30 )
          Offset = -0.75*(fbm(vec2(10.0*v_p.x + 0.5*TimeVar)) - 0.30);
      MH += Offset;
      return MH;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
     vec2 p = vec2( (iResolution.x/iResolution.y)*(fragCoord.x - iResolution.x/2.0) / iResolution.x,
                    fragCoord.y / iResolution.y);

     // Making the mouse interactive for absolutely no reason
     float TiltX = -0.001*(iMouse.x - iResolution.x/2.0);
     float AltiY =  0.005*(iMouse.y - iResolution.y/2.0);

     // Propagating user-induced chaos...
     p = p * (1.2 - 0.1*AltiY);
     p *= r2d(TiltX);

     // This, gentlemen, is our World : a single vector.
     // Don't tell Elon, he's gonna freak out.
     vec4 col = vec4(0.0,0.0,0.0,1.0);

     // Here's an atmosphere so you can choke...
     col = drawAtmoGradient(p + vec2(0.0,0.75));

     // For 25 TimeVars, make the screen ondulate
     // Like it's hot in the desert or something...
     // Use iTime instead of TimeVar because whatever
     // the speed of the desert, heat distorsion should stay realtime.
     if( mod(TimeVar,50.0) < 25.0 ) p += vec2((0.0005 + 0.0005*fbm(vec2(0.2*iTime)))*sin(50.0*p.y - 25.0*iTime),0.0);

     // Classic French cuisine : how to make croissants.
     float FD1 = sdCircle(p,0.50,vec2(-0.70,1.0),0.01);
     float DS1 = sdCircle(p,0.57,vec2(-0.75,1.0),0.07);
     float Croissant1 = FD1 - DS1;
     col += clamp(Croissant1,0.0,1.0);

     // I'm a friendly guy : I offer you another croissant !
     float FD2 = sdCircle(p,0.20,vec2(-0.75,0.75),0.01);
     float DS2 = sdCircle(p,0.27,vec2(-0.79,0.74),0.07);
     float Croissant2 = FD2 - DS2;
     col += 0.3*FD2*texture(iChannel2,2.0*r2d(3.5)*p + vec2(0.003*TimeVar,0.0));
     col += clamp(2.0*Croissant2,0.0,1.0);

     // Okay you get a third one.
     float FD3 = sdCircle(p,0.10,vec2( 0.80, 0.77),0.01);
     float DS3 = sdCircle(p,0.16,vec2( 0.83, 0.76),0.07);
     float Croissant3 = FD3 - DS3;
     col += 0.3*FD3*texture(iChannel1,1.5*r2d(3.5)*p + vec2(0.001*TimeVar,0.0));
     col += clamp(2.0*Croissant3,0.0,1.0);

     // Trinary Star System + Some Modulation
     float BV1 = 0.7 + 0.3*fbm(vec2(0.3*TimeVar         ));
     float BV2 = 0.7 + 0.3*fbm(vec2(0.3*TimeVar + 250.0 ));
     float BV3 = 0.5 + 0.5*fbm(vec2(0.3*TimeVar + 350.0 ));

     // Star Cross (with gimbal-assist)
     p += vec2(-0.5,-0.9);
     p *= r2d(-TiltX);
     col += metaDiamond( p, vec2( 0.0, 0.0), BV1*0.020, 0.0);
     col += 0.5*smoothstep(0.08,0.0,abs(p.y))*smoothstep(0.0015,0.0,abs(p.x));
     col += 0.5*smoothstep(0.08,0.0,abs(p.x))*smoothstep(0.0015,0.0,abs(p.y));
     p *= r2d(3.14159/4.0);
     col += 0.5*smoothstep(0.05,0.0,abs(p.y))*smoothstep(0.0015,0.0,abs(p.x));
     col += 0.5*smoothstep(0.05,0.0,abs(p.x))*smoothstep(0.0015,0.0,abs(p.y));
     p *= r2d(-3.14159/4.0);
     p *= r2d( TiltX);
     p -= vec2(-0.5,-0.9);

     // Medium Star
     p += vec2(-0.30,-1.05);
     p *= r2d(-TiltX);
     col += metaDiamond( p, vec2( 0.0, 0.0), BV2*0.005, 0.0);
     p *= r2d( TiltX);
     p -= vec2(-0.30,-1.05);

     // Small Star
     p += vec2(-0.25,-1.08);
     p *= r2d(-TiltX);
     col += metaDiamond( p, vec2( 0.0, 0.0), BV3*0.002, 0.0);
     p *= r2d( TiltX);
     p -= vec2(-0.25,-1.08);

     if( p.y < 0.5 )
     {
         // Beneath 0.5 : The Salt Flats
         col  = vec4(0.7,0.7,0.6,1.0);
         col += 0.5*vec4(texture(iChannel2,vec2( 0.50*(p.x)/((0.50-p.y)) + 4.0*TimeVar, log(0.50-p.y))));
     }else{
         // Above 0.5 : The mountains of "New New Mexico" (aka "Calientis V")
         col = mix(col,
                   vec4(0.74,0.74,0.9,1.0)*(0.5+0.2*texture(iChannel0,2.0*(p + vec2(0.02*TimeVar,0.0)))),
                   smoothstep(0.005,0.0,p.y + 0.05*fbm(vec2(2.5*p.x + 0.05*TimeVar)) - 0.57));
     };

     // Moebius-like floating temple right in the middle of the desert.
     // Because existential horror can strike anytime, anywhere. :p
     // Dedicated to Arzak fans...
     // Alternative title : "Easy fake lame DIY 3D in your 2D scene : an introduction"

     vec4 Color1 = vec4(0.9,0.9,1.0,1.0) - 0.15*texture(iChannel0,vec2(0.01*p.x,p.y - 0.01*sin(0.4*TimeVar))).xxxx;
     vec4 Color2 = vec4(0.7,0.7,1.0,1.0) - 0.15*texture(iChannel0,vec2(0.01*p.x,p.y - 0.01*sin(0.4*TimeVar))).xxxx;
     vec4 Face1;
     vec4 Face2;

     // BOOM ! You didn't see anything... ... Oh shut up, Gandalf !
     if( mod(0.005*TimeVar,0.08) < 0.04)
     {
         Face1 = Color1;
         Face2 = Color2;
     }else{
         Face1 = Color2;
         Face2 = Color1;
     };

     // The Moebius Rock floats. And sings.
     // I can hear it. My dog can hear it.
     // Why can't you ?!
     float AltitudeMoebius = 0.550 + 0.01*sin(0.4*TimeVar);

     p += vec2(mod(0.03*TimeVar,4.0) - 2.0,0.0);

     // Top Pylon
     col = mix(Face1,col,smoothstep(0.0,0.002,
     sdTriangle(p,
     vec2(-0.0200, AltitudeMoebius + 0.04),
     vec2( 0.0200, AltitudeMoebius + 0.04),
     vec2( 0.0000, AltitudeMoebius + 0.48) )));
     col = mix(Face2,col,smoothstep(0.0,0.002,
     sdTriangle(p,
     vec2(-0.0200                          , AltitudeMoebius + 0.04),
     vec2( 0.0200 - mod(0.005*TimeVar,0.04), AltitudeMoebius + 0.04),
     vec2( 0.0000                          , AltitudeMoebius + 0.48) )));

     // Bottom Tetrahedron
     col = mix(0.9*Face1,col,smoothstep(0.0,0.002,
     sdTriangle(p,
     vec2(-0.0200, AltitudeMoebius + 0.03),
     vec2( 0.0200, AltitudeMoebius + 0.03),
     vec2( 0.0000, AltitudeMoebius - 0.02) )));
     col = mix(0.9*Face2,col,smoothstep(0.0,0.002,
     sdTriangle(p,
     vec2(-0.0200                          , AltitudeMoebius + 0.03),
     vec2( 0.0200 - mod(0.005*TimeVar,0.04), AltitudeMoebius + 0.03),
     vec2( 0.0000                          , AltitudeMoebius - 0.02) )));

     // Ghostly Beacons
     col += metaDiamond( p, vec2( 0.0,AltitudeMoebius + 0.50), 0.001, 0.0);
     col += vec4(1.0,0.0,0.0,1.0)*metaDiamond( p, vec2( 0.0,AltitudeMoebius + 0.52), 0.001, 0.0);

     p -= vec2(mod(0.03*TimeVar,4.0) - 2.0,0.0);

     if( p.y > 0.5 )
     {
         // Very strange method to make 2D "mesas". Not sure it actually makes sense.
         // The final shapes are a bit pointy, which is fine for an extraterrestrial
         // desert, I suppose. Less so for martian mesas... Ah, well. Next time, in
         // another shader (incidentally I just figured out how to do it properly).
         // Anyway, let's pretend these are "giant ventifacts".
         float Inc = 1.0*p.x + 0.05*TimeVar; // Unit speed of Ventifacts relative to p.x (20:1 ratio)
         float MesaMaxHeight = GetMesaMaxHeight(p);
         float MesaLine = clamp( fbm(vec2(2.0*Inc + 0.005*fbm(vec2(80.0*p.y)))), 0.0, MesaMaxHeight);

         // Make the Sand follow a curve that is (more or less) realistic
         // Adding octaves, usual fbm impro stuff, you know the drill...
         float SandLine = 0.480 + 0.100*fbm(vec2( 2.0*Inc))
                                + 0.008*fbm(vec2(20.0*Inc))
                                + 0.002*fbm(vec2(60.0*Inc));

         // Basic Color + Vertically-stretched Texture + Horizontally-stretched Texture
         vec4 MesaColors = vec4(1.0,0.8,0.7,1.0);
         MesaColors += 0.5*texture(iChannel1,vec2(     Inc, 0.2*p.y));
         MesaColors += 0.5*texture(iChannel1,vec2( 0.1*Inc,     p.y));

         // Basic random shadows + slanted highlights...
         MesaColors -= 0.35*smoothstep( 0.0, 1.0, fbm(vec2(40.0*Inc)) + fbm(vec2(15.0*p.y - 30.0*Inc)));

         // More Shadows !
         MesaColors = MesaColors*( 0.2 + 0.8*smoothstep( 0.0, 0.4, (MesaLine - SandLine)));

         // Additional shadows at mesa's base.
         float VerticalWeathering = 1.0;
         VerticalWeathering *= (0.8+0.2*smoothstep(0.0,0.02,(p.y - 0.6 + 0.25*fbm(vec2(80.0*Inc)))));
         MesaColors *= VerticalWeathering;

         // Outputing mesas like big giant rotten teeth on a dead dragon's jaw...
         col = mix( col, MesaColors, smoothstep(0.007,0.0,p.y - MesaLine));

         // Adding highlights, because "secondary reflections", "ambient occlusion", etc
         // (haha, yeah right)
         col *= clamp(smoothstep(-0.15,0.0,p.y - MesaLine + 0.01*fbm(vec2(10.0*Inc))),0.5,1.0);

         // Mesas shadows on the sand...
         float SandShadows = 0.0;
         // If we're in the shadow of a mesa, SandLine altitude should decrease (...feeling of volume)
         if( SandLine < MesaLine ) SandLine = SandLine - 0.2*(MesaLine - SandLine);
         // Defining SandColors. Adding some y-stretched texture to simulate local sandslides.
         vec4 SandColors = 0.80*vec4(0.3,0.2,0.2,1.0)
                         + 0.20*texture(iChannel0,vec2(2.0*Inc,0.1*p.y + 0.0));

         // If we are in the shadow of a mesa
         if( SandLine < MesaLine)
         {
             // on-the-fly logic, probably false, but
             // just right enough to be useful.
             // "Paint-coding", guys...
             if( p.y > SandLine - (MesaLine - SandLine) )
             {
                 SandShadows = 0.7;
             }else{
                 SandShadows = 1.0;
             };
         }else{
             SandShadows = 1.0;
         };

         // Outputing shaded sand dune, "MY DUNE !" haha
         col = mix(col,SandShadows*SandColors,smoothstep(0.0025,0.0,p.y - SandLine));
    };

     vec2  ConsortiumShipPos = vec2( 2.0-mod(0.01*TimeVar + 1.0,4.0), -1.2);
     float ConsortiumShipPrec = 0.0035;
     vec4  HullColorFix = vec4(0.5,0.8,1.0,1.0);
     vec4  HullColorTop;
     vec4  HullColorBottom;


     // Move ship to position !
     p += ConsortiumShipPos;
     // Zoom Zoom Zoom !
     p *= 0.75;

     // Tweaking Ship Colors to make them just right (i.e. blend into the sky).
     HullColorTop    = HullColorFix*vec4(0.6,0.6,1.0,1.0) + 0.2*texture(iChannel0,vec2(2.0*p.x,0.1*p.y)).xxxx;
     HullColorTop *= 1.2;
     // Tweaking Ship Colors.
     HullColorBottom = HullColorFix*vec4(0.8,0.8,1.0,1.0) + 0.4*texture(iChannel0,vec2(0.5*p.x,0.1*p.y)).xxxx;
     HullColorBottom *= 0.6;

     // Fusion-Drive Tail visible due to reaction mass impurities (grey water from comets).
     if(p.x < 0.0) col += smoothstep(0.12,0.0,abs(0.2*p.x))*smoothstep(0.01,0.0,abs(p.y));

     // How to draw a spaceship in six triangles : a tutorial.

     // Forward part
     col = mix(HullColorTop,col,smoothstep(0.0,ConsortiumShipPrec,
     sdTriangle(p,
     vec2( 0.145, 0.00),
     vec2( 0.200, 0.01),
     vec2( 0.355, 0.00) )));
     col = mix(HullColorBottom,col,smoothstep(0.0,ConsortiumShipPrec,
     sdTriangle(p,
     vec2( 0.145, 0.00),
     vec2( 0.200,-0.015),
     vec2( 0.355, 0.00) )));

     // Middle Part
     col = mix(HullColorTop,col,smoothstep(0.0,ConsortiumShipPrec,
     sdTriangle(p,
     vec2( 0.000, 0.00),
     vec2( 0.005, 0.01),
     vec2( 0.150, 0.00) )));
     col = mix(HullColorBottom,col,smoothstep(0.0,ConsortiumShipPrec,
     sdTriangle(p,
     vec2( 0.000, 0.00),
     vec2( 0.005,-0.01),
     vec2( 0.150, 0.00) )));

     // Back Part
     col = mix(HullColorTop,col,smoothstep(0.0,ConsortiumShipPrec,
     sdTriangle(p,
     vec2(-0.005, 0.00),
     vec2( 0.010, 0.02),
     vec2( 0.070, 0.00) )));
     col = mix(HullColorBottom,col,smoothstep(0.0,ConsortiumShipPrec,
     sdTriangle(p,
     vec2(-0.005, 0.00),
     vec2( 0.010,-0.02),
     vec2( 0.070, 0.00) )));

     // End tutorial. You're welcome. :D

     // Fusion-Drive Glow (...keep this end at a distance)
     p += vec2( 0.005,-0.002);
     p *= r2d(-TiltX);
     col += metaDiamond(p,vec2(0.0,0.0), 0.010, 0.0);
     p *= r2d( TiltX);
     p -= vec2( 0.005,-0.002);
     // De-Zoom
     p *= 1.0/0.75;
     // Back to normal p.
     p -= ConsortiumShipPos;

     // Le Hovercraft

     vec2  HovercraftPos   = vec2(0.05  - 0.3*fbm(vec2(0.1*TimeVar)),-0.35);
     float HovercraftTrail = 0.335;
     float HovercraftBoost = 0.0;

     // Shadow
     col = mix(vec4(0.5),col,smoothstep(0.0,0.001,
     sdTriangle(p + HovercraftPos + vec2( 0.0, 0.01 ),
     vec2(-0.01+ 0.001*sin(2.0*TimeVar), 0.0050),
     vec2(-0.01+ 0.001*sin(2.0*TimeVar),-0.0050),
     vec2( 0.04- 0.001*sin(2.0*TimeVar), 0.000) )));

     // Lifting Body
     col = mix(vec4(0.5),col,smoothstep(0.0,0.001,
     sdTriangle(p + HovercraftPos + vec2( 0.0,- 0.001*sin(2.0*TimeVar) ),
     vec2(-0.01, 0.0050),
     vec2(-0.01,-0.0050),
     vec2( 0.04, 0.000) )));

     // Vertical Tail
     col = mix(vec4(0.4),col,smoothstep(0.0,0.001,
     sdTriangle(p + HovercraftPos + vec2( 0.0,- 0.001*sin(2.0*TimeVar) ),
     vec2(-0.010, 0.0050),
     vec2(-0.015, 0.015),
     vec2( 0.000, 0.0050) )));

     // Cockpit Canopy
     col = mix(vec4(0.2),col,smoothstep(0.0,0.001,
     sdTriangle(p + HovercraftPos + vec2( 0.0,- 0.001*sin(2.0*TimeVar) ),
     vec2( 0.000, 0.0050),
     vec2( 0.005, 0.0000),
     vec2( 0.025, 0.0010) )));

     // Dust Trail
     if( p.x < -0.05  + 0.3*fbm(vec2(0.1*TimeVar)) - 0.01 )
         col += 0.1*smoothstep(0.0,0.01,p.y - HovercraftTrail)
                   *smoothstep(0.035,0.0, p.y -0.015*abs(5.0*(p.x + HovercraftPos.x))
                                                    *fbm(vec2(10.0*(p.x + HovercraftPos.x) + 10.0*TimeVar)) - 0.98*HovercraftTrail);
     // Very lame yet mostly accurate thruster simulation.
     // This shader is a disgrace to mathematics, exhibit 41 :
     if( fbm(vec2(0.1*(TimeVar + 0.1))) - fbm(vec2(0.1*(TimeVar))) > 0.005)
     {
        // Haha rocket goes BRRRRRR !
        HovercraftBoost = 0.005;
     }else{
        // Puff Puff Puff Puff Puff
        HovercraftBoost = abs(0.003*sin(20.0*TimeVar));
     };

     // Rocket Blast
     col += vec4(1.0,0.5,0.5,1.0)*metaDiamond(p + HovercraftPos + vec2(  0.015,- 0.0015*sin(2.0*TimeVar)),vec2( 0.0,0.0), HovercraftBoost, 10.0);

     // A bit of dust in the air...
     if( p.y > 0.5) col += 0.25*smoothstep(0.25,0.0,p.y - 0.1*fbm(vec2(2.0*p.x + 1.0*TimeVar)) - 0.5);

     // Make a haze just above the ground in the distance.
     col += 0.2*smoothstep(0.01,0.0,abs(p.y-0.5));

     // For the last 25 TimeVars of a 50 TimeVars cycle...
     if( mod(TimeVar,50.0) > 25.0 )
     {
         // Draw some Nasa camera crosses to look cool and realistic (hahahahaha)
         if(mod(fragCoord.y + 200.0,400.0) > 399.0) if(mod(fragCoord.x + 50.0 + 200.0,400.0) < 100.0) col = vec4(0.2,0.2,0.2,1.0);
         if(mod(fragCoord.x + 200.0,400.0) > 399.0) if(mod(fragCoord.y + 50.0 + 200.0,400.0) < 100.0) col = vec4(0.2,0.2,0.2,1.0);
     };

     // Lensflares ! Lensflares everywhere !
     
     // Big Star
     // Let's compute Mesa's height at the Big Star's x-coordinate.
     float NewMesaLine = clamp( fbm(vec2(2.0*(0.5 + 0.05*TimeVar)) + 0.005*fbm(vec2(80.0*0.9))),0.0,GetMesaMaxHeight(vec2(0.5,0.0)));

     p += vec2(-0.5,-0.9);
     p *= r2d(-TiltX);
     col += 0.7*vec4(0.5,0.8,1.0,1.0)
               // If the mesa's top is above the Big Star, remove lensflare.
               *smoothstep(0.0,0.01,0.9-NewMesaLine)
               // If the Moebius Rock clips the Big Star, remove lensflare.
               *smoothstep(0.0,0.01,abs(mod(0.03*TimeVar,4.0) - 2.0 + 0.5))
               // Basic Hand-Made Linear 2D Lensflare
               // the best kind, like granma used to...
               *smoothstep(0.03,0.0,abs(p.y))
               *smoothstep(2.00,0.0,abs(p.x));
     // Circle around the Big Star. Not exactly JWST-worthy, I know.
     // Look, I'm just doing my best, okay ?! :D
     col += 0.1*smoothstep(0.0125,0.0,abs(sdCircle(p,0.05,vec2(0.0),0.07) - 0.0125))
               *smoothstep(0.0,0.01,0.9-NewMesaLine)
               *smoothstep(0.0,0.01,abs(mod(0.03*TimeVar,4.0) - 2.0 + 0.5));
     p *= r2d( TiltX);
     p -= vec2(-0.5,-0.9);

     // Medium Star
     p += vec2(-0.30,-1.05);
     p *= r2d(-TiltX);
     col += 0.35*vec4(0.5,0.8,1.0,1.0)
               *smoothstep(0.01,0.0,abs(p.y))
               *smoothstep(0.50,0.0,abs(p.x));
     p *= r2d( TiltX);
     p -= vec2(-0.30,-1.05);

     // Small Star
     p += vec2(-0.25,-1.08);
     p *= r2d(-TiltX);
     col += 0.35*vec4(0.5,0.8,1.0,1.0)
               *smoothstep(0.01,0.0,abs(p.y))
               *smoothstep(0.25,0.0,abs(p.x));
     p *= r2d( TiltX);
     p -= vec2(-0.25,-1.08);

     // Every 25 TimeVars, pretend like you're watching through an Active SunShade
     // that cancels heat distorsion through some kind of adaptative optics magic.
     // Hey, it's the future after all.
     if(mod(TimeVar,50.0) > 25.0) col *= vec4(1.0,0.8,0.7,1.0);

     // HO MY GOD !
     fragColor = col;
}