float rnd(vec2 inval){
    return fract(sin(dot(inval.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float on_at(vec2 grid_num, int col, int row) {
	float rv = 0.0;
	rv = float(grid_num.x == float(col) && grid_num.y == float(row));		
	return rv;	
}

float dstate(int digit, vec2 gn) {
	float is0, is1, is2, is3, is4, is5, is6, is7, is8, is9;
	float shape0, shape1, shape2, shape3, shape4, shape5;
	float shape6, shape7, shape8, shape9;
	float rval = 0.0;
	
	is0 = float(digit == 0);
	is1 = float(digit == 1);
	is2 = float(digit == 2);
	is3 = float(digit == 3);
	is4 = float(digit == 4);	
	is5 = float(digit == 5);
	is6 = float(digit == 6);
	is7 = float(digit == 7);
	is8 = float(digit == 8);
	is9 = float(digit == 9);		
		
	shape0 = on_at(gn, 2,1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +					
			on_at(gn, 1, 2) +		
			on_at(gn, 1, 3) +		
			on_at(gn, 1, 4) +		
			on_at(gn, 1, 5) +		
			on_at(gn, 1, 6) +		
			on_at(gn, 1, 7) +				
			on_at(gn, 5, 2) +		
			on_at(gn, 5, 3) +		
			on_at(gn, 5, 4) +		
			on_at(gn, 5, 5) +		
			on_at(gn, 5, 6) +		
			on_at(gn, 5, 7) +		
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8);						
	shape0 = float(shape0 > 0.0);	
	
	shape1 = on_at(gn, 2,1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +					
			on_at(gn, 3, 2) +		
			on_at(gn, 3, 3) +		
			on_at(gn, 3, 4) +		
			on_at(gn, 3, 5) +		
			on_at(gn, 3, 6) +		
			on_at(gn, 3, 7) +				
			on_at(gn, 3, 8) +				
			on_at(gn, 2, 7);						
	shape1 = float(shape1 > 0.0);	
	
	shape2 = on_at(gn, 1, 1) +
			on_at(gn, 2, 1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +
			on_at(gn, 5, 1) +			
			on_at(gn, 2, 1) +			
			on_at(gn, 1, 2) +
			on_at(gn, 2, 3) +
			on_at(gn, 3, 4) +
			on_at(gn, 4, 5) +
			on_at(gn, 5, 6) +
			on_at(gn, 5, 7) +			
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8) +			
			on_at(gn, 1, 7);			
	shape2 = float(shape2 > 0.0);
	
	shape3 = on_at(gn, 1, 2) +
			on_at(gn, 2, 1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +			
			on_at(gn, 5, 2) +
			on_at(gn, 5, 3) +
			on_at(gn, 5, 4) +			
			on_at(gn, 5, 6) +
			on_at(gn, 5, 7) +			
			on_at(gn, 3, 5) +
			on_at(gn, 4, 5) +			
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8) +
			on_at(gn, 1, 7);			
	shape3 = float(shape3 > 0.0);
	
	shape4 = on_at(gn, 4, 1) +
			on_at(gn, 4, 2) +
			on_at(gn, 4, 3) +
			on_at(gn, 4, 4) +
			on_at(gn, 4, 5) +
			on_at(gn, 4, 6) +
			on_at(gn, 4, 7) +
			on_at(gn, 4, 8) +			
			on_at(gn, 1, 4) +
			on_at(gn, 2, 4) +
			on_at(gn, 3, 4) +
			on_at(gn, 4, 4) +
			on_at(gn, 5, 4) +			
			on_at(gn, 1, 5) +
			on_at(gn, 2, 6) +
			on_at(gn, 3, 7);
	shape4 = float(shape4 > 0.0);
	
	shape5 = on_at(gn, 2, 1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +
			on_at(gn, 1, 2) +			
			on_at(gn, 5, 2) +
			on_at(gn, 5, 3) +
			on_at(gn, 5, 4) +			
			on_at(gn, 1, 5) +
			on_at(gn, 2, 5) +
			on_at(gn, 3, 5) +
			on_at(gn, 4, 5) +			
			on_at(gn, 1, 6) +
			on_at(gn, 1, 7) +
			on_at(gn, 1, 8) +			
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8) +
			on_at(gn, 5, 8);
	shape5 = float(shape5 > 0.0);
	
	shape6 = on_at(gn, 2, 1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +
			on_at(gn, 5, 2) +
			on_at(gn, 5, 3) +
			on_at(gn, 5, 4) +			
			on_at(gn, 1, 2) +
			on_at(gn, 1, 3) +
			on_at(gn, 1, 4) +
			on_at(gn, 1, 5) +
			on_at(gn, 1, 6) +			
			on_at(gn, 2, 5) +
			on_at(gn, 3, 5) +
			on_at(gn, 4, 5) +
			on_at(gn, 2, 7) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8);
	shape6 = float(shape6 > 0.0);
	
	shape7 = on_at(gn, 2, 1) +
			on_at(gn, 2, 2) +
			on_at(gn, 3, 3) +
			on_at(gn, 3, 4) +
			on_at(gn, 4, 5) +
			on_at(gn, 4, 6) +
			on_at(gn, 5, 7) +
			on_at(gn, 5, 8) +			
			on_at(gn, 1, 8) +
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8);
	shape7 = float(shape7 > 0.0);
	
	shape8 = on_at(gn, 2, 1) +
			on_at(gn, 3, 1) +
			on_at(gn, 4, 1) +			
			on_at(gn, 1, 2) +
			on_at(gn, 1, 3) +
			on_at(gn, 1, 4) +			
			on_at(gn, 5, 2) +
			on_at(gn, 5, 3) +
			on_at(gn, 5, 4) +			
			on_at(gn, 2, 5) +
			on_at(gn, 3, 5) +
			on_at(gn, 4, 5) +			
			on_at(gn, 1, 6) +
			on_at(gn, 1, 7) +			
			on_at(gn, 5, 6) +
			on_at(gn, 5, 7) +			
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8);
	shape8 = float(shape8 > 0.0);
	
	shape9 = on_at(gn, 2, 1) +
			on_at(gn, 3, 1) +			
			on_at(gn, 4, 2) +
			on_at(gn, 5, 3) +			
			on_at(gn, 2, 4) +
			on_at(gn, 3, 4) +
			on_at(gn, 4, 4) +
			on_at(gn, 5, 4) +			
			on_at(gn, 5, 5) +
			on_at(gn, 5, 6) +
			on_at(gn, 5, 7) +			
			on_at(gn, 1, 5) +
			on_at(gn, 1, 6) +
			on_at(gn, 1, 7) +			
			on_at(gn, 2, 8) +
			on_at(gn, 3, 8) +
			on_at(gn, 4, 8);						
	shape9 = float(shape9 > 0.0);
	
	rval = shape0 * is0 +
			shape1 * is1 +
			shape2 * is2 +
			shape3 * is3 + 
			shape4 * is4 + 
			shape5 * is5 + 
			shape6 * is6 + 
			shape7 * is7 + 
			shape8 * is8 + 
			shape9 * is9;				
	return rval;	
}

float bin_digit(int clock, vec2 grid_num, bool bits[8]) {
	bool isActive = false;
	float rv = 0.0;
		
	isActive = 
			grid_num.x == 7.0 && bits[0] ||
			grid_num.x == 6.0 && bits[1] ||
			grid_num.x == 5.0 && bits[2] ||
			grid_num.x == 4.0 && bits[3] ||
			grid_num.x == 3.0 && bits[4] ||
			grid_num.x == 2.0 && bits[5] ||
			grid_num.x == 1.0 && bits[6] ||
			grid_num.x == 0.0 && bits[7];			
	return float(isActive);
}

float dist_to_line(vec2 pt1, vec2 pt2, vec2 testPt)
{
  vec2 lineDir = pt2 - pt1;
  vec2 perpDir = vec2(lineDir.y, -lineDir.x);
  vec2 dirToPt1 = pt1 - testPt;
  return abs(dot(normalize(perpDir), dirToPt1));
}

float frame(vec2 pix, vec2 pos, vec2 size, float scale) {
	vec2 rpix, uv;
	float is_inside, is_core;
	float rv = 0.0;
	vec2 dist_to_edge;
	float ew_px = 8.0 * scale;
	float shine_px = 2.0;
	float isShadow;
	float shiner;
	float d;
	float show;
	
	// Area check
	rpix = pix - pos;
	is_inside = float(rpix.x >= 0.0 && rpix.x < size.x && 
				  rpix.y >= 0.0 && rpix.y < size.y);
	is_core = float(rpix.x > ew_px && rpix.x < size.x - ew_px &&
					rpix.y > ew_px && rpix.y < size.y - ew_px);
		
	// Basic frame
	dist_to_edge = min(rpix, size-rpix);
	float v = min(dist_to_edge.x, dist_to_edge.y);
	//rv = 1.0 - smoothstep(0.0, ew_px, v);	
	rv = 0.7 * (1.0 - step(ew_px, v));
	
	// Shine A
	d = dist_to_line(vec2(0.0, size.y), vec2(ew_px, size.y - ew_px), rpix);
	shiner = 1.0 - smoothstep(0.0, shine_px, d);
	show = float(rpix.x < ew_px);
	rv += shiner * 3.0 * show;
	
	// Shine B
	d = dist_to_line(vec2(ew_px, size.y - ew_px), vec2(ew_px, ew_px), rpix);
	shiner = 1.0 - smoothstep(0.0, shine_px, d);
	show = float(rpix.y > ew_px && rpix.y < size.y - ew_px);
	rv += shiner * 3.0 * show;
	
	// Shine C
	d = dist_to_line(vec2(0.0, size.y - ew_px), vec2(size.x, size.y - ew_px), rpix);
	shiner = 1.0 - smoothstep(0.0, shine_px, d);
	show = float(rpix.x > ew_px && rpix.x < size.x - ew_px);
	rv += shiner * 3.0 * show;	
	rv = clamp(rv, 0.0, 1.0);
	
	// Shadow
	bool a = (rpix.x >= rpix.y) && (rpix.y <= ew_px);
	bool b = (size.x - rpix.x  <= size.y - rpix.y) && (rpix.x >= size.x - ew_px);
	isShadow = float(a || b);
	rv = rv - isShadow * 0.25;
	
	rv = clamp(rv, 0.0, 1.0);
	return rv * is_inside * (1.0 - is_core);
}

float bar_counter(int clock, vec2 pix, vec2 pos, vec2 size, bool bits[8]) {
	vec2 rpix;
	vec2 grids, grid_num;
	vec2 spacing;
	vec2 cell_min, cell_max, cell_uv;
	vec4 rc = vec4(0.0);
	float d;
	float on_level, on_off, off_level, brightness, light_level;
	float is_inside;
	float algo;
	
	rpix = pix - pos;
	is_inside = float(rpix.x >= 0.0 && rpix.x < size.x && 
	                  rpix.y >= 0.0 && rpix.y < size.y);
	
	grids = vec2(8.0, 1.0);
	spacing = size / grids;	
	grid_num = floor(rpix / spacing);
	cell_min = grid_num * spacing;
	cell_max = cell_min + spacing;
	cell_uv = (rpix - cell_min) / (cell_max - cell_min);
	cell_uv = (2.0 * cell_uv) - 1.0;	
	
	// Base level
	vec2 ev = abs(cell_uv);
	float d_to_edge = 1.0 - max(ev.x, ev.y);
	float base_cut = 0.10;
	float base_level = 0.12 * smoothstep(0.0, base_cut, d_to_edge);	
			
	// Light level
	d = length(cell_uv);
	on_level = 1.0;
	off_level = 0.125;
	on_off = bin_digit(clock, grid_num, bits);
	brightness = (on_level * on_off) + ( off_level * (1.0 - on_off)); // One or the other only
	light_level = (brightness*2.5) * (1.0 - sqrt(d));	
	light_level = clamp(light_level, 0.0, 1.0);
	
	algo = (0.001 + base_level + light_level) * is_inside;	
	return algo;	
}

float disp(int digit, vec2 pix, vec2 pos, vec2 size) {
	vec2 rpix;
	vec2 grids, grid_num;
	vec2 spacing;
	vec2 cell_min, cell_max, cell_uv;
	vec4 rc = vec4(0.0);
	float d;
	float on_level, on_off, off_level, brightness, light_level;
	float is_inside;
	float algo;
	
	rpix = pix - pos;
	is_inside = float(rpix.x >= 0.0 && rpix.x < size.x && 
	                  rpix.y >= 0.0 && rpix.y < size.y);
	
	grids = vec2(7.0, 10.0);
	spacing = size / grids;	
	grid_num = floor(rpix / spacing);
	cell_min = grid_num * spacing;
	cell_max = cell_min + spacing;
	cell_uv = (rpix - cell_min) / (cell_max - cell_min);
	cell_uv = (2.0 * cell_uv) - 1.0;	
			
	// Light level
	d = length(cell_uv);
	on_level = 1.0;
	off_level = 0.125;
	on_off = dstate(digit, grid_num);
	brightness = (on_level * on_off) + ( off_level * (1.0 - on_off)); // One or the other only
	light_level = (brightness*2.5) * (1.0-sqrt(d));	
	light_level = clamp(light_level, 0.0, 1.0);
	
	algo = (0.001 + light_level) * is_inside;
	
	return algo;	
}

float background(vec2 pix, vec2 size) {
	vec2 rpix;
	vec2 grids, grid_num;
	vec2 spacing;
	vec2 cell_min, cell_max, cell_uv;	
	float algo;
	
	grids = vec2(300.0, 200.0);
	spacing = size / grids;	
	grid_num = floor(pix / spacing);
	cell_min = grid_num * spacing;
	cell_max = cell_min + spacing;
	cell_uv = (pix - cell_min) / (cell_max - cell_min);
	cell_uv = (2.0 * cell_uv) - 1.0;	
			
	algo = 0.50 + (0.08 * rnd(grid_num * 0.01234));
		
	return algo;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 fc = vec4(0.0);
    vec2 reso = iResolution.xy;
    vec2 pix = fragCoord;    		
	vec2 uv = pix / reso;	
	vec2 pos, size;
	float hwa = 10.0 / 7.0;
	vec2 info_size;	
	float info[24];	
	vec2 bar_pos, bar_size;
	//
	vec4 info_c = vec4(0.0);
	vec4 bar_c = vec4(0.0);
	vec4 bar_fr_c = vec4(0.0);
	vec4 decimal_fr_c = vec4(0.0);
	vec4 bin_c = vec4(0.0);
	vec4 dpa_c = vec4(0.0); 
	vec4 dpb_c = vec4(0.0);
	vec4 dpc_c = vec4(0.0);
	//
	vec2 fr_ext, fr_size, fr_pos;		
	vec2 decimal_pos, decimal_size;	
	float frame_slower, clock;	
	int d1, d10, d100;
	float cval;
	float px, py;
	float info_py;
	float isBackground;
	float fval;
	float scale = 1.0;
	
	// Clock
	cval = mix(30.0, 1.0, 50.0 * 0.01);
	frame_slower = floor(float(iFrame) / cval);
	clock = mod(frame_slower, 256.0);
	// Scale
	scale = 0.70;

	bool bits[8];
	// Bits
	bits[0] = !(mod(floor(clock / 1.0), 2.0) == 0.0);
	bits[1] = !(mod(floor(clock / 2.0), 2.0) == 0.0);
	bits[2] = !(mod(floor(clock / 4.0), 2.0) == 0.0);
	bits[3] = !(mod(floor(clock / 8.0), 2.0) == 0.0);
	//
	bits[4] = !(mod(floor(clock / 16.0), 2.0) == 0.0);
	bits[5] = !(mod(floor(clock / 32.0), 2.0) == 0.0);
	bits[6] = !(mod(floor(clock / 64.0), 2.0) == 0.0);
	bits[7] = !(mod(floor(clock / 128.0), 2.0) == 0.0);	
	
	
	// LEDs
	bar_size.x = 0.92 * reso.x * scale;
	bar_size.y = bar_size.x / 8.0;
	bar_pos.x = (reso.x / 2.0) - (bar_size.x / 2.0);
	bar_pos.y = reso.y * 0.25;
	if (pix.y >= bar_pos.y && pix.y <= bar_pos.y + bar_size.y) {
		bar_c = vec4(bar_counter(int(clock), pix, bar_pos, bar_size, bits));
	}
		
	// LED Bar frame
	fr_ext = bar_size * vec2(0.02, 0.20);
	fr_size = bar_size + fr_ext;
	fr_pos = bar_pos - (fr_ext / 2.0);		
	if (pix.y >= fr_pos.y && pix.y <= fr_pos.y + fr_size.y) {
		bar_fr_c = vec4(frame(pix, fr_pos, fr_size, scale)) * vec4(1.0, 1.0, 1.0, 1.0);		
	}
	
	// Info digits
	info_size.x = 1.32 * ((bar_size.x / 8.0) / 3.0);
	info_size.y = info_size.x * hwa;	
	info_py = bar_pos.y + bar_size.y + (12.0 * scale);		
	if (pix.y >= info_py && pix.y <= info_py + info_size.y) {		
		fval = 0.0;
		fval += disp(1, pix, vec2(bar_pos.x + (-0.029 * bar_size.x), info_py), info_size);
		fval += disp(2, pix, vec2(bar_pos.x + ( 0.012 * bar_size.x), info_py), info_size);
		fval += disp(8, pix, vec2(bar_pos.x + ( 0.060 * bar_size.x), info_py), info_size);		
		fval += disp(6, pix, vec2(bar_pos.x + ( 0.130 * bar_size.x), info_py), info_size);		
		fval += disp(4, pix, vec2(bar_pos.x + ( 0.180 * bar_size.x), info_py), info_size);		
		fval += disp(3, pix, vec2(bar_pos.x + ( 0.255 * bar_size.x), info_py), info_size);		
		fval += disp(2, pix, vec2(bar_pos.x + ( 0.305 * bar_size.x), info_py), info_size);		
		fval += disp(1, pix, vec2(bar_pos.x + ( 0.383 * bar_size.x), info_py), info_size);		
		fval += disp(6, pix, vec2(bar_pos.x + ( 0.428 * bar_size.x), info_py), info_size);		
		fval += disp(8, pix, vec2(bar_pos.x + ( 0.532 * bar_size.x), info_py), info_size);		
		fval += disp(4, pix, vec2(bar_pos.x + ( 0.655 * bar_size.x), info_py), info_size);		
		fval += disp(2, pix, vec2(bar_pos.x + ( 0.785 * bar_size.x), info_py), info_size);		
		fval += disp(1, pix, vec2(bar_pos.x + ( 0.910 * bar_size.x), info_py), info_size);		
		info_c = vec4(fval) * vec4(1.0, 1.0, 0.0, 1.0);		
	}
	
	// Binary digits
	vec2 bin_size;
	bin_size.x = bar_size.x / 8.0;
	bin_size.y = bin_size.x * (10.0 / 7.0);	
	py = bar_pos.y - (reso.y * 0.25 * scale);	
	if (pix.y >= py && pix.y <= py + bin_size.y) {
		float fv = 0.0;
		for (int idx = 0; idx < 8; idx++) {
			px = bar_pos.x + (7.5 * scale) + (float(idx) * bin_size.x);		
			fv += disp(int(bits[7-idx]), pix, vec2(px, py), bin_size * 0.70);
		}
		bin_c = vec4(fv) * vec4(1.0);
	}
	
	// Decimal counter
	decimal_size.x = 0.12 * reso.x * scale;
	decimal_size.y = decimal_size.x * (10.0 / 7.0);	
	py = bar_pos.y + (reso.y * 0.55 * scale);
	if (pix.y >= py && pix.y <= py + decimal_size.y) {
		d1 = int(mod(floor(clock), 10.0));
		d10 = int(mod(floor(clock / 10.0), 10.0));
		d100 = int(mod(floor(clock / 100.0), 10.0));		
		px = (reso.x / 2.0) - (1.5 * decimal_size.x);
		dpa_c = vec4(disp(d100, pix, vec2(px, py), decimal_size)); // 100x	
		px = (reso.x / 2.0) - (0.5 * decimal_size.x);
		dpb_c = vec4(disp(d10, pix, vec2(px, py), decimal_size)); // 10x	
		px = (reso.x / 2.0) + (0.5 * decimal_size.x);
		dpc_c = vec4(disp(d1, pix, vec2(px, py), decimal_size)); // 1x
	}
	
	// Decimal frame
	vec2 all_size = decimal_size;
	all_size.x *= 3.0;
	fr_ext = all_size * vec2(0.06, 0.14);
	fr_size = all_size + fr_ext;	
	px = (reso.x / 2.0) - (1.5 * decimal_size.x);
	fr_pos = vec2(px, py) - (fr_ext / 2.0);		
	if (pix.y >= fr_pos.y && pix.y <= fr_pos.y + fr_size.y) {
		decimal_fr_c = vec4(frame(pix, fr_pos, fr_size, scale)) * vec4(1.0, 1.0, 1.0, 1.0);
	}
				
	// Final gather
	fc = info_c;
	fc += bar_c + bar_fr_c;
	fc += bin_c;
	fc += (dpa_c + dpb_c + dpc_c) * vec4(1.0, 1.0, 0.0, 1.0);
	fc += decimal_fr_c;	
	fc += vec4(background(pix, reso)) * vec4(0.0, 0.5, 0.0, 1.0) * float(fc.a == 0.0);

	fragColor = fc;
}

