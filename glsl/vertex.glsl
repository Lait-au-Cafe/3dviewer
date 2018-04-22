#version 450

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

out vec4 v_color;

void main(){
	float vx, vy, vz;

	float rot_y = radians(-50.0);
	vx = cos(rot_y) * position.x + sin(rot_y) * position.z;
	vy = position.y;
	vz = -sin(rot_y) * position.x + cos(rot_y) * position.z;

	vx -= 0.0;
	vy -= 1.0;

	vx /= vz + 1.0;
	vy /= vz + 1.0;

	gl_Position = vec4(vx, vy, 1.0, 1.0);

	if(gl_VertexID >= 200 * 400){
		v_color = vec4(1.0, 0.0, 0.0, 1.0);
	} else {
		v_color = vec4(0.1, 1.0, 0.0, 1.0);
	}
}
