#version 440 core

layout(triangles, equal_spacing, ccw) in;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

// Output variables
out vec2 texCoord;

void main(){
	vec3 v1 = gl_TessCoord.x * gl_in[0].gl_Position.xyz;
	vec3 v2 = gl_TessCoord.y * gl_in[1].gl_Position.xyz;
	vec3 v3 = gl_TessCoord.z * gl_in[2].gl_Position.xyz;
	
	vec4 vert = vec4(v1+v2+v3, 1.0);
	
	gl_Position = MVP * vert;
	
	// Texture coords
	texCoord = vec2((atan(vert.x, vert.z) / 3.1415926 + 1.0) * 0.5, (asin(vert.y) / 3.1415926 + 0.5));
}
