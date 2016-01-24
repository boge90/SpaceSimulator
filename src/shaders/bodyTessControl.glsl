#version 440 core

layout(vertices = 3) out;


void main(){
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	
	gl_TessLevelInner[0] = 2;
	gl_TessLevelOuter[0] = 2;
	gl_TessLevelOuter[1] = 2;
	gl_TessLevelOuter[2] = 2;
}
