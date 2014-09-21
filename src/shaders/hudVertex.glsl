#version 440 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 in_vertex;
layout(location = 1) in vec2 in_texCoord;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

// Output values
out vec2 texCoord;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(in_vertex, 1.0);
	
	// Color
	texCoord = in_texCoord;
}
