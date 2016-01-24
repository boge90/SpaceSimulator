#version 440 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 in_vertex;
layout(location = 1) in float in_temperature;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

// Solar coverage
out float temperature;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(in_vertex, 1.0);
	
	// Pass through variables
	temperature = in_temperature;
}
