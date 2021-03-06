#version 440 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 in_vertex;
layout(location = 1) in vec3 in_color;
layout(location = 2) in float in_solarCoverage;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

// Output values
out vec3 fragmentColor;

// Solar coverage
out float solarCoverage;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(in_vertex, 1.0);
	
	solarCoverage = in_solarCoverage;
	
	// Color
	fragmentColor = in_color;
}
