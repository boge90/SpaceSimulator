#version 440 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 in_vertex;
layout(location = 1) in float in_coverage;

// Atmosphere color
uniform vec3 in_color;

// Values that stay constant for the whole execution
uniform mat4 MVP;

// Output values
out vec3 fragmentColor;
out float solarCoverage;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(in_vertex, 1.0);
	
	// Color
	fragmentColor = in_color;
	
	solarCoverage = in_coverage;
}
