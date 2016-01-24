#version 440 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 in_vertex; 
layout(location = 1) in vec3 in_color;
layout(location = 2) in float in_solarCoverage;
layout(location = 3) in vec2 in_texCoord;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

// Output values
out vec3 fragmentColor;

// Solar coverage
out float solarCoverage;

// Texture
out vec2 texCoord;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(in_vertex, 1.0);
	
	// Pass through variables
	texCoord = in_texCoord;
	solarCoverage = in_solarCoverage;
	fragmentColor = in_color;
}
