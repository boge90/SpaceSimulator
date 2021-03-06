#version 440 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 color;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

// Output values
out vec3 fragmentColor;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(vertex, 1.0);
	   
	// Color
	fragmentColor = color;
}
