#version 440 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;

// Color
out vec4 color;

void main(){
	// Output color = color of the texture at the specified UV
	color = vec4(fragmentColor, 1.0);
}
