#version 440 core

// Interpolated values from the vertex shaders
in float temperature;

// Color
out vec4 color;

void main(){
	// Output color = color of the texture at the specified UV
	color = vec4(temperature, temperature, temperature, 1.0);
}
