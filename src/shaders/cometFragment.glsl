#version 440 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;
in float solarCoverage;

// Color
out vec4 color;

void main(){
	// Output color = color of the texture at the specified UV
	color = vec4(fragmentColor * solarCoverage, 1.0);
}
