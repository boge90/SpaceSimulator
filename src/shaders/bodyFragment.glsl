#version 440 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;
in float solarCoverage;
in vec2 texCoord;

// Texture
uniform sampler2D tex;

// Color
out vec4 color;

void main(){
	// Output color = color of the texture at the specified UV
	color = vec4(fragmentColor, 1.0) * texture2D(tex, texCoord) * solarCoverage;
}
