#version 440 core

// Interpolated values from the vertex shaders
in vec2 texCoord;

// Texture
uniform sampler2D tex;

// Color
out vec4 color;

void main(){
	// Output color = color of the texture at the specified UV
	color = texture2D(tex, texCoord);
}
