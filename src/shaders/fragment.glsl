#version 440 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;
in vec4 texCoords;

// Texture
uniform sampler2D mytexture;

// Color
out vec4 color;

void main(){
	// Calculating texCords
	vec2 longitudeLatitude = vec2((atan(texCoords.y, texCoords.x) / 3.1415926 + 1.0) * 0.5, (asin(texCoords.z) / 3.1415926 + 0.5));

	// Output color = color of the texture at the specified UV
	color = vec4(fragmentColor, 1.0);// * texture2D(mytexture, longitudeLatitude);
}