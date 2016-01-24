#include "../include/Shader.hpp"

Shader::Shader(Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "Shader.cpp\t\tInitializing\n";
	}
	
	// Create and compile our GLSL program from the shaders
	programId = glCreateProgram();
}

Shader::~Shader(void){
	if((debugLevel & 0x10) == 16){			
		std::cout << "Shader.cpp\t\tFinalizing\n";
	}
	glDeleteProgram(programId);
}

void Shader::bind(void){
	glUseProgram(programId);
}

void Shader::addShader(const char *path, GLenum shaderType){
	// Debug
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tCreating the shader\n";
	}
	
	// Create the shaders
	GLuint shaderID = glCreateShader(shaderType);
	
	// Read the Shader code from the file
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tRead the Shader code from the file\n";
	}
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(path, std::ios::in);
	if(VertexShaderStream.is_open()){
		std::string Line = "";
		while(getline(VertexShaderStream, Line))
			VertexShaderCode += "\n" + Line;
		VertexShaderStream.close();
	}
	
	// Compile Shader
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tCompiling shader : " << path << std::endl;
	}
	
	char const * sourcePointer = VertexShaderCode.c_str();
	glShaderSource(shaderID, 1, &sourcePointer , NULL);
	glCompileShader(shaderID);
	
	// Check Shader
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> shaderErrorMessage(InfoLogLength);
	glGetShaderInfoLog(shaderID, InfoLogLength, NULL, &shaderErrorMessage[0]);
	
	if(InfoLogLength > 1){	
		fprintf(stdout, "%s\n", &shaderErrorMessage[0]);
	}
	
	glAttachShader(programId, shaderID);
	glDeleteShader(shaderID);
}

void Shader::link(void){
	// Link the program
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tLinking program\n";
	}
	
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glLinkProgram(programId);
	
	// Check the program
	glGetProgramiv(programId, GL_LINK_STATUS, &Result);
	glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> ProgramErrorMessage( max(InfoLogLength, int(1)) );
	glGetProgramInfoLog(programId, InfoLogLength, NULL, &ProgramErrorMessage[0]);
	
	if(InfoLogLength > 1){
		fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
	}
}

GLuint Shader::getID(void){
	return programId;
}
