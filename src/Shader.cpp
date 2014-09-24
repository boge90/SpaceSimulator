#include "../include/Shader.hpp"

Shader::Shader(const char *vertexShader, const char *fragmentShader, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "Shader.cpp\t\tInitializing\n";
	}
	
	// Create and compile our GLSL program from the shaders
	shaderId = loadShaders(vertexShader, fragmentShader);
}

Shader::~Shader(void){
	if((debugLevel & 0x10) == 16){			
		std::cout << "Shader.cpp\t\tFinalizing\n";
	}
	glDeleteProgram(shaderId);
}

void Shader::bind(void){
	glUseProgram(shaderId);
}

GLuint Shader::loadShaders(const char *vertex_file_path, const char *fragment_file_path){
	// Debug
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tCreating the shader\n";
	}
	
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	
	// Read the Vertex Shader code from the file
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tRead the Vertex Shader code from the file\n";
	}
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if(VertexShaderStream.is_open()){
		std::string Line = "";
		while(getline(VertexShaderStream, Line))
			VertexShaderCode += "\n" + Line;
		VertexShaderStream.close();
	}
	
	// Read the Fragment Shader code from the file
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tRead the Fragment Shader code from the file\n";
	}
	
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if(FragmentShaderStream.is_open()){
		std::string Line = "";
		while(getline(FragmentShaderStream, Line))
			FragmentShaderCode += "\n" + Line;
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;
	
	// Compile Vertex Shader
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tCompiling shader : " << vertex_file_path << std::endl;
	}
	
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);
	
	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> VertexShaderErrorMessage(InfoLogLength);
	glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
	
	if(InfoLogLength > 1){	
		fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
	}
	
	// Compile Fragment Shader
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tCompiling shader : " << fragment_file_path << std::endl; 
	}
	
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);
	
	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
	glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
	
	if(InfoLogLength > 1){	
		fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
	}
	
	// Link the program
	if((debugLevel & 0x8) == 8){		
		std::cout << "Shader.cpp\t\tLinking program\n";
	}
	
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);
	
	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> ProgramErrorMessage( max(InfoLogLength, int(1)) );
	glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
	
	if(InfoLogLength > 1){	
		fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
	}
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);
	
	return ProgramID;
}

GLuint Shader::getID(void){
	return shaderId;
}
