#include "../include/BodyIO.hpp"
#include <iostream>
#include <fstream>

void BodyIO::read(double *time, std::vector<Body*> *bodies, Config *config){
	if((config->getDebugLevel() & 0x8) == 8){	
		std::cout << "BodyIO.cpp\t\tRead\n";
	}

	std::string line;
	std::ifstream dataFile("bodies.data");
	if (dataFile.is_open()){
		// Reading time
		std::getline(dataFile,line);
		double t0 = atof(line.c_str());
		*time = t0;
		
		// Reading bodies
		size_t pos;
		std::string delimiter = ",";
		std::string token;
		std::string texturePath;
		std::string name;
		
		while(std::getline(dataFile,line)){	
			pos = line.find(delimiter);
			name = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double posX = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double posY = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double posZ = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double velX = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double velY = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double velZ = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			float r = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			float g = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			float b = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double rotation = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double radius = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double mass = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double inclination = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			double rotationSpeed = atof(token.c_str());
			
			pos = line.find(delimiter);
			token = line.substr(0, pos);
		    line.erase(0, pos + delimiter.length());
			int bodyType = atoi(token.c_str());
			
			// Only stars and planets have textures
			if(bodyType < 2){			
				pos = line.find(delimiter);
				texturePath = line.substr(1, pos);
				line.erase(0, pos + delimiter.length());
			}
			
			Body *body = new Body(name, glm::dvec3(posX, posY, posZ), glm::dvec3(velX, velY, velZ), glm::vec3(r, g, b), rotation, radius, mass, inclination, rotationSpeed, static_cast<BodyType>(bodyType), texturePath, config);
			bodies->push_back(body);
		}
		
		dataFile.close();
	}else{
		std::cout << "Unable to open file"; 
	}
}

void BodyIO::write(double time, std::vector<Body*> *bodies, Config *config){
	if((config->getDebugLevel() & 0x8) == 8){	
		std::cout << "BodyIO.cpp\t\tWrite\n";
	}

	// Opening data file
	std::ofstream dataFile;
	dataFile.open ("bodies.data");
	
	// Write body data
	dataFile << time << "\n";

	int size = bodies->size();
	for(int i=0; i<size; i++){
		// Get body data
		Body *body = (*bodies)[i];
		glm::dvec3 position = body->getCenter();
		glm::dvec3 velocity = body->getVelocity();
		glm::vec3 rgb = body->getRGB();
		BodyType type = body->getBodyType();
		
		// Write body data
		dataFile << body->getName()->c_str() << ", ";
		dataFile << position.x << ", ";
		dataFile << position.y << ", ";
		dataFile << position.z << ", ";
		dataFile << velocity.x << ", ";
		dataFile << velocity.y << ", ";
		dataFile << velocity.z << ", ";
		dataFile << rgb.x << ", ";
		dataFile << rgb.y << ", ";
		dataFile << rgb.z << ", ";
		dataFile << body->getRotation() << ", ";
		dataFile << body->getRadius() << ", ";
		dataFile << body->getMass() << ", ";
		dataFile << body->getInclination() << ", ";
		dataFile << body->getRotationSpeed() << ", ";
	
		dataFile << type;		
		if(type < 2){			
			dataFile << ", " << body->getTexturePath()->c_str() << "\n";
		}else{
			dataFile << "\n";
		}
		
		// Free memory
		delete body;
	}

	// Closing data stream
	dataFile.close();	
}

