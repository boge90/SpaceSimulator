#include "../include/BodyIO.hpp"
#include <iostream>
#include <fstream>

void BodyIO::read(double *time, std::vector<Body*> *bodies){
	std::cout << "BodyIO.cpp\t\tRead\n";

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
		
		while(std::getline(dataFile,line)){	
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
			int star = atoi(token.c_str());
			
			// star var is representing a boolean
			assert(star == 0 || star == 1);
			
			Body *body = new Body(glm::dvec3(posX, posY, posZ), glm::dvec3(velX, velY, velZ), glm::vec3(r, g, b), radius, mass, inclination, rotationSpeed, star);
			bodies->push_back(body);
		}
		
		dataFile.close();
	}else{
		std::cout << "Unable to open file"; 
	}
}

void BodyIO::write(double time, std::vector<Body*> *bodies){
	std::cout << "BodyIO.cpp\t\tWrite\n";

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
		double radius = body->getRadius();
		double mass = body->getMass();
		double inclination = body->getInclination();
		double rotationSpeed = body->getRotationSpeed();
		bool star = body->isStar();
		
		// Write body data
		dataFile << position.x << ", ";
		dataFile << position.y << ", ";
		dataFile << position.z << ", ";
		dataFile << velocity.x << ", ";
		dataFile << velocity.y << ", ";
		dataFile << velocity.z << ", ";
		dataFile << rgb.x << ", ";
		dataFile << rgb.y << ", ";
		dataFile << rgb.z << ", ";
		dataFile << radius << ", ";
		dataFile << mass << ", ";
		dataFile << inclination << ", ";
		dataFile << rotationSpeed << ", ";
		dataFile << star << "\n";
		
		// Free memory
		delete body;
	}

	// Closing data stream
	dataFile.close();	
}

