#include "../include/BodyTracer.hpp"
#include <iostream>

BodyTracer::BodyTracer(std::vector<Body*> *bodies, double dt){	
	// DEBUG
	std::cout << "BodyTracer.cpp\t\tInitializing" << std::endl;

	// Init
	this->active = false;
	this->bodyNum = -1;
	this->numVertices = 0;
	this->MAX_VERTICES = 5000000;
	this->dt = dt;
	this->G = 6.7 * pow(10, -11);
	this->bodies = bodies;
	
	// Creating OpenGL buffer
	glGenBuffers(1, &vertexBuffer);
	glGenBuffers(1, &colorBuffer);
	
	// Shader
	shader = new Shader("src/shaders/vertex.glsl", "src/shaders/fragment.glsl");
}

BodyTracer::~BodyTracer(void){
	std::cout << "BodyTracer.cpp\t\tFinalizing" << std::endl;
}

void BodyTracer::render(const GLfloat *mvp){	
	if(active){	
		// Shader
		shader->bind();
	
		// Get a handle for our "MVP" uniform.
		GLuint mvpMatrixId = glGetUniformLocation(shader->getID(), "MVP");
		glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, mvp);
	
		// Binding vertex VBO
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	
		// Binding color VBO
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	
		// Draw
		glDrawArrays(GL_LINES, 0, numVertices);
	
		// Disabling buffers
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void BodyTracer::setActive(bool active){
	this->active = active;
	if(active){
		std::cout << "BodyTracer.cpp\t\tTurning ON path visualization" << std::endl;
	}else{
		std::cout << "BodyTracer.cpp\t\tTurning OFF path visualization" << std::endl;
	}
}

void BodyTracer::calculateFuturePath(size_t bodyNum){
	// Num bodies
	size_t size = bodies->size();
	if(bodyNum >= size){
		std::cout << "Body number " << bodyNum << " does not exist" << std::endl;
		return;
	}
	
	// Bodies data
	glm::dvec3 startVertex;
	glm::dvec3 positions[size];
	glm::dvec3 forces[size];
	glm::dvec3 velocity[size];
	double masses[size];
	
	// Path for the body to calculate
	double maxSpeed = 0.0;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> colors;
	
	// Init arrays
	for(size_t i=0; i<size; i++){
		velocity[i] = (*bodies)[i]->getVelocity();
		positions[i] = (*bodies)[i]->getCenter();
		masses[i] = (*bodies)[i]->getMass();
		forces[i] = glm::vec3(0, 0, 0);
	}
	startVertex = positions[bodyNum];

	// Calculating future path	
	size_t num=0;
	std::cout << "BodyTracer.cpp\t\tCalculating future path for body " << bodyNum << std::endl;
	double t0 = glfwGetTime();
	while(num < MAX_VERTICES){	
		// Calculate force
		for(size_t i=0; i<size; i++){
			for(size_t j=0; j<size; j++){
				if(i != j){				
					double dist_x = positions[j].x - positions[i].x;
					double dist_y = positions[j].y - positions[i].y;
					double dist_z = positions[j].z - positions[i].z;

					double abs_dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
					double dist_cubed = abs_dist*abs_dist*abs_dist;

					glm::dvec3 force = forces[i];
					force.x += (G * masses[i] * masses[j])/dist_cubed * dist_x;
					force.y += (G * masses[i] * masses[j])/dist_cubed * dist_y;
					force.z += (G * masses[i] * masses[j])/dist_cubed * dist_z;
					forces[i] = force;
				}
			}
		}
	
		// Update position
		for(size_t i=0; i<size; i++){
			// Retrive calculated force
			glm::dvec3 f = forces[i];
		
			// Using current center and delta to generate master matrix
			glm::dvec3 vel = velocity[i];
			glm::dvec3 center = positions[i];
			glm::dvec3 delta;
		
			delta.x = dt*velocity[i].x;
			delta.y = dt*velocity[i].y;
			delta.z = dt*velocity[i].z;
		
			// Updating new center
			center.x += delta.x;
			center.y += delta.y;
			center.z += delta.z;
			positions[i] = center;
		
			// Adding vertex to vertexBuffer
			if(i == bodyNum){
				vertices.push_back(glm::vec3(center));
				colors.push_back(glm::vec3(vel));
				num++;
			}
			
			// Max speed
			double speed = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
			if(speed > maxSpeed && i == bodyNum){
				maxSpeed = speed;
			}
		
			// Updating new velocity
			vel.x += dt * f.x/masses[i];
			vel.y += dt * f.y/masses[i];
			vel.z += dt * f.z/masses[i];
			velocity[i] = vel;
		
			// Reset force for next iteration
			forces[i] = glm::vec3(0, 0, 0);
		}
	}	
	
	// DEBUG
	std::cout << "BodyTracer.cpp\t\tCalculated " << vertices.size() << " vertices in " << glfwGetTime()-t0 << " seconds" << std::endl;
	std::cout << "BodyTracer.cpp\t\tThe track represents the path for the next " << (num*dt)/(3600.0*24.0*365.242199) << " earth years" << std::endl;
	
	// Vertex count used for rendering
	numVertices = vertices.size();
	this->bodyNum = bodyNum;

	// DEBUG
	std::cout << "BodyTracer.cpp\t\tVertices and colors are using " << (2*numVertices*3*sizeof(float))/(1024*1024) << " MiB" << std::endl;
	
	// Setting vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*3*sizeof(float), &(vertices.front()), GL_STATIC_DRAW);
    
    // Modifying colors HSV color model
    for(size_t i=0; i<numVertices; i++){
    	glm::vec3 vel = colors[i];
    	double value = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
    	
		int j;
		float f, p, q, t;
		float s = 1.f;
		float v = 1.f;
		float h = 240.f - 240.f*(value/maxSpeed);
	
		if(value > maxSpeed){value = maxSpeed;}

		h /= 60;			// sector 0 to 5
		j = floor(h);
		f = h - j;			// factorial part of h
		p = v * (1 - s);
		q = v * (1 - s*f);
		t = v * (1 - s*(1 - f));
		switch( j ) {
			case 0:
				colors[i] = glm::vec3(v, t, p);
				break;
			case 1:
				colors[i] = glm::vec3(q, v, p);
				break;
			case 2:
				colors[i] = glm::vec3(p, v, t);
				break;
			case 3:
				colors[i] = glm::vec3(p, q, v);
				break;
			case 4:
				colors[i] = glm::vec3(t, p, v);
				break;
			default:		// case 5:
				colors[i] = glm::vec3(v, p, q);
				break;
		}
    }
    
    // Setting colors
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*3*sizeof(float), &(colors.front()), GL_STATIC_DRAW);
}
