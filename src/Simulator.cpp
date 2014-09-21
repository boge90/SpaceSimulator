#include "../include/Simulator.hpp"
#include <iostream>

Simulator::Simulator(double time, double dt, std::vector<Body*> *bodies){
	// Debug
	std::cout << "Simulator.cpp\t\tInitializing\n";

	// Init
	this->time = time;
	this->dt = dt;
	this->bodies = bodies;
	this->simulationSteps = 0;
	
	// Initializing visualization system
	renderer = new Renderer(this);
	frame = new Frame(1500, 900, "Space", renderer, this);
	
	// Setting bodies for rendering and initializes them
	for(unsigned int i=0; i<bodies->size(); i++){
		(*bodies)[i]->init();
		renderer->addRenderable((*bodies)[i]);
	}
	
	// Initializing sub renderers
	bodyTracer = new BodyTracer(bodies, dt);
	renderer->addRenderable(bodyTracer);
	
	// Initializing sub simulators
	nbody = new Nbody(bodies, dt);
	rayTracer = new RayTracer(bodies);
}

Simulator::~Simulator(){
	// Debug
	std::cout << "Simulator.cpp\t\tFinalizing after executing " << simulationSteps << " simulations steps (" << (simulationSteps*dt)/(3600.0) << " hours)\n";
	
	// Free
	delete frame;
	delete renderer;
	delete nbody;
	delete rayTracer;
	delete bodyTracer;
}

void Simulator::simulate(void){	
	// Sub - simulations
	nbody->simulateGravity();
	rayTracer->simulateRays();

	// Check user input
	frame->getMenu()->getActivatedCamera()->checkUserInput();

	// Misc
	simulationSteps++;
	time += dt;
	
	// Print time
	if(simulationSteps % 1000 == 0){
		std::cout << "Simulator.cpp\t\tTime = " << time << "\n";
	}
	
	// Update visualization
	frame->update();
}

double Simulator::getTime(void){
	return time;
}

std::vector<Body*>* Simulator::getBodies(void){
	return bodies;
}

bool Simulator::isRunning(){
	return !frame->isClosed();
}

Nbody* Simulator::getNbodySimulator(void){
	return nbody;
}

RayTracer* Simulator::getRayTracerSimulator(void){
	return rayTracer;
}

BodyTracer* Simulator::getBodyTracer(void){
	return bodyTracer;
}
