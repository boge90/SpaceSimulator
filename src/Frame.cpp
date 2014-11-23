#include "../include/Frame.hpp"

int Frame::frameWidth = 0;
int Frame::frameHeight = 0;
Frame* Frame::instance;

Frame::Frame(int width, int height, const char *title, Renderer *renderer, Simulator *simulator, Config *config){	
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){
		std::cout << "Frame.cpp\t\tInitializing\n";
	}
	
	// Fullscreen
	GLFWmonitor *monitor = NULL;
	if(config->isFullscreen()){
		monitor = glfwGetPrimaryMonitor();
	}
	
	// Open a window and create its OpenGL context
	this->window = glfwCreateWindow(width, height, title, monitor, NULL);
	glfwMakeContextCurrent(window);
	
	// This
	Frame::instance = this;
	this->simulator = simulator;
	this->totalFrameCount = 0;
	this->lastSecondFrameCount = 0;
	this->renderer = renderer;
	this->frameWidth = width;
	this->frameHeight = height;
	this->prevTime = 0;
	this->hud = new HUD(window, simulator, this, config);
	this->keyboardInput = KeyboardInput::getInstance();
	
	// Without this vertex array which is not even used, nothing is displayed
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

	// Enabling CULLING of back faces
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	
	// Blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	// GL
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	
	// Callback functions
	glfwSetMouseButtonCallback(window, mouseCallback);
	glfwSetWindowCloseCallback(window, windowCloseCallback);
	glfwSetWindowSizeCallback(window, windowSizeChangeCallback);
	glfwSetKeyCallback(window, keyCallback);
}

Frame::~Frame(void){
	if((debugLevel & 0x10) == 16){		
		std::cout << "Frame.cpp\t\tFinalizing\n";
	}

	glDeleteBuffers(1, &VertexArrayID);

	delete hud;

	glfwTerminate();
}

void Frame::update(void){
	// Misc
	totalFrameCount++;
	lastSecondFrameCount++;

	// Reset transformations and Clear
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Getting view and projection matrix, and camera attributes
	glm::mat4 vp = hud->getActivatedCamera()->getVP();
	glm::dvec3 position = hud->getActivatedCamera()->getPosition();
	glm::dvec3 direction = hud->getActivatedCamera()->getDirection();
	glm::dvec3 up = hud->getActivatedCamera()->getUp();
	
	// Rendering
	renderer->render(&vp, position, direction, up);
	
	// Draw menu if present
	hud->render();

	// Swap buffers
	glfwSwapBuffers(window);
	
	// Events
	glfwPollEvents();
	
	// Calculate FPS
	double currTime = glfwGetTime();
	if((currTime - prevTime) > 1.0){
		fps = lastSecondFrameCount/(currTime - prevTime);
		prevTime = currTime;
		lastSecondFrameCount = 0;
		
		if((debugLevel & 0x80) == 128){
			std::cout << "Frame.cpp\t\tFPS " << fps << "\n";
		}
	}
}

bool Frame::isClosed(void){
	return glfwWindowShouldClose(window);
}

long unsigned int Frame::getFrameCount(void){
	return totalFrameCount;
}

double Frame::getFPS(void){
	return fps;
}

void Frame::windowCloseCallback(GLFWwindow* window){
	glfwSetWindowShouldClose(window, GL_TRUE);
}

void Frame::windowSizeChangeCallback(GLFWwindow* window, int width, int height){
	Frame::frameWidth = width;
	Frame::frameHeight = height;
	glViewport(0, 0, width, height);
}

void Frame::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
		Frame::instance->hud->toggleVisibility();
	}
	
	// Keyboard input
	if(key != GLFW_KEY_ESCAPE && action == GLFW_PRESS){	
		Frame::instance->keyboardInput->addInput(key);
	}
}

void Frame::mouseCallback(GLFWwindow *window, int button, int action, int mods){
	// Getting mouse position
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	
	// Passing data to menu
	Frame::instance->hud->hudClicked(button, action, xpos, ypos);
}

HUD* Frame::getHud(){
	return hud;
}

Simulator* Frame::getSimulator(void){
	return simulator;
}

int Frame::getWidth(void){
	return frameWidth;
}

int Frame::getHeight(void){
	return frameHeight;
}

GLFWwindow* Frame::getWindow(void){
	return window;
}
