#ifndef BODY_H
#define BODY_H

#include <vector>

#include "../include/common.hpp"
#include "../include/Renderable.hpp"
#include "../include/BMP.hpp"
#include "../include/Shader.hpp"
#include "../include/Config.hpp"

class Body: public Renderable{
	private:
		// Data
		glm::dvec3 center;
		glm::dvec3 velocity;
		glm::dvec3 force;
		double mass;
		double radius;
		double inclination;
		double rotationSpeed;
		bool star;
		
		// Functions
		void generateVertices(std::vector<glm::dvec3> *vertices, std::vector<glm::vec3> *colors, std::vector<GLuint> *indices, int i1, int i2, int i3, int currentDepth, int finalDepth);
		
		// Body number
		static int bodyNumber;
		int bodyNum;
		
		// Rendering
		Shader *shader;
		int numVertices, numIndices;
		bool wireFrame;
		glm::vec3 rgb;
		GLuint indexBuffer;
		GLuint texCoordBuffer;
		GLuint vertexBuffer;
		GLuint colorBuffer;
		GLuint solarCoverBuffer;
		
		// Texture
		GLuint tex;
		BMP *bmp;
		
		// Misc
		Config *config;
		size_t debugLevel;
	public:
		/**
		* Creates a body
		**/
		Body(glm::dvec3 center, glm::dvec3 velocity, glm::vec3 rgb, double radius, double mass, double inclination, double rotationSpeed, bool star, Config *config);
		
		/**
		* Finalizes
		**/
		~Body();
		
		/**
		* Creates the vertex VBO, color VBO and the index VBO
		**/
		void init();
		
		/**
		* Uses OpenGL to draw this body with the current shader
		**/
		void render(const GLfloat *mvp);
		
		/**
		* Returns the center for this body
		**/
		glm::dvec3 getCenter(void);
		
		/**
		* Returns the velocity for this body
		**/
		glm::dvec3 getVelocity(void);
		
		/**
		* Returns the current force acting on the body
		**/
		glm::dvec3 getForce(void);
		
		/**
		* Returns the color
		**/
		glm::vec3 getRGB(void);
		
		/**
		* Returns the mass for this body
		**/
		double getMass(void);
		
		/**
		* Returns the radius for this body
		**/
		double getRadius(void);
		
		/**
		* sets the center position for this body
		**/
		void setCenter(glm::dvec3 center);
		
		/**
		* Sets the velocity for this body
		**/
		void setVelocity(glm::dvec3 velocity);
		
		/**
		* Sets the velocity for this body
		**/
		void setForce(glm::dvec3 force);
		
		/**
		* Returns the vertex buffer
		**/
		GLuint getVertexBuffer(void);
		
		/**
		* Returns the color buffer
		**/
		GLuint getColorBuffer(void);
		
		/**
		* Returns the solar coverage buffer
		**/
		GLuint getSolarCoverageBuffer(void);
		
		/**
		* Returns the solar coverage buffer
		**/
		GLuint getVertexSpeedBuffer(void);
		
		/**
		* Returns the number of vertices for this body
		**/
		int getNumVertices(void);
		
		/**
		* Changes the state of the wireFrame mode
		**/
		void setWireframeMode(bool active);
		
		/**
		* Returns this body's inclination (rads)
		**/
		double getInclination(void);
		
		/**
		* Retunrs this body's rotation speed (rads per second)
		**/
		double getRotationSpeed(void);
		
		/**
		* Returns true if the body is a star
		**/
		bool isStar(void);
};

#endif
