#ifndef BODY_H
#define BODY_H

#include <vector>
#include <string>

class Body;

#include "../include/BMP.hpp"
#include "../include/common.hpp"
#include "../include/Shader.hpp"
#include "../include/Config.hpp"
#include "../include/Renderable.hpp"
#include "../include/Atmosphere.hpp"

enum BodyType{PLANET, STAR, COMET};

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
		double rotation;
		BodyType bodyType;
		std::string texturePath;
		std::string name;
		int lod;
		
		// Functions
		void calculateVertices(std::vector<glm::vec3> *vertices, std::vector<glm::vec3> *colors, std::vector<GLuint> *indices, int i1, int i2, int i3, int currentDepth, int finalDepth);
		
		// Body number
		static int bodyNumber;
		int bodyNum;
		
		// Rendering
		Shader *shader;
		size_t numVertices, numIndices;
		bool wireFrame;
		glm::vec3 rgb;
		GLuint indexBuffer;
		GLuint texCoordBuffer;
		GLuint vertexBuffer;
		GLuint colorBuffer;
		GLuint solarCoverBuffer;
		
		// Atmosphere
		Atmosphere *atmosphere;
		glm::vec3 atmosphereColor;
		
		// Texture
		GLuint tex;
		
		// Misc
		Config *config;
		size_t debugLevel;
	public:
		/**
		* Creates a body
		**/
		Body(std::string name, glm::dvec3 center, glm::dvec3 velocity, glm::vec3 rgb, glm::vec3 atmosphereColor, double rotation, double radius, double mass, double inclination, double rotationSpeed, BodyType bodyType, std::string texturePath, Config *config);
		
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
		void render(glm::mat4 *mvp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up);
		
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
		* Returns the color
		**/
		glm::vec3 getAtmosphereColor(void);
		
		/**
		* Returns the mass for this body
		**/
		double getMass(void);
		
		/**
		* Returns the radius for this body
		**/
		double getRadius(void);
		
		/**
		* Sets the current rotation of the body in radians
		**/
		void setRotation(double rotation);
		
		/**
		* Roturns the current rotation of the body in radians
		**/
		double getRotation(void);
		
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
		* Returns the current Level Of Detail
		**/
		int getLOD(void);
		
		/**
		* sets the current Level Of Detail
		**/
		void setLOD(int lod);
		
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
		* Returns the vertex index buffer
		**/
		GLuint getVertexIndexBuffer(void);
		
		/**
		* Returns the number of indices in theindex buffer
		**/
		size_t getNumIndices(void);
		
		/**
		* Returns the number of vertices for this body
		**/
		int getNumVertices(void);
		
		/**
		* Generates a set of vertices and updates the buffers
		**/
		void generateVertices(int depth);
		
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
		
		/**
		* Returns true if the body is a star
		**/
		BodyType getBodyType(void);
		
		/**
		* Returns the relative path of the texture for this body
		**/
		std::string* getTexturePath(void);
		
		/**
		* Returns the name of this body
		**/
		std::string* getName(void);
};

#endif
