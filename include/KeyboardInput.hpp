#ifndef KEYBOARD_INPUT_H
#define KEYBOARD_INPUT_H

#include "../include/KeyboardInputAction.hpp"
#include <vector>
#include <stdlib.h>

class KeyboardInput{
	private:		
		// Active listeners
		std::vector<KeyboardInputAction*> *listeners;
		
		// instance
		static KeyboardInput *instance;
		
		/**
		* Finds the index of the action in the list of listeners
		* -1 is returned if the action is not found
		**/
		int getIndex(KeyboardInputAction *action);
		
		/**
		* Creates the KeyboardInput singleton object
		**/
		KeyboardInput(void);
		
		/**
		* Finalizes the KeyboardInput singleton object
		**/
		~KeyboardInput(void);
	public:
		/**
		* Called when a key is pressed
		**/
		void addInput(int key);
		
		/**
		* Adds a action that is called when a input is added
		**/
		void addInputAction(KeyboardInputAction *action);
		
		/**
		* Removes the action from the listeners
		**/
		void removeInputAction(KeyboardInputAction *action);
	
		/**
		* Returns the pointer to the singleton object
		**/
		static KeyboardInput* getInstance(void);
	
		/**
		* Deletes the singleton object
		**/
		static void destroy(void);
};

#endif
