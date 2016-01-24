#ifndef KEYBOARD_H
#define KEYBOARD_H

#include <vector>
#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include "../include/KeyboardInputAction.hpp"

class Keyboard: public KeyboardInputAction{
	private:
		// Misc
		size_t activeBody;
		size_t debugLevel;
		
		// Data
		std::vector<Body*> *bodies;
		
		/**
		* Called when the user presses the keyboard
		**/
		void onKeyInput(int key);
		
		/**
		* Called when the user presses the 'N' button, causing the keyboard behaviour
		* to focus on the next body
		**/
		void nextBody();
		
		/**
		* Called when the user presses the 'P' button, causing the keyboard behaviour
		* to focus on the previous body
		**/
		void previousBody();
	public:
		/**
		* Creates the keyboard listener / acter
		**/
		Keyboard(std::vector<Body*> *bodies, Config *config);
		
		/**
		* Destroyes the object
		**/
		virtual ~Keyboard(void);
};

#endif
