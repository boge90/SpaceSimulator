#ifndef VIEW_H
#define VIEW_H

class View;

#include "../include/ViewClickedAction.hpp"
#include "../include/DrawService.hpp"
#include <vector>

class View{
	private:		
		// View data
		int x;
		int y;
		int width;
		int height;
		
		// Color
		unsigned char red;
		unsigned char green;
		unsigned char blue;
		
		// Listeners
		std::vector<ViewClickedAction*> *clickActions;
	public:
		/**
		* Creates a view object at the specified location
		**/
		View(int x, int y, int width, int height);
		
		/**
		* Destroys the view object, and will recurrsivly destroy children
		**/
		~View();
		
		/**
		* Adds an click action which is triggered when the view is clicked
		**/
		void addViewClickedAction(ViewClickedAction *action);
		
		/**
		* Updates this view
		**/
		void draw(DrawService *service);
		
		/**
		* returns true if the input coordinates is inside this view
		**/
		bool isInside(int x, int y);
		
		/**
		* Called when the view has been clicked on
		**/
		void clicked(int button, int action);
};

#endif
