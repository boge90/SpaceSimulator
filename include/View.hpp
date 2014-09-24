#ifndef VIEW_H
#define VIEW_H

class View;

#include "../include/ViewClickedAction.hpp"
#include "../include/DrawService.hpp"
#include "../include/Config.hpp"
#include <vector>

class View{
	private:				
		// Listeners
		std::vector<ViewClickedAction*> *clickActions;
		
		/**
		* Initializes the view
		**/
		void init(int x, int y, int width, int height, Config *config);
	protected:
		// View data
		int x;
		int y;
		int width;
		int height;
		
		// Misc
		size_t debugLevel;
		
		// Color
		unsigned char red;
		unsigned char green;
		unsigned char blue;
	public:
		/**
		* Creates a view object which has to be added to a Layout view before is can be used
		* Widt of this object will MATCH the parent view
		**/
		View(int height, Config *config);
		
		/**
		* Creates a view object which has to be added to a Layout view before is can be used
		**/
		View(int width, int height, Config *config);
		
		/**
		* Creates a view object at the specified location
		**/
		View(int x, int y, int width, int height, Config *config);
		
		/**
		* Destroys the view object, and will recurrsivly destroy children
		**/
		virtual ~View();
		
		/**
		* Adds an click action which is triggered when the view is clicked
		**/
		void addViewClickedAction(ViewClickedAction *action);
		
		/**
		* Updates this view
		**/
		virtual void draw(DrawService *service);
		
		/**
		* returns true if the input coordinates is inside this view
		**/
		bool isInside(int x, int y);
		
		/**
		* Called when the view has been clicked on
		**/
		virtual void clicked(int button, int action);
		
		/**
		* Updates the X coordinate of the view
		**/
		void setX(int x);
		
		/**
		* Returns the X coordinate of the view
		**/
		int getX(void);
		
		/**
		* Updates the Y coordinate of the view
		**/
		void setY(int y);
		
		/**
		* Returns the Y coordinate of the view
		**/
		int getY(void);
		
		/**
		* Updates the width of this view
		**/
		void setWidth(int width);
		
		/**
		* Returns the width of the view
		**/
		int getWidth(void);
		
		/**
		* Updates the height of this view
		**/
		void setHeight(int height);
		
		/**
		* Returns the height of the view
		**/
		int getHeight(void);
};

#endif
