#ifndef LAYOUT_H
#define LAYOUT_H

#include "../include/View.hpp"
#include "../include/Config.hpp"
#include <vector>

class Layout: public View{
	protected:
		std::vector<View*> *children;	
		
	public:
		/**
		*
		**/
		Layout(int x, int y, int width, int height, Config *config);
		
		/**
		*
		**/
		~Layout();
		
		/**
		* Executed when the view should be drawn
		**/
		void draw(DrawService *draw);
		
		/**
		* Adds a child view to this layout
		**/
		virtual void addChild(View *view);
		
		/**
		* Returns the pointer to the list of children
		**/
		std::vector<View*>* getChildren(void);
};

#endif
