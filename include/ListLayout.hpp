#ifndef LIST_LAYOUT_H
#define LIST_LAYOUT_H

#include "../include/DrawService.hpp"
#include "../include/ScrollLayout.hpp"
#include "../include/Config.hpp"
#include <vector>

class ListLayout: public ScrollLayout{
	private:
		
	public:
		/**
		* Creates an empty list layout object
		**/
		ListLayout(int x, int y, int width, int height, Config *config);
		
		/**
		* Finalizes this ListLayout object
		**/
		~ListLayout();
		
		/**
		* Layout class function
		**/ 
		virtual void addChild(View *view);
		
		/**
		* Scroll
		**/ 
		void scroll( double xoffset, double yoffset );
};

#endif
