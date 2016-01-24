#ifndef SCROLL_LAYOUT_H
#define SCROLL_LAYOUT_H

#include "../include/DrawService.hpp"
#include "../include/Layout.hpp"
#include "../include/Config.hpp"

class ScrollLayout: public Layout{
	protected:
		double scrolled_x;
		double scrolled_y;

		bool has_been_scrolled;

	public:
		/**
		* Creates an empty list layout object
		**/
		ScrollLayout(int x, int y, int width, int height, Config *config);
		
		/**
		* Finalizes this ListLayout object
		**/
		~ScrollLayout();
		
		/**
		* Scroll
		**/ 
		virtual void scroll( double xoffset, double yoffset );
		
		/**
		* Executed when the view should be drawn
		**/
		void draw(DrawService *draw);
};

#endif
