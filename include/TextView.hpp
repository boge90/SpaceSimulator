#ifndef TEXT_VIEW_H
#define TEXT_VIEW_H

#include "../include/View.hpp"
#include <string>

class TextView: public View{
	private:
		std::string text;
		
	public:
		/**
		*
		**/
		TextView(std::string text);
		
		/**
		*
		**/
		~TextView(void);
		
		/**
		*
		**/
		void draw(DrawService *drawService);
};

#endif
