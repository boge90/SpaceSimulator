#ifndef DELTA_TIME_CHECKER_H
#define DELTA_TIME_CHECKER_H

#include "../include/Config.hpp"
#include <stdio.h>
#include <time.h>

class DeltaTimeChecker{
	private:
		double *dt;

	public:
		/*
		 *
		 */
		DeltaTimeChecker(Config *config);

		/*
		 *
		 */
		~DeltaTimeChecker( void );

		/*
		 *
		 */
		void update_delta_time_if_needed( double fps, double simulation_time );
};

#endif
