#include "../include/DeltaTimeChecker.hpp"

DeltaTimeChecker::DeltaTimeChecker(Config *config)
{
	this->dt = config->getDt();
}

DeltaTimeChecker::~DeltaTimeChecker( void )
{

}

void DeltaTimeChecker::update_delta_time_if_needed( double fps, double simulation_time )
{
	// Finding now since 01-01-2000
	time_t now = time(0);
	now = now - ( 30 * 365.242199 * 24 * 3600 );
	now -= 4 * 24 * 3600;
	now += 7 * 3600;
	now += 20 * 60;

	// Finding seconds of simulation time until NOW is reached
	double sec = (now - simulation_time) / (fps * (*dt));

	// Half the dt if its less than 2 seconds to NOW
	if ( sec < 2.0 )
	{
		*dt = (*dt) / 2.0;

		// Don't reduce below 1/FPS, else NOW will run away from us
		if ( *dt < 1.0/fps )
		{
			*dt = 1.0/fps;
		}
	}
}
