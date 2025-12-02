#pragma once

struct cpuData
{
	double readtime;
	double distance_t;
	double total_t;

	cpuData(double r = 0, double d = 0, double t = 0) : readtime(r), distance_t(d), total_t(t) {}

	double get_time(int flag) const
	{
		switch (flag)
		{
		case 0: return readtime;
		case 1: return distance_t;
		case 2: return total_t;
		default: return 0;
		}
	}
};

struct gpuData
{
	double readtime;
	double h2d_t;
	double popcount_t;
	double d2h_t;
	double total_t;
	double* ptr[5] = { nullptr, nullptr , nullptr , nullptr , nullptr };

	gpuData(double r = 0, double h = 0, double p = 0, double d = 0, double t = 0) : readtime(r), h2d_t(h), popcount_t(p), d2h_t(d), total_t(t) {}
	
	double get_time(int flag) const
	{
		switch (flag)
		{
			case 0: return readtime;
			case 1: return h2d_t;
			case 2: return popcount_t;
			case 3: return d2h_t;
			case 4: return total_t;
			default: return 0;
		}
	}
};