#include <Python.h>
#include <Numeric/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "constants.h"

// When the state is at ARRAY_STATE_UNKNOWN,
// each player put the dangerousness of the bullet toward it, 
// and raise by one the ARRAY_STATE value
// until the value ARRAY_STATE_DANG + #(players)
// so that all the players have treated the bullet state


inline double dabs(double x)
{
	return ((x>0) ? x : (-1)*x);
}

inline double max(double x,double y)
{
	return ((x>y) ? x : y);
}

inline double min(double x,double y)
{
	return ((x<y) ? x : y);
}

inline int truemod(int x,int n)
{
	int y=x%n;
	if (y<0)
		y += n;
	return y;
} 

double SINUS_LIST[3600];
double COSINUS_LIST[3600];

#define array_elem(type) (*(double *)(array->data + ARRAY_##type*array->strides[0] + i*array->strides[1]))

static PyObject *coll(PyObject *self, PyObject *args)
{
	double x,y;
	double rat_x,rat_y;
	double t_x,t_y;
	double b_x,b_y;
	double b_d,b_s;

	int nb_players;
	int size;
	int signe_x,signe_y;
	int i;

	PyObject *coll_indices = PyList_New(0);
	PyObject *out_indices = PyList_New(0);

	PyArrayObject *array;
	PyArg_ParseTuple(args,"O!iddi",&PyArray_Type,&array,&size,&x,&y,&nb_players);

	for (i=0;i<size;i++)
	{
		b_x = array_elem(X);
		b_y = array_elem(Y);

		array_elem(OUT_TIME)--;
		if (array_elem(OUT_TIME) < 0)
			if (array_elem(LIST_INDEX) <= 0)
			{
				PyList_Append(out_indices, Py_BuildValue("i", i));
				continue;
			}

		/* Check collision for ML bullets */
		if ((int) array_elem(STATE) == ARRAY_STATE_ML)
			if ((dabs(b_x - x) < RADIUS) && (dabs(b_y - y)< RADIUS))
			{
				PyList_Append(coll_indices, Py_BuildValue("i",i));
				continue;
			}

		
		if ((int) array_elem(STATE) >= ARRAY_STATE_DANG)
		{
				b_d = array_elem(DIRECTION);
				b_s = array_elem(SPEED);

				array_elem(STATE) = (double)(ARRAY_STATE_NO_DANG);
				if (dabs(b_x - x) > RADIUS)
				{
					signe_x = (b_x > x) + (b_x >= x ) - 1;
					rat_x = SINUS_LIST[truemod(10*b_d, 3600)]*b_s - signe_x*PLAYER_SPEED;
					if (rat_x != 0)
					{
						t_x = (signe_x * RADIUS - b_x + x) / rat_x;
						if (t_x >= 0 && dabs(x + signe_x * t_x * PLAYER_SPEED) < UNIT_WIDTH)
						{
							if (dabs(b_y - y) > RADIUS)
							{
								signe_y = (b_y > y) + (b_y >= y) - 1;
								rat_y = (-1)*COSINUS_LIST[truemod((int)(10*b_d),3600)]*b_s - signe_y * PLAYER_SPEED;
								if (rat_y != 0)
								{
									t_y = (signe_y * RADIUS - b_y + y) / rat_y;
									if (t_y >= 0)
									{
										if ((int) array_elem(STATE) >= ARRAY_STATE_DANG + nb_players - 1) 
											array_elem(STATE) = (double)(ARRAY_STATE_DANG);
										else
											array_elem(STATE)++;

										array_elem(UNTIL) = min(max(t_x, t_y), array_elem(UNTIL));
									}
								}
							}
							else
							{
								if ((int)array_elem(STATE) >= ARRAY_STATE_DANG + nb_players - 1)
									array_elem(STATE) = (double)(ARRAY_STATE_DANG);
								else
									array_elem(STATE)++;

								array_elem(UNTIL) = min(t_x, array_elem(UNTIL));
							}
						}
					}
				}
				else if (dabs(b_y - y) > RADIUS)
				{
					signe_y = (b_y > y) + (b_y >= y) - 1;
					rat_y = (-1)*COSINUS_LIST[truemod((int)(10*b_d),3600)]*b_s - signe_y * PLAYER_SPEED;
					if (rat_y != 0)
					{
						t_y = (signe_y * RADIUS - b_y + y) / rat_y;
						if (t_y >= 0)
						{
							if ((int)array_elem(STATE) >= ARRAY_STATE_DANG + nb_players - 1)
								array_elem(STATE) = (double)(ARRAY_STATE_DANG);
							else
								array_elem(STATE)++;

							array_elem(UNTIL) = min(t_y, array_elem(UNTIL));
						}
					}
				}
				else
					PyList_Append(coll_indices, Py_BuildValue("i", i));
			}
		
	}

	return Py_BuildValue("(OO)", coll_indices, out_indices);
}

static PyMethodDef DrawMethods[] = {
	{"coll", coll, METH_VARARGS, "Search for collisions"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcoll(void)
{
	int i;
	(void) Py_InitModule("coll", DrawMethods);
	import_array();
	for (i=0;i<3600;i++)
	{
		SINUS_LIST[i] = sin(((double)i*M_PI)/(double)1800);
		COSINUS_LIST[i] = cos(((double)i*M_PI)/(double)1800);
	}
}
