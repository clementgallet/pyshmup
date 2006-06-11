#include <python2.3/Python.h>
#include <python2.3/Numeric/arrayobject.h>
#include <math.h>
#include <stdio.h>
#define ARRAY_X 0
#define ARRAY_Y 1
#define ARRAY_DIRECTION 3
#define ARRAY_SPEED 4
#define ARRAY_STATE 6
#define ARRAY_UNTIL 7
#define ARRAY_STATE_ML 0
#define ARRAY_STATE_DANG 1
#define ARRAY_STATE_NO_DANG 2
#define RADIUS 3.0
#define PLAYER_SPEED 2.0
#define WIDTH 640
#define HEIGHT 480
#define UNIT_HEIGHT 200
#define UNIT_WIDTH (UNIT_HEIGHT * WIDTH) / HEIGHT



double dabs(double x)
{
  return ((x>0) ? x : (-1)*x);
}

double max(double x,double y)
{
  return ((x>y) ? x : y);
}

int truemod(int x,int n)
{
  int y=x;
  while (y<0)
    y += n;
  return (y%n);
} 

double SINUS_LIST[3600];
double COSINUS_LIST[3600];

static PyObject *coll(PyObject *self, PyObject *args)
{
  double x,y,rat_x,rat_y,t_x,t_y,b_x,b_y,b_d,b_s;
  int prem,size,signe_x,signe_y,i;
  PyArrayObject *array;
  PyArg_ParseTuple(args,"O!iddi",&PyArray_Type,&array,&size,&x,&y,&prem);
  
  for (i=0;i<size;i++)
    {
      b_x = *(double *)(array->data + ARRAY_X*array->strides[0] + i*array->strides[1]);
      b_y = *(double *)(array->data + ARRAY_Y*array->strides[0] + i*array->strides[1]);
      
      if ((int)*(double *)(array->data + ARRAY_STATE*array->strides[0] + i*array->strides[1]) == ARRAY_STATE_ML)
	if ((dabs(b_x - x) < RADIUS) && (dabs(b_y - y)< RADIUS))
	  return (Py_BuildValue("i",i));
      
      if ((int)*(double *)(array->data + ARRAY_STATE*array->strides[0] + i*array->strides[1]) == ARRAY_STATE_DANG)
	{	      
	  if (*(double *)(array->data + ARRAY_UNTIL*array->strides[0] + i*array->strides[1]) > 0)
	    (*(double *)(array->data + ARRAY_UNTIL*array->strides[0] + i*array->strides[1]))--;
	  else
	    {
	      b_d = *(double *)(array->data + ARRAY_DIRECTION*array->strides[0] + i*array->strides[1]);
	      b_s = *(double *)(array->data + ARRAY_SPEED*array->strides[0] + i*array->strides[1]);
	      
	      
	      
	      (*(double *)(array->data + ARRAY_STATE*array->strides[0] + i*array->strides[1])) = (double)(ARRAY_STATE_NO_DANG);
	      if (dabs(b_x - x) > RADIUS)
		{
		  signe_x = (b_x > x) + (b_x >= x ) - 1;
		  rat_x = SINUS_LIST[truemod((int)(10*b_d),3600)]*b_s - (double)(signe_x * PLAYER_SPEED);
		  if (rat_x != 0)
		    {
		      t_x = (signe_x * RADIUS - b_x + x) / rat_x;
		      if (t_x >= 0 && (-1)*UNIT_WIDTH < (x + signe_x * t_x * PLAYER_SPEED) && (x + signe_x * t_x * PLAYER_SPEED) < UNIT_WIDTH)
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
				      (*(double *)(array->data + ARRAY_STATE*array->strides[0] + i*array->strides[1])) = (double)(ARRAY_STATE_DANG);
				      (*(double *)(array->data + ARRAY_UNTIL*array->strides[0] + i*array->strides[1])) = max (t_x, t_y);
				    }
				}
			    }
			  else
			    {
			      (*(double *)(array->data + ARRAY_STATE*array->strides[0] + i*array->strides[1])) = (double)(ARRAY_STATE_DANG);
			      (*(double *)(array->data + ARRAY_UNTIL*array->strides[0] + i*array->strides[1])) = t_x;
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
			  (*(double *)(array->data + ARRAY_STATE*array->strides[0] + i*array->strides[1])) = (double)(ARRAY_STATE_DANG + 0.5);
			  (*(double *)(array->data + ARRAY_UNTIL*array->strides[0] + i*array->strides[1])) = t_y;
			}
		    }
		}
	      else
		return (Py_BuildValue("i",i));
	    }
	}
    }
  return (Py_BuildValue("i",-1));
}

static PyMethodDef DrawMethods[] = {
	{"coll", coll, METH_VARARGS, "Search for colisions"},
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
