#include <python2.3/Python.h>
#include <python2.3/Numeric/arrayobject.h>
#include <GL/gl.h>
#define ARRAY_X 0
#define ARRAY_Y 1
#define ARRAY_Z 2
#define ARRAY_LIST 5

static PyObject *draw(PyObject *self, PyObject *args)
{
	int card,j;
	PyArrayObject *array;
	PyArg_ParseTuple(args,"O!i",&PyArray_Type,&array,&card);
	
	for (j=0;j<card;j++)
	  {
	    glPushMatrix();
	    glColor4f(1.0, 1.0, 1.0, 0.2);
	    glTranslatef((*(double *)(array->data + ARRAY_X*array->strides[0] + j*array->strides[1])),(*(double *)(array->data + ARRAY_Y*array->strides[0] + j*array->strides[1])), (*(double *)(array->data + ARRAY_Z*array->strides[0] + j*array->strides[1])));
	    glCallList((int)*(double*)(array->data + ARRAY_LIST*array->strides[0] + j*array->strides[1]));
	    glPopMatrix();
	  }
	
	return Py_INCREF(Py_None), Py_None;
}

static PyMethodDef DrawMethods[] = {
	{"draw", draw, METH_VARARGS, "Draw something."},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdraw(void)
{
	(void) Py_InitModule("draw", DrawMethods);
	import_array();
}
