// Cuda Tutorial Example
// From: https://devblogs.nvidia.com/even-easier-introduction-cuda/
//


#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <unordered_map>

//OpenGL Includes
#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OpenGL/gl.h>
#else
#include <windows.h>
#include "GL/glut.h"
#include <gl/gl.h>
#endif

using namespace std;

//Window diminsions
#define ImageX 1920
#define ImageY 1080
#define ImageZ 1000//1000

#define ImageSize (ImageX * ImageY)

//Grid diminsions
#define GridX 100
#define GridY 100
#define GridZ 100

#define GridSize (GridX * GridY)

#define MAX_UNIT_TIME 200
#define UNIT_TIME 0//16.67		//~60Hz

//-------------------------------Window Variables-------------------------------
int WINDOW_WIDTH = ImageX;
int WINDOW_HEIGHT = ImageY;
int WINDOW_DEPTH = ImageZ;

float Z_NEAR = 0.0001;
float Z_FAR = 1000;

int field_of_view = 90;
int unit_time = UNIT_TIME;
int max_unit_time = MAX_UNIT_TIME;

int fps_delay = 10;
int fps_min = 10000;
int fps_max = 0;
int fps_ct = 0;
int fps;

//-------------------------------Mouse Variables-------------------------------
int g_mouse_x, g_mouse_y, g_mouse_z;
bool mouse_left_pressed = false;
bool mouse_right_pressed = false;


//-------------------------------Draw Variables-------------------------------
bool toggle_simulation = true;
bool swap_buffers = true;

float alive_color[3] = { 1.0, 1.0, 1.0 };
float dead_color[3] = { 0.0, 0.0, 0.0 };


//-------------------------------Timer Struct-------------------------------
struct timer {
	chrono::high_resolution_clock::time_point t_start, t_end;

	void start() { t_start = chrono::high_resolution_clock::now(); }
	void end() { t_end = chrono::high_resolution_clock::now(); }

	float get_time() {
		chrono::duration<float> t = chrono::duration_cast<chrono::duration<float> >(t_end - t_start);
		return t.count();
	}
};

//-------------------------------Utility Variables-------------------------------
timer t;
string info_str = "";


//-------------------------------Object Data-------------------------------

float color[3][3] = { { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }, { 0.0, 1.0, 0.0 } };

//Framebuffer used to draw the image
float framebuffer[ImageY][ImageX][3];

//Generation buffers
int *gen_1;
int *gen_2;




//----------------------------------------OpenGL Functions----------------------------------------

int getValue(int *buf, int x, int y);
void set(int *buf, int x, int y, int value);
int getX(int idx);
int getY(int idx);
float getFPS();
void clearFramebuffer();
void setFramebuffer(int x, int y, float R, float G, float B);
int wrap(const int& limit, const int& arg);	//Enables pixel wrapping in the framebuffer. The world now acts as if it were on a sphere.
void setbuffer(bool *buf, int _x, int _y, int _state);
void drawit(void);
void display(void);
void mouseMove(int x, int y);
void mouseClick(int btn, int state, int x, int y);
void passiveMouseMove(int x, int y);
void specialKeyboard(int key, int x, int y);
void keyboard(unsigned char key, int x, int y);
void reshape(int width, int height);
void init(void);


//=========================================================================================================================
//---------------------------------------------------------Cuda------------------------------------------------------------
//=========================================================================================================================


#define NBLOCKS 1
#define NX 24
#define NY 16
#define NTHREAD (NX * NY)


//Convert from 2D to 1D index
__device__ int cuBufIdx(int x, int y)
{
    return (y * ImageX) + x;
}

//Wrap index values
__device__ int cuWrap(int limit, int idx)
{
    return (limit + idx) % limit;
}

// Kernel function to add the elements of two arrays
__global__ void sim( int *buf_1, int *buf_2)
{
	//int id = threadIdx.x + threadIdx.y * NX;
	int id = threadIdx.x + threadIdx.y * NX;
    int x, y, alive_ct;

    //Clear buf_2, prepare it for writing
    for (int i = id; i < ImageSize; i+= NTHREAD)
        buf_2[i] = 0;

    __syncthreads();

    //Write values to buf_2
    for (int i = id; i < ImageSize; i+= NTHREAD)
    {
        x = i % ImageX;
        y = (i - x) / ImageX;
        //idx = (y * ImageX) + x;

        alive_ct = 0;

        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x - 1), cuWrap(ImageY - 1, y - 1) ) ];
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x    ), cuWrap(ImageY - 1, y - 1) ) ];
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x + 1), cuWrap(ImageY - 1, y - 1) ) ];
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x - 1), cuWrap(ImageY - 1, y    ) ) ];
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x + 1), cuWrap(ImageY - 1, y    ) ) ]; 
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x - 1), cuWrap(ImageY - 1, y + 1) ) ];
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x    ), cuWrap(ImageY - 1, y + 1) ) ];
        alive_ct += buf_1[ cuBufIdx( cuWrap(ImageX - 1, x + 1), cuWrap(ImageY - 1, y + 1) ) ];

        //Apply conditions to current cell
        if (buf_1[i])
        {
            if (alive_ct < 2)
            {
                buf_2[i] = false;
                //population_ct--;
            }
            else if (alive_ct == 2 || alive_ct == 3)
            {
                buf_2[i] = true;
            }
            else if (alive_ct > 3)
            {
                buf_2[i] = false;
                //population_ct--;
            }
            else
            {
                buf_2[i] = true;
            }
        }
        else if (alive_ct == 3)
        {
            buf_2[i] = true;
            //population_ct++;
        }

    }
    __syncthreads();

    //Reset buf_1
    for (int i = id; i < ImageSize; i+= NTHREAD)
    {
        buf_1[i] = buf_2[i]; 
    }

	__syncthreads();
}


//=========================================================================================================================
//---------------------------------------------------------Main------------------------------------------------------------
//=========================================================================================================================

int main(int argc, char** argv)
{
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&gen_1, ImageSize*sizeof(int));
    cudaMallocManaged(&gen_2, ImageSize*sizeof(int));

    for (int i = 0; i < ImageSize; i++)
    {
        if(rand()%100 <= 10)
            gen_1[i] = 1;
    }
        

    sim<<<NBLOCKS, dim3(NX, NY)>>>(gen_1, gen_2);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    t.start();
    

//------------------------------------OpenGL-------------------------------------

    //----------------Create Window----------------
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Game of Life");
	init();

    glutMainLoop();
    

    //Clean up memory
    cudaFree(gen_1);
    cudaFree(gen_2);

    return 0;
}












//=========================================================================================================================
//-------------------------------------------------------OpenGL------------------------------------------------------------
//=========================================================================================================================




//=========================================================================================================================
//-----------------------------------------------------Draw Function-------------------------------------------------------
//=========================================================================================================================

// Draws the scene
void drawit(void)
{
    //Draws the pixel values from the framebuffer
	glDrawPixels(ImageX, ImageY, GL_RGB, GL_FLOAT, framebuffer);
	glFlush();
}

//Draw buffer
void drawBuf(int *buf)
{
    int x, y;
    for (int i = 0; i < ImageSize; i++) {
        x = getX(i);
        y = getY(i);

        framebuffer[y][x][0] = color[buf[i]][0];
        framebuffer[y][x][1] = color[buf[i]][1];
        framebuffer[y][x][2] = color[buf[i]][2];

		//framebuffer[y][x][0] = (buf[i]) ? alive_color[0] : dead_color[0];
		//framebuffer[y][x][1] = (buf[i]) ? alive_color[1] : dead_color[1];
		//framebuffer[y][x][2] = (buf[i]) ? alive_color[2] : dead_color[2];

	}

	drawit();

}


//=========================================================================================================================
//---------------------------------------------------Display Function------------------------------------------------------
//=========================================================================================================================

void display(void)
{
    fps_ct++;

	glClear(GL_COLOR_BUFFER_BIT);

    //info_str = "Generations: " + to_string(generation_ct) + "    Population: " + to_string(population_ct) + "    FPS: " + to_string(fps) + "    Min FPS: " + to_string(fps_min) + "    Draw Size: " + to_string(dot_size);
	info_str = "FPS: " + to_string(fps) + "    Min FPS: " + to_string(fps_min);
    glutSetWindowTitle(info_str.c_str());


    if(toggle_simulation)
    {    
        //Run kernel
        sim<<<NBLOCKS, dim3(NX, NY)>>>(gen_1, gen_2);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        drawBuf(gen_2);

    }


    if(fps_ct % fps_delay)
	{
		t.end();
		fps = (1000 / (t.get_time() * 1000));
		fps_min = ( fps < fps_min ) ? fps : fps_min;
		t.start();
	}

}





int getValue(int *buf, int x, int y)
{
    // id = (y * X_Dim) + x
    
    //  0 1 2
    //  3 4 5
    //  6 7 8

    return buf[ (y * ImageX) + x];
}

void set(int *buf, int x, int y, int value)
{
    buf[(y * ImageX) + x] = value;
}


int getX(int idx)
{
    return idx % ImageX;
}

int getY(int idx)
{
    return (idx - (idx % ImageX)) / ImageX;
}


float getFPS()
{
	return 0.0f;
}



//=========================================================================================================================
//-----------------------------------------------------Clear Frame Buffer--------------------------------------------------
//=========================================================================================================================

// Clears framebuffer to black
void clearFramebuffer()
{
	for (int i = 0; i < ImageSize; i++) {
        framebuffer[ getY(i) ][ getX(i) ][0] = 0.0;
        framebuffer[ getY(i) ][ getX(i) ][1] = 0.0;
        framebuffer[ getY(i) ][ getX(i) ][2] = 0.0;
    }
}

//=========================================================================================================================
//-----------------------------------------------------Set Frame Buffer----------------------------------------------------
//=========================================================================================================================

// Sets pixel x,y to the color RGB
void setFramebuffer(int x, int y, float R, float G, float B)
{
	// changes the origin from the lower-left corner to the upper-left corner
    y = ImageY - 1 - y;
    
    //At this point I am still unsure as to why the 'x' and 'y' must be reversed
    framebuffer[y][x][0] = R;
    framebuffer[y][x][1] = G;
    framebuffer[y][x][2] = B;

}


//Enables pixel wrapping in the framebuffer. The world now acts as if it were on a sphere.
int wrap(const int& limit, const int& arg)
{
	return (limit + arg) % limit;
}


void setbuffer(bool *buf, int _x, int _y, int _state)
{
	int x = wrap(ImageX - 1, _x);
	int y = wrap(ImageY - 1, _y);

    buf[(y * ImageX) + x] = _state;
	
}

//=========================================================================================================================
//----------------------------------------------------Mouse Movement-------------------------------------------------------
//=========================================================================================================================

//Draws when the mouse is clicked and dragged
void mouseMove(int x, int y)
{
	y = ImageY - 1 - y;
	//Compute mouse movement
	//float dx = (x - g_mouse_x);
	//float dy = (y - g_mouse_y);

	//-------------------------Left Mouse Button-------------------------------
	if (mouse_left_pressed) {
		
	}

	//-------------------------Right Mouse Button-------------------------------
	if (mouse_right_pressed) {

	}

	//Update previous mouse position
	g_mouse_x = x;
	g_mouse_y = y;
	
	glutPostRedisplay();
}

//=========================================================================================================================
//-----------------------------------------------------Mouse Click---------------------------------------------------------
//=========================================================================================================================

//Draws when the mouse button is clicked
void mouseClick(int btn, int state, int x, int y)
{
	y = ImageY - 1 - y;
	//Update mouse position
	g_mouse_x = x;
	g_mouse_y = y;

	//-------------------------Left Mouse Button-------------------------------
	if ( btn == GLUT_LEFT_BUTTON ) {

		//Toggle mouse button pressed
		if (state == GLUT_DOWN) 
        {
			mouse_left_pressed = true;

			
		}
		else    //Reset button state
        {	
			mouse_left_pressed = false;

		}
	}

	//-------------------------Right Mouse Button-------------------------------
	if ( btn == GLUT_RIGHT_BUTTON ) 
    {	
		//Toggle mouse button pressed
		if (state == GLUT_DOWN) {
			mouse_right_pressed = true;
		}
		else {	//Reset button state
			mouse_right_pressed = false;
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------

	if (btn == 3) {		//Scroll wheel up
		
	}

	//-------------------------------------------------------------------------------------------------------------------------

	if (btn == 4) {		//Scroll wheel down
		
	}

	//glutPostRedisplay();
}

//=========================================================================================================================
//------------------------------------------------Passive Mouse Function---------------------------------------------------
//=========================================================================================================================

void passiveMouseMove(int x, int y)
{
	y = ImageY - 1 - y;
	//Update mouse position
	g_mouse_x = x;
	g_mouse_y = y;

	//glutPostRedisplay();
}


//=========================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------
//=========================================================================================================================

void specialKeyboard(int key, int x, int y)
{
	switch (key)
	{
		case GLUT_KEY_RIGHT: {
			
			break;
		}
		
		//-------------------------------------------------------------------------------------------------------------------------

		case GLUT_KEY_LEFT: {
			
			break;
		}
		
		//-------------------------------------------------------------------------------------------------------------------------

		case GLUT_KEY_UP: {

			break;
		}

		//-------------------------------------------------------------------------------------------------------------------------

		case GLUT_KEY_DOWN: {	

			break;
		}

		//-------------------------------------------------------------------------------------------------------------------------
		
		case GLUT_KEY_F1: {		//Toggle GameMode

			break;
		}

		//-------------------------------------------------------------------------------------------------------------------------
		
		case GLUT_KEY_F4: {
			exit(0);
			break;
		}
	}

	//glutPostRedisplay();
}

//=========================================================================================================================
//--------------------------------------------------Keyboard Function------------------------------------------------------
//=========================================================================================================================

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
		case '1': {		//Change material | Green
			drawBuf(gen_1);
        
			break;
		}

		case '2': {		//Change material | Blue
			drawBuf(gen_2);

			break;
		}

		case '3': {		//Change material | Red
			
			break;
		}

		

		//---------------------------------Draw Style Options---------------------------------

		case 32: {		
            toggle_simulation = !toggle_simulation;

			break;
		}

		case 'p': {		
			
			break;
		}

		case 'c': {		

			break;
		}

		case 'r': {		//Assign each pixel to a random color

			break;
		}

		//-------------------------------------------------------------------------------------------------------------------------
		
		case '_':
		case '-': {
			
			break;
		}

		//-------------------------------------------------------------------------------------------------------------------------
		
		case '=':
		case '+': {

			break;
		}

		//-------------------------------------------------------------------------------------------------------------------------
	
		default: {
            printf("Key: %c\n", key);
			break;
		}

	}
	//glutPostRedisplay();
}

//=========================================================================================================================
//----------------------------------------------------Reshape Function-----------------------------------------------------
//=========================================================================================================================

void reshape(int width, int height)
{
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	gluPerspective(field_of_view, float(WINDOW_WIDTH) / WINDOW_HEIGHT, Z_NEAR, Z_FAR);
	glOrtho(-1 * WINDOW_WIDTH / 2.0f, WINDOW_WIDTH / 2.0f, -1 * WINDOW_HEIGHT / 2.0f, WINDOW_HEIGHT / 2.0f, -1 * WINDOW_DEPTH / 2.0f, WINDOW_DEPTH / 2.0f);
	glMatrixMode(GL_MODELVIEW);
}

void updateFrameTimer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(unit_time, updateFrameTimer, 0);
}

//=========================================================================================================================
//-----------------------------------------------------Init Function-------------------------------------------------------
//=========================================================================================================================

void init(void)
{
	
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	gluPerspective(field_of_view, float(WINDOW_WIDTH) / WINDOW_HEIGHT, Z_NEAR, Z_FAR);
	glOrtho(-1 * WINDOW_WIDTH / 2.0f, WINDOW_WIDTH / 2.0f, -1 * WINDOW_HEIGHT / 2.0f, WINDOW_HEIGHT / 2.0f, -1 * WINDOW_DEPTH / 2.0f, WINDOW_DEPTH / 2.0f);
	glMatrixMode(GL_MODELVIEW);
	

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMotionFunc(mouseMove);
	glutMouseFunc(mouseClick);
	glutPassiveMotionFunc(passiveMouseMove);
	glutReshapeFunc(reshape);
	glutTimerFunc(unit_time, updateFrameTimer, 0);

}
