#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iterator>
#include <string.h>

// OpenGL includes
#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

// CUDA Includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


class Viewer {
public:
    Viewer(int const, const char *const);
    bool update();
	void mapCudaResource(void**, size_t*);
	void unmapCudaResource();

private:
    GLFWwindow* window;
    GLuint shader_program;
    GLuint va_object;

	cudaGraphicsResource* cuda_resource;

    static bool loadShader(GLuint shader_id, const char* filename){
        // load shader file
        std::ifstream shader_file(filename);
        if(shader_file.fail()){
            std::cerr
                << "Failed to load file: "
                << filename
                << std::endl;
            return false;
        }

        // interpret as chars
        std::string shader_code(
            (std::istreambuf_iterator<char>(shader_file)),
            std::istreambuf_iterator<char>());

        // bind shader code
        const char *source = shader_code.c_str();
        //std::cout << source << std::endl;
        glShaderSource(shader_id, 1, &source, NULL);

        // compile shader code
        glCompileShader(shader_id);
        GLint result = GL_FALSE;
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
        if(result != GL_TRUE){
            GLsizei log_len = 0;
            GLchar log_msg[1024] = {};
            glGetShaderInfoLog(shader_id, 1024, &log_len, log_msg);
            
            std::cerr 
                << "Failed to compiler shader: "
                << filename << "\n"
                << log_msg
                << std::endl;
            return false;
        }

        return true;
    }

    static void onError(int err, const char* msg){
        std::cerr 
            << "GLFW Error at " << __FILE__ << ":" << __LINE__
			<< " Code:" << err << "\n"
            << msg
            << std::endl;
        return;
    }
};

#ifndef checkCudaErrors
#define checkCudaErrors(val) impl_checkCudaErrors((val), __FILE__, __LINE__)
#endif
void impl_checkCudaErrors(cudaError_t, const char *const, int const);

#ifndef checkGLErrors
#define checkGLErrors() impl_checkGLErrors(__FILE__, __LINE__)
#endif
void impl_checkGLErrors(const char *const, int const);
