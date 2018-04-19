#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iterator>

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

class Viewer {
public:
    void init();
    bool update();

private:
    GLFWwindow* window;
    GLuint shader_program;
    GLuint va_object;

//    static bool loadShader(GLuint, const char*);
//    static void onError(int, const char*);
//    static void checkGLError();

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
            << "GLFW Error  Code:" << err << "\n"
            << msg
            << std::endl;
        return;
    }

    static void checkGLError(){
        GLenum err = glGetError();

        if(err == GL_NO_ERROR){
            std::cerr << "No error is reported. " << std::endl;
            return;
        }

        std::stringstream msg;
        switch(err){
            case GL_INVALID_ENUM:
                msg << "An unacceptable value is specified"
                    << " for an enumerated argument. ";
                break;
            case GL_INVALID_VALUE:
                msg << "A numeric argument is out of range. ";
                break;
            case GL_INVALID_OPERATION:
                msg << "The specified operation is not allowed"
                    << " in the current state. ";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                msg << "The framebuffer object is not complete.";
                break;
            case GL_OUT_OF_MEMORY:
                msg << "There is not enough memory left"
                    << " to execute the command. ";
                break;
            case GL_STACK_UNDERFLOW:
                msg << "An attempt has been made to perform an operation"
                    << " that would cause an internal stack to underflow.";
                break;
            case GL_STACK_OVERFLOW:
                msg << "An attempt has been made to perform an operation"
                    << " that would cause an internal stack to overflow.";
                break;
            default:
                msg << "Unknown error id";
        }

        std::cerr 
            << "OpenGL Error at " << __FILE__ << ":" << __LINE__ << "\n"
            << msg.str()
            << std::endl;
        
        return;
    }

};
