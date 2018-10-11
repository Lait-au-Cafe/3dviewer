#include "viewer.hpp"

Viewer::Viewer(int const num_vertex, const char *const window_name){
	std::cout << "Initializing Viewer... " << std::endl;

    // set callback
    glfwSetErrorCallback(onError);

    // initialization
    if(glfwInit() != GL_TRUE){
        std::cerr << "Failed to initialize opengl. " << std::endl;
        exit(EXIT_FAILURE);
    }
	
	// glfw hints
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
    // create window
    window = glfwCreateWindow(
        1240 * 2, // width
        1240 * 2, // height
        window_name, // title
        NULL,
        NULL
    );
    if(window == NULL){
        std::cerr << "Failed to create a window. " << std::endl;
        exit(EXIT_FAILURE);
    }

    // create opengl context
    glfwMakeContextCurrent(window);

	// set background color (= default color)
	glClearColor(0.0, 0.0, 0.0, 1.0);

	//=========================================
	// Prepare Shader Programs
	//=========================================

	// create shader objects
    bool is_loaded = false;
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    is_loaded = loadShader(vertex_shader, "glsl/vertex.glsl");
	if(!is_loaded){ exit(EXIT_FAILURE); }

	GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    is_loaded = loadShader(frag_shader, "glsl/fragment.glsl");
	if(!is_loaded){ exit(EXIT_FAILURE); }

	// create shader program
	shader_program = glCreateProgram();

	// bind shader objects
	glAttachShader(shader_program, vertex_shader);
	glAttachShader(shader_program, frag_shader);
	glDeleteShader(vertex_shader);
	glDeleteShader(frag_shader);

	// link
	glLinkProgram(shader_program);
checkGLErrors();
	GLint result = GL_FALSE;
	glGetProgramiv(shader_program, GL_LINK_STATUS, &result);
checkGLErrors();
	if(result != GL_TRUE){
		GLsizei log_len = 0;
		GLchar log_msg[1024] = {};
		glGetProgramInfoLog(shader_program, 1024, &log_len, log_msg);

		std::cerr 
			<< "Failed to link shader program. \n"
			<< log_msg
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	//=========================================
	// Prepare Buffers
	//=========================================
	
	// Position Attribute
	// generate & bind buffer
	GLuint vertex_buffer;
	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

	// allocate memory
	this->num_vertex = num_vertex;
	GLint buf_size = num_vertex * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, buf_size, NULL, GL_DYNAMIC_DRAW);
checkGLErrors();

	GLint size_allocated = 0;
	glGetBufferParameteriv(
        GL_ARRAY_BUFFER, 
        GL_BUFFER_SIZE, 
        &size_allocated);
checkGLErrors();
	std::cout << "Requested OpenGL Buffer:"
		<< buf_size << "Byte" << std::endl;
	std::cout << "Allocated OpenGL Buffer:" 
		<< size_allocated << "Byte" << std::endl;
	if(size_allocated != buf_size){
		std::cerr 
			<< "Failed to allocate memory for buffer. " 
			<< std::endl;
		glDeleteBuffers(1, &vertex_buffer);
		exit(EXIT_FAILURE);
	}

	// bind to cuda resource
	cudaDeviceProp prop;
	int device;
	memset(&prop, 0, sizeof(cudaDeviceProp)); // 0 fill
	prop.major = 1; prop.minor = 0;
	checkCudaErrors(cudaChooseDevice(&device, &prop));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(
			&cuda_resource, 
			vertex_buffer,
			cudaGraphicsMapFlagsNone));

	// Color Attribute
	GLuint color_buffer;
	glGenBuffers(1, &color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, color_buffer);

	// bind to vertex array object
	glGenVertexArrays(1, &va_object);
checkGLErrors();
	glBindVertexArray(va_object);
checkGLErrors();
	// vertex
	glEnableVertexAttribArray(0);
checkGLErrors();
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
checkGLErrors();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
checkGLErrors();
	// color
	glEnableVertexAttribArray(1);
checkGLErrors();
	glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
checkGLErrors();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
checkGLErrors();

	// unbind
	glBindVertexArray(0);
checkGLErrors();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
checkGLErrors();

	glPointSize(8);
checkGLErrors();
}

bool Viewer::update(){
    // initialize
    glClear(GL_COLOR_BUFFER_BIT);
checkGLErrors();

    // bind program
    glUseProgram(shader_program);
checkGLErrors();

    // bind buffer
    glBindVertexArray(va_object);
checkGLErrors();

    // draw
    //glDrawArrays(GL_TRIANGLES, 0, 3);
    glDrawArrays(GL_POINTS, 0, num_vertex);
checkGLErrors();

	// unbind
	glBindVertexArray(0);
checkGLErrors();

    glfwSwapBuffers(window);
    glfwPollEvents();

    return (glfwWindowShouldClose(window) != GL_TRUE);
}

void Viewer::mapCudaResource(void** devPtr, size_t* size){
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(devPtr, size, cuda_resource));
}

void Viewer::unmapCudaResource(){
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, NULL));
}





void impl_checkGLErrors(const char *const file, int const line){
	GLenum err = glGetError();

	if(err == GL_NO_ERROR){
//		std::cout << "No GL Error is reported. " << std::endl;
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
		<< "OpenGL Error at " << file << ":" << line << "\n"
		<< msg.str()
		<< std::endl;
	
	return;
}

void impl_checkCudaErrors(cudaError_t err, const char *const file, int const line){
	if(err == cudaSuccess){
//		std::cout << "No CUDA Error is reported. " << std::endl;
		return;
	}

	std::stringstream msg;
	msg << "Error Code:" << err;

	std::cerr
		<< "CUDA error at " << file << ":" << line << "\n"
		<< msg.str()
		<< std::endl;
	cudaDeviceReset();
	exit(EXIT_FAILURE);
}
