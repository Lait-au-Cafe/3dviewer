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

    // create window
    window = glfwCreateWindow(
        640, // width
        480, // height
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
	GLint result = GL_FALSE;
	glGetProgramiv(shader_program, GL_LINK_STATUS, &result);
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
	
	float points[] = {
		0.0f, 0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};

	// generate & bind buffer
	GLuint vertex_buffer;
	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

	// allocate memory
	GLint buf_size = num_vertex * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, buf_size, points, GL_STATIC_DRAW);

	GLint size_allocated = 0;
	glGetBufferParameteriv(
        GL_ARRAY_BUFFER, 
        GL_BUFFER_SIZE, 
        &size_allocated);
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

	// bind to vertex array object
	glGenVertexArrays(1, &va_object);
	glBindVertexArray(va_object);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	// unbind
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool Viewer::update(){
    // initialize
    glClear(GL_COLOR_BUFFER_BIT);

    // bind program
    glUseProgram(shader_program);

    // bind buffer
    glBindVertexArray(va_object);

    // draw
    glDrawArrays(GL_TRIANGLES, 0, 3);

	// unbind
	glBindVertexArray(0);

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
