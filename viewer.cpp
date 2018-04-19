#include "viewer.hpp"

void Viewer::init(){
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
        "OpenGL Sample", // title
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
	GLint buf_size = 9 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, buf_size, points, GL_STATIC_DRAW);

	GLint size_allocated = 0;
	glGetBufferParameteriv(
        GL_ARRAY_BUFFER, 
        GL_BUFFER_SIZE, 
        &size_allocated);

	if(size_allocated != buf_size){
		std::cerr << "Failed to allocate memory for buffer. " << std::endl;
		glDeleteBuffers(1, &vertex_buffer);
		exit(EXIT_FAILURE);
	}

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
